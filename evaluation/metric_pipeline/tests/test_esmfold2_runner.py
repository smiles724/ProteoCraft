from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from eval_pipeline import esmfold2_runner
from eval_pipeline.cli import (
    DEFAULT_EVALUATOR,
    LEGACY_ESMFOLD_EVALUATOR,
    _build_argparser,
    evaluate_one_esmfold2,
)


class FakeBackend:
    def __init__(self):
        self.config = SimpleNamespace(
            target_chain_roles=None,
            verifier_model_name="biohub/ESMFold2",
            verifier_num_loops=20,
            verifier_num_sampling_steps=200,
            verifier_loss_mode="aligned_common",
        )
        self.calls = []

    def verify(self, **kwargs):
        self.calls.append((kwargs, self.config.target_chain_roles))
        return SimpleNamespace(
            total_loss=0.18,
            summary_confidence={
                "pair_chain_iptm": 0.82,
                "iptm": 0.91,
                "ptm": 0.88,
                "plddt": 0.86,
                "binder_plddt": 0.89,
                "binder_rg_excess": 0.02,
                "binder_radius_gyration": 14.2,
                "hotspot_mean_min_heavy_atom_distance": 6.1,
                "hotspot_heavy_atom_coverage": 0.75,
                "hotspot_heavy_atom_coverage_5a": 0.50,
                "mmcif": "data_test\n#\n",
            },
        )


def _row():
    return {
        "method": "BoltzGen",
        "case": "TCR_CDR",
        "paper_case": "TCR_CDR",
        "design_id": "d1",
        "target_pdb_path": "/tmp/target.pdb",
        "target_chain": "A+B",
        "target_seq": "AAAA:CCCC",
        "binder_seq": "YYYY",
        "designed_complex_pdb_path": "/tmp/design.cif",
        "target_chain_in_complex": "A+B",
        "binder_chain_in_complex": "C",
        "target_chain_roles": "target+scaffold",
        "target_hotspot_concat_indices": "[1, 3]",
        "n_generated_trajectories": 20,
        "n_mpnn_attempted": -1,
        "n_accepted_by_method": 1,
        "candidate_rank": 0,
        "source_filter_status": "generated",
        "verifier_calls": -1,
        "design_gpu_seconds": 12.0,
    }


def test_cli_defaults_to_current_esmfold2():
    args = _build_argparser().parse_args(
        ["--manifest", "m.csv", "--out-dir", "/tmp/out"]
    )
    assert args.evaluator == DEFAULT_EVALUATOR
    assert args.esmfold2_num_loops == 20
    assert args.esmfold2_num_sampling_steps == 200
    assert args.esmfold2_seed == 43
    assert args.esmfold2_loss_mode == "aligned_common"


def test_cli_accepts_timestamped_legacy_name_and_alias():
    parser = _build_argparser()
    timestamped = parser.parse_args(
        [
            "--manifest",
            "m.csv",
            "--out-dir",
            "/tmp/out",
            "--evaluator",
            LEGACY_ESMFOLD_EVALUATOR,
        ]
    )
    alias = parser.parse_args(
        [
            "--manifest",
            "m.csv",
            "--out-dir",
            "/tmp/out",
            "--evaluator",
            "esmfold",
        ]
    )
    assert timestamped.evaluator == "esmfold_202605"
    assert alias.evaluator == "esmfold"


def test_parse_multichain_roles_and_hotspots():
    assert esmfold2_runner.parse_target_sequences("AAAA:CCCC") == ["AAAA", "CCCC"]
    assert esmfold2_runner.parse_target_chain_roles("target+scaffold", 2) == (
        "target",
        "scaffold",
    )
    assert esmfold2_runner.parse_hotspot_indices("[3, 1, 3]") == [3, 1]
    with pytest.raises(ValueError, match="length must match"):
        esmfold2_runner.parse_target_chain_roles("target", 2)


def test_current_evaluator_maps_metrics_and_saves_mmcif(tmp_path: Path):
    backend = FakeBackend()
    out = evaluate_one_esmfold2(_row(), tmp_path, backend, seed=43)

    assert out["error_msg"] == ""
    assert out["evaluator"] == "esmfold2"
    assert out["eval_n_successes"] == 1
    assert out["pair_chain_iptm"] == pytest.approx(0.82)
    assert out["iptm"] == pytest.approx(0.91)
    assert out["ptm"] == pytest.approx(0.88)
    assert out["complex_pLDDT"] == pytest.approx(86.0)
    assert out["binder_pLDDT"] == pytest.approx(89.0)
    assert out["esmfold2_total_loss"] == pytest.approx(0.18)
    assert out["esmfold2_loss_mode"] == "aligned_common"
    assert Path(out["esmfold2_mmcif_path"]).read_text() == "data_test\n#\n"
    kwargs, roles = backend.calls[0]
    assert kwargs["target_seq"] == ["AAAA", "CCCC"]
    assert kwargs["hotspot_params"] == {"target_hotspot_concat_indices": [1, 3]}
    assert roles == ("target", "scaffold")
    assert backend.config.target_chain_roles is None
