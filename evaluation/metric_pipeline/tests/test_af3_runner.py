"""Tests for af3_runner.py."""
from __future__ import annotations

import json
import subprocess

import pytest


def test_build_clean_segments_chain_ids_AB():
    from eval_pipeline.af3_runner import build_clean_segments_af3

    segs = build_clean_segments_af3(
        binder_seq="MSKGEELFT",
        target_seqs=["PEPTIDE", "ANOTHER"],
    )
    assert len(segs) == 3
    assert segs[0]["kind"] == "binder"
    assert segs[0]["pdb_chain_id"] == "A"
    assert segs[0]["clean_start"] == 0
    assert segs[0]["clean_end"] == 9
    assert segs[0]["dirty_start"] == 0
    assert segs[0]["dirty_end"] == 9
    assert segs[1]["kind"] == "target_chain_0"
    assert segs[1]["pdb_chain_id"] == "B"
    assert segs[1]["clean_start"] == 9
    assert segs[1]["clean_end"] == 16
    assert segs[2]["kind"] == "target_chain_1"
    assert segs[2]["pdb_chain_id"] == "C"
    assert segs[2]["clean_start"] == 16
    assert segs[2]["clean_end"] == 23


def test_build_af3_input_json_schema():
    from eval_pipeline.af3_runner import build_af3_input_json

    obj = build_af3_input_json(
        job_name="BetV1_h23__d_001",
        target_seqs=["PEPTIDETARGET"],
        binder_seq="MSKGEELFT",
        target_msa_paths=["/path/to/target_chain_B.a3m"],
        seed=42,
    )
    assert obj["name"] == "BetV1_h23__d_001"
    assert obj["modelSeeds"] == [42]
    assert obj["dialect"] == "alphafold3"
    assert obj["version"] == 4
    assert len(obj["sequences"]) == 2
    binder_entry = obj["sequences"][0]["protein"]
    assert binder_entry["id"] == "A"
    assert binder_entry["sequence"] == "MSKGEELFT"
    assert binder_entry["unpairedMsa"] == ""
    assert binder_entry["templates"] == []
    target_entry = obj["sequences"][1]["protein"]
    assert target_entry["id"] == "B"
    assert target_entry["sequence"] == "PEPTIDETARGET"
    assert target_entry["unpairedMsaPath"] == "/path/to/target_chain_B.a3m"
    assert target_entry["templates"] == []


def test_build_af3_input_json_multi_chain():
    from eval_pipeline.af3_runner import build_af3_input_json

    obj = build_af3_input_json(
        job_name="TCR_1QRN__d_002",
        target_seqs=["AAA", "BBB", "CCC"],
        binder_seq="XXX",
        target_msa_paths=["/p/B.a3m", "/p/C.a3m", "/p/D.a3m"],
    )
    ids = [e["protein"]["id"] for e in obj["sequences"]]
    assert ids == ["A", "B", "C", "D"]
    seqs = [e["protein"]["sequence"] for e in obj["sequences"]]
    assert seqs == ["XXX", "AAA", "BBB", "CCC"]


def test_build_af3_input_json_msa_count_mismatch_raises():
    from eval_pipeline.af3_runner import build_af3_input_json

    with pytest.raises(ValueError, match="target_msa_paths"):
        build_af3_input_json(
            job_name="X",
            target_seqs=["A", "B"],
            binder_seq="Z",
            target_msa_paths=["/only_one.a3m"],
        )


def test_input_json_validates_against_af3_loader():
    pytest.importorskip("alphafold3")
    from alphafold3.common.folding_input import Input
    from eval_pipeline.af3_runner import build_af3_input_json

    obj = build_af3_input_json(
        job_name="smoke_test",
        target_seqs=["MEPEEPVK"],
        binder_seq="MSKGEELFT",
        target_msa_paths=["/dev/null"],
    )
    Input.from_json(json.dumps(obj))


def _make_tiny_structure_with_chains(chain_ids: list[str]):
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure

    structure = Structure("test")
    model = Model(0)
    structure.add(model)
    for i, cid in enumerate(chain_ids):
        chain = Chain(cid)
        model.add(chain)
        res = Residue((" ", 1, " "), "ALA", "")
        res.add(Atom("N", (i * 5.0, 0.0, 0.0), 80.0, 1.0, " ", " N  ", 1, "N"))
        res.add(Atom("CA", (i * 5.0 + 1.0, 0.0, 0.0), 80.0, 1.0, " ", " CA ", 2, "C"))
        chain.add(res)
    return structure


def _pdb_atom_line(serial: int, atom: str, resseq: int, x: float, y: float, z: float,
                   bfactor: float = 80.0, chain_id: str = "A") -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s} ALA {chain_id}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           C\n"
    )


def test_mmcif_to_pdb_roundtrip_plain(tmp_path):
    from Bio.PDB import MMCIFIO, PDBParser
    from eval_pipeline.af3_runner import mmcif_to_pdb

    structure = _make_tiny_structure_with_chains(["A", "B"])
    cif_path = tmp_path / "test.cif"
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(cif_path))

    pdb_out = tmp_path / "out.pdb"
    mmcif_to_pdb(cif_path, pdb_out, decompress_first=False)

    assert pdb_out.exists()
    assert pdb_out.stat().st_size > 0
    out_struct = PDBParser(QUIET=True).get_structure("o", str(pdb_out))
    out_chains = {c.id for m in out_struct for c in m}
    assert {"A", "B"} <= out_chains


def test_mmcif_zst_decompression(tmp_path):
    pytest.importorskip("zstandard")
    import zstandard as zstd
    from Bio.PDB import MMCIFIO
    from eval_pipeline.af3_runner import mmcif_to_pdb

    structure = _make_tiny_structure_with_chains(["A", "B"])
    cif_path = tmp_path / "test.cif"
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(cif_path))

    cif_zst = tmp_path / "test.cif.zst"
    cif_zst.write_bytes(zstd.ZstdCompressor().compress(cif_path.read_bytes()))

    pdb_out = tmp_path / "out.pdb"
    mmcif_to_pdb(cif_zst, pdb_out, decompress_first=True)
    assert pdb_out.exists()
    assert pdb_out.stat().st_size > 0


def test_extract_residue_plddt_from_pdb_unit_check_high(tmp_path):
    from eval_pipeline.af3_runner import extract_residue_plddt_from_pdb

    pdb_content = (
        "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 200.00           N\n"
        "ATOM      2  CA  ALA A   1      11.104  13.207  10.000  1.00 200.00           C\n"
        "TER\nEND\n"
    )
    p = tmp_path / "bad.pdb"
    p.write_text(pdb_content)
    segs = [{"kind": "binder", "pdb_chain_id": "A", "clean_start": 0,
             "clean_end": 1, "dirty_start": 0, "dirty_end": 1, "length": 1}]
    with pytest.raises(AssertionError, match="plddt"):
        extract_residue_plddt_from_pdb(p, segs)


def test_extract_residue_plddt_from_pdb_low_confidence_OK(tmp_path):
    from eval_pipeline.af3_runner import extract_residue_plddt_from_pdb

    pdb_content = (
        "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00 30.00           N\n"
        "ATOM      2  CA  ALA A   1      11.104  13.207  10.000  1.00 30.00           C\n"
        "TER\nEND\n"
    )
    p = tmp_path / "low.pdb"
    p.write_text(pdb_content)
    segs = [{"kind": "binder", "pdb_chain_id": "A", "clean_start": 0,
             "clean_end": 1, "dirty_start": 0, "dirty_end": 1, "length": 1}]
    plddt = extract_residue_plddt_from_pdb(p, segs)
    assert plddt.shape == (1,)
    assert abs(plddt[0] - 30.0) < 0.01


def test_extract_residue_plddt_segment_length_mismatch_raises(tmp_path):
    from eval_pipeline.af3_runner import extract_residue_plddt_from_pdb

    pdb_content = (
        _pdb_atom_line(1, "CA", 1, 11.0, 13.0, 10.0)
        + _pdb_atom_line(2, "CA", 2, 12.0, 13.0, 10.0)
        + "TER\nEND\n"
    )
    p = tmp_path / "two_res.pdb"
    p.write_text(pdb_content)
    segs = [{"kind": "binder", "pdb_chain_id": "A", "clean_start": 0,
             "clean_end": 5, "dirty_start": 0, "dirty_end": 5, "length": 5}]
    with pytest.raises(ValueError, match="2 residues, expected 5"):
        extract_residue_plddt_from_pdb(p, segs)


def test_extract_residue_plddt_missing_chain_raises(tmp_path):
    from eval_pipeline.af3_runner import extract_residue_plddt_from_pdb

    pdb_content = (
        _pdb_atom_line(1, "CA", 1, 11.0, 13.0, 10.0)
        + "TER\nEND\n"
    )
    p = tmp_path / "one_chain.pdb"
    p.write_text(pdb_content)
    segs = [{"kind": "binder", "pdb_chain_id": "Z", "clean_start": 0,
             "clean_end": 1, "dirty_start": 0, "dirty_end": 1, "length": 1}]
    with pytest.raises(ValueError, match="chain 'Z' missing"):
        extract_residue_plddt_from_pdb(p, segs)


def test_extract_residue_plddt_nonmonotonic_resseq_raises(tmp_path):
    """SF-15: residue ids must be exactly 1..N in clean segment order."""
    from eval_pipeline.af3_runner import extract_residue_plddt_from_pdb

    pdb_content = (
        _pdb_atom_line(1, "CA", 10, 11.0, 13.0, 10.0)
        + _pdb_atom_line(2, "CA", 11, 12.0, 13.0, 10.0)
        + "TER\nEND\n"
    )
    p = tmp_path / "offset_resseq.pdb"
    p.write_text(pdb_content)
    segs = [{"kind": "binder", "pdb_chain_id": "A", "clean_start": 0,
             "clean_end": 2, "dirty_start": 0, "dirty_end": 2, "length": 2}]
    with pytest.raises(ValueError, match="resseq order"):
        extract_residue_plddt_from_pdb(p, segs)


def test_extract_pae_from_confidences_json(tmp_path):
    from eval_pipeline.af3_runner import extract_pae_from_confidences_json

    obj = {
        "atom_chain_ids": ["A", "A", "B", "B"],
        "atom_plddts": [50.0, 50.0, 60.0, 60.0],
        "contact_probs": [[0.0] * 4] * 4,
        "pae": [
            [0.0, 1.0, 5.0, 6.0],
            [1.0, 0.0, 5.0, 6.0],
            [5.0, 5.0, 0.0, 1.0],
            [6.0, 6.0, 1.0, 0.0],
        ],
        "token_chain_ids": ["A", "A", "B", "B"],
        "token_res_ids": [1, 2, 1, 2],
    }
    p = tmp_path / "conf.json"
    p.write_text(json.dumps(obj))
    pae, token_chain_ids, token_res_ids = extract_pae_from_confidences_json(
        p, is_compressed=False
    )
    assert pae.shape == (4, 4)
    assert token_chain_ids == ["A", "A", "B", "B"]
    assert token_res_ids == [1, 2, 1, 2]


def test_extract_pae_missing_token_ids_raises(tmp_path):
    """SF-14: missing token id arrays must raise instead of returning empty lists."""
    from eval_pipeline.af3_runner import extract_pae_from_confidences_json

    obj = {
        "atom_chain_ids": ["A", "A"],
        "atom_plddts": [80.0, 80.0],
        "contact_probs": [[0.0, 0.0], [0.0, 0.0]],
        "pae": [[0.0, 5.0], [5.0, 0.0]],
    }
    p = tmp_path / "conf.json"
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="token id arrays length mismatch"):
        extract_pae_from_confidences_json(p, is_compressed=False)


def test_extract_pae_unit_check_raises_on_huge_value(tmp_path):
    from eval_pipeline.af3_runner import extract_pae_from_confidences_json

    obj = {
        "atom_chain_ids": ["A"],
        "atom_plddts": [50.0],
        "contact_probs": [[0.0]],
        "pae": [[200.0]],
        "token_chain_ids": ["A"],
        "token_res_ids": [1],
    }
    p = tmp_path / "conf.json"
    p.write_text(json.dumps(obj))
    with pytest.raises(AssertionError, match="PAE max"):
        extract_pae_from_confidences_json(p, is_compressed=False)


def test_load_native_summary_full_keys(tmp_path):
    from eval_pipeline.af3_runner import load_native_summary

    obj = {
        "ranking_score": 0.8,
        "ptm": 0.85,
        "iptm": 0.7,
        "has_clash": False,
        "fraction_disordered": 0.05,
        "chain_ptm": [0.85, 0.85],
        "chain_iptm": [0.7, 0.7],
        "chain_pair_iptm": [[0.0, 0.7], [0.7, 0.0]],
        "chain_pair_pae_min": [[0.0, 5.0], [5.0, 0.0]],
    }
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(obj))
    out = load_native_summary(p)
    assert out["ranking_score"] == 0.8
    assert out["ptm"] == 0.85
    assert out["iptm"] == 0.7
    assert out["has_clash"] == 0.0
    assert out["fraction_disordered"] == 0.05


def test_load_native_summary_missing_keys_defaults_NaN(tmp_path):
    from eval_pipeline.af3_runner import load_native_summary
    import math

    p = tmp_path / "minimal.json"
    p.write_text(json.dumps({"ranking_score": 0.5}))
    out = load_native_summary(p)
    assert out["ranking_score"] == 0.5
    assert math.isnan(out["ptm"])
    assert math.isnan(out["iptm"])
    assert math.isnan(out["has_clash"])
    assert math.isnan(out["fraction_disordered"])


def test_find_top_ranked_outputs_with_sanitised_subdir(tmp_path):
    from eval_pipeline.af3_runner import find_top_ranked_outputs

    sub_run_out = tmp_path / "BetV1" / "BetV1_h23"
    inner = sub_run_out / "BetV1_h23__d_001"
    inner.mkdir(parents=True)
    (inner / "BetV1_h23__d_001_model.cif").write_text("# fake cif\n")
    (inner / "BetV1_h23__d_001_summary_confidences.json").write_text("{}")
    (inner / "BetV1_h23__d_001_confidences.json").write_text("{}")

    paths = find_top_ranked_outputs(sub_run_out, "BetV1_h23__d_001")
    assert paths["model_cif"].name == "BetV1_h23__d_001_model.cif"
    assert paths["summary_json"].exists()
    assert paths["confidences_json"].exists()
    assert paths["is_compressed"] is False


def test_find_top_ranked_outputs_zst(tmp_path):
    from eval_pipeline.af3_runner import find_top_ranked_outputs

    sub_run_out = tmp_path / "X" / "sub"
    inner = sub_run_out / "job_xx"
    inner.mkdir(parents=True)
    (inner / "job_xx_model.cif.zst").write_bytes(b"\x28\xb5\x2f\xfd\x00\x00\x00\x00")
    (inner / "job_xx_confidences.json.zst").write_bytes(b"\x28\xb5\x2f\xfd")
    (inner / "job_xx_summary_confidences.json").write_text("{}")

    paths = find_top_ranked_outputs(sub_run_out, "job_xx")
    assert paths["is_compressed"] is True
    assert paths["model_cif"].name.endswith(".cif.zst")
    assert paths["confidences_json"].name.endswith(".json.zst")
    assert paths["summary_json"].name == "job_xx_summary_confidences.json"


def test_find_top_ranked_outputs_missing_raises(tmp_path):
    from eval_pipeline.af3_runner import find_top_ranked_outputs

    sub_run_out = tmp_path / "empty"
    sub_run_out.mkdir()
    with pytest.raises(FileNotFoundError):
        find_top_ranked_outputs(sub_run_out, "missing_job")


def test_find_top_ranked_outputs_does_not_match_prefix_sibling(tmp_path):
    """N-1: fallback lookup must not match prefix-overlapping sibling outputs."""
    from eval_pipeline.af3_runner import find_top_ranked_outputs

    sub_run_out = tmp_path / "out"
    sibling = sub_run_out / "job_extra"
    sibling.mkdir(parents=True)
    (sibling / "job_extra_model.cif").write_text("# fake cif\n")
    (sibling / "job_extra_summary_confidences.json").write_text("{}")
    (sibling / "job_extra_confidences.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="job 'job'"):
        find_top_ranked_outputs(sub_run_out, "job")


def test_sanitize_af3_job_name_matches_af3_drop_rule():
    """NIT2-2: local output lookup must match AF3's sanitised_name behavior."""
    from eval_pipeline.af3_runner import _sanitize_af3_job_name

    assert _sanitize_af3_job_name("PDL1@d 1-v2.test") == "PDL1d_1v2test"


def test_stage_inputs_for_sub_run(tmp_path):
    from eval_pipeline.af3_runner import stage_inputs_for_sub_run

    rows = [
        {"design_id": "PDL1_d_001", "case": "PDL1", "method": "BCv1",
         "target_seq": "PEPTIDE", "binder_seq": "MSKG"},
        {"design_id": "PDL1_d_002", "case": "PDL1", "method": "BCv1",
         "target_seq": "PEPTIDE", "binder_seq": "AAAA"},
    ]
    staging = tmp_path / "staging" / "PDL1" / "PDL1"
    msa_dir = tmp_path / "msa" / "PDL1"
    msa_dir.mkdir(parents=True)
    (msa_dir / "target_chain_B.a3m").write_text(">seq\nPEPTIDE\n")

    staged = stage_inputs_for_sub_run(rows=rows, staging_dir=staging, target_msa_dir=msa_dir)

    assert len(staged) == 2
    for s in staged:
        assert s["json_path"].exists()
        assert s["job_name"].startswith("PDL1__")
    obj = json.loads(staged[0]["json_path"].read_text())
    assert obj["dialect"] == "alphafold3"
    assert obj["sequences"][0]["protein"]["sequence"] == "MSKG"
    assert obj["sequences"][1]["protein"]["unpairedMsaPath"].endswith("target_chain_B.a3m")


def test_stage_inputs_for_sub_run_missing_msa_raises(tmp_path):
    from eval_pipeline.af3_runner import stage_inputs_for_sub_run

    rows = [{"design_id": "x", "case": "PDL1", "method": "BCv1",
             "target_seq": "PEPTIDE", "binder_seq": "X"}]
    staging = tmp_path / "staging"
    msa_dir = tmp_path / "missing_msa"
    msa_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="target_chain_B.a3m"):
        stage_inputs_for_sub_run(rows=rows, staging_dir=staging, target_msa_dir=msa_dir)


def test_cleanup_for_rerun(tmp_path):
    import pandas as pd
    from eval_pipeline.af3_runner import cleanup_for_rerun

    staging = tmp_path / "staging"
    output = tmp_path / "af3_outputs"
    staging.mkdir()
    output.mkdir()
    for design_id in ["rerun", "keep"]:
        job_name = f"PD1__{design_id}"
        (staging / f"{job_name}.json").write_text("{}")
        job_out = output / job_name
        job_out.mkdir()
        (job_out / "stale.txt").write_text("old")

    rows_to_rerun = pd.DataFrame([{"case": "PD1", "design_id": "rerun"}])
    cleanup_for_rerun(staging, output, rows_to_rerun)

    assert not (staging / "PD1__rerun.json").exists()
    assert not (output / "PD1__rerun").exists()
    assert (staging / "PD1__keep.json").exists()
    assert (output / "PD1__keep" / "stale.txt").exists()


def test_cleanup_for_rerun_accepts_list_of_dicts(tmp_path):
    """SF-13: cleanup_for_rerun must remove stale files for list[dict] rows."""
    from eval_pipeline.af3_runner import cleanup_for_rerun

    staging = tmp_path / "staging"
    output = tmp_path / "af3_outputs"
    staging.mkdir()
    output.mkdir()
    for design_id in ["rerun", "keep"]:
        job_name = f"PD1__{design_id}"
        (staging / f"{job_name}.json").write_text("{}")
        job_out = output / job_name
        job_out.mkdir()
        (job_out / "stale.txt").write_text("old")

    cleanup_for_rerun(staging, output, [{"case": "PD1", "design_id": "rerun"}])

    assert not (staging / "PD1__rerun.json").exists()
    assert not (output / "PD1__rerun").exists()
    assert (staging / "PD1__keep.json").exists()
    assert (output / "PD1__keep" / "stale.txt").exists()


def test_cleanup_for_rerun_full_clean_with_none(tmp_path):
    """SF-13: rows_to_rerun=None triggers full cleanup of staging and output dirs."""
    from eval_pipeline.af3_runner import cleanup_for_rerun

    staging = tmp_path / "staging"
    output = tmp_path / "af3_outputs"
    staging.mkdir()
    output.mkdir()
    (staging / "x.json").write_text("{}")
    (output / "x").mkdir()

    cleanup_for_rerun(staging, output, None)

    assert not staging.exists()
    assert not output.exists()


def test_cleanup_for_rerun_rejects_bad_type(tmp_path):
    """SF-13: unsupported row collections must raise instead of silent no-op."""
    from eval_pipeline.af3_runner import cleanup_for_rerun

    with pytest.raises(TypeError, match="DataFrame, or list"):
        cleanup_for_rerun(tmp_path / "s", tmp_path / "o", "not_a_collection")


def test_run_af3_batched_check_false_returns_dict(tmp_path, monkeypatch):
    from eval_pipeline import af3_runner

    class FakeCompleted:
        def __init__(self, code, stdout, stderr):
            self.returncode = code
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, check=False, timeout=None, capture_output=True, **kw):
        assert check is False
        return FakeCompleted(code=1, stdout="processed 5 designs\n", stderr="error on 6th\n")

    monkeypatch.setattr(af3_runner.subprocess, "run", fake_run)
    result = af3_runner.run_af3_batched(
        staging_dir=tmp_path / "staging",
        output_dir=tmp_path / "out",
        af3_env={"AF3_PYTHON": "/bin/python", "AF3_RUN_SCRIPT": "/x/run.py",
                 "AF3_MODEL_DIR": "/m", "AF3_DB_DIR": "/d"},
        timeout_s=60,
    )
    assert result["exit_code"] == 1
    assert "processed 5" in result["stdout_tail"]
    assert "error on 6th" in result["stderr_tail"]
    assert result["timed_out"] is False
    assert result["wall_clock_s"] >= 0


def test_run_af3_batched_force_output_dir_in_cmd(tmp_path, monkeypatch):
    from eval_pipeline import af3_runner

    captured = {}

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return FakeCompleted()

    monkeypatch.setattr(af3_runner.subprocess, "run", fake_run)
    af3_runner.run_af3_batched(
        staging_dir=tmp_path / "staging",
        output_dir=tmp_path / "out",
        af3_env={"AF3_PYTHON": "/bin/python", "AF3_RUN_SCRIPT": "/x/run.py",
                 "AF3_MODEL_DIR": "/m", "AF3_DB_DIR": "/d"},
        timeout_s=60,
    )
    cmd = captured["cmd"]
    assert "--force_output_dir" in cmd
    assert "--run_data_pipeline=False" in cmd


def test_run_af3_batched_timeout(tmp_path, monkeypatch):
    from eval_pipeline import af3_runner

    def fake_run(cmd, check=False, timeout=None, capture_output=True, **kw):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(af3_runner.subprocess, "run", fake_run)
    result = af3_runner.run_af3_batched(
        staging_dir=tmp_path / "staging",
        output_dir=tmp_path / "out",
        af3_env={"AF3_PYTHON": "/p", "AF3_RUN_SCRIPT": "/r",
                 "AF3_MODEL_DIR": "/m", "AF3_DB_DIR": "/d"},
        timeout_s=1,
    )
    assert result["timed_out"] is True
    assert result["exit_code"] == -1


def test_parse_one_af3_design_full_path(tmp_path):
    from Bio.PDB import MMCIFIO
    from eval_pipeline import af3_runner

    sub_run_out = tmp_path / "PDL1" / "PDL1"
    inner = sub_run_out / "PDL1__d_001"
    inner.mkdir(parents=True)

    structure = _make_tiny_structure_with_chains(["A", "B"])
    cif_path = inner / "PDL1__d_001_model.cif"
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(cif_path))

    summary = inner / "PDL1__d_001_summary_confidences.json"
    summary.write_text(json.dumps({
        "ranking_score": 0.75,
        "ptm": 0.82,
        "iptm": 0.65,
        "has_clash": False,
        "fraction_disordered": 0.1,
    }))
    confidences = inner / "PDL1__d_001_confidences.json"
    confidences.write_text(json.dumps({
        "atom_chain_ids": ["A", "A", "B", "B"],
        "atom_plddts": [85.0, 85.0, 92.0, 92.0],
        "contact_probs": [[0.0] * 2] * 2,
        "pae": [[0.0, 5.0], [5.0, 0.0]],
        "token_chain_ids": ["A", "B"],
        "token_res_ids": [1, 1],
    }))

    out_dir = tmp_path / "design_artifacts"
    staged = {
        "row": {"design_id": "d_001", "case": "PDL1", "method": "BCv1",
                "target_seq": "A", "binder_seq": "M"},
        "job_name": "PDL1__d_001",
        "json_path": tmp_path / "fake_input.json",
    }
    (tmp_path / "fake_input.json").write_text("{}")

    result = af3_runner.parse_one_af3_design(
        af3_sub_run_output_dir=sub_run_out,
        staged_design=staged,
        target_seqs=["A"],
        binder_seq="M",
        amortized_runtime_s=12.5,
        out_dir=out_dir,
    )
    assert result.runtime_s == 12.5
    assert result.plddt.shape == (2,)
    assert result.pae.shape == (2, 2)
    import numpy as np
    np.testing.assert_allclose(result.plddt, [80.0, 80.0], atol=0.01)
    np.testing.assert_allclose(result.pae, [[0.0, 5.0], [5.0, 0.0]], atol=0.01)
    assert result.native["ranking_score"] == 0.75
    assert result.native["iptm"] == 0.65
    assert (out_dir / "af3_pred.pdb").exists()
    assert (out_dir / "plddt.npy").exists()
    assert (out_dir / "pae.npy").exists()
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "input.json").exists()


def test_prepare_msa_extracts_unpaired_msa_from_data_json(tmp_path):
    from scripts.prepare_af3_target_msa import _extract_msas_from_data_json

    data = {
        "name": "PDL1__target_only",
        "sequences": [
            {"protein": {"id": "B", "sequence": "PEPTIDE",
                         "unpairedMsa": ">a\nPEPTIDE\n>b\nPEPTIDA"}},
            {"protein": {"id": "C", "sequence": "ANOTHER",
                         "unpairedMsa": ">a\nANOTHER\n"}},
        ],
    }
    data_json = tmp_path / "data.json"
    data_json.write_text(json.dumps(data))

    out_dir = tmp_path / "out_msa"
    paths = _extract_msas_from_data_json(data_json, out_dir)
    assert (out_dir / "target_chain_B.a3m").exists()
    assert (out_dir / "target_chain_C.a3m").exists()
    assert (out_dir / "target_chain_B.a3m").read_text().startswith(">a")
    assert set(paths) == {"B", "C"}
