import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_pipeline.esmfold_runner import (  # noqa: E402
    _assign_pdb_chain_ids,
    build_esmfold_input,
    compute_clean_segments,
    get_clean_index_arrays,
    rechain_stripped_pdb_by_segments,
    strip_linker,
)


# ─── build_esmfold_input ────────────────────────────────────────────
def test_single_chain_target():
    seq, segs = build_esmfold_input("AAAAA", "BBB")
    assert seq == "AAAAA" + "G" * 25 + "BBB"
    assert len(segs) == 2
    assert segs[0]["kind"] == "target_chain_0"
    assert segs[0]["dirty_start"] == 0 and segs[0]["dirty_end"] == 5
    assert segs[1]["kind"] == "binder"
    assert segs[1]["dirty_start"] == 30 and segs[1]["dirty_end"] == 33


def test_multichain_target_colon_split():
    seq, segs = build_esmfold_input("AAA:CCC", "BB")
    expected = "AAA" + "G" * 25 + "CCC" + "G" * 25 + "BB"
    assert seq == expected
    assert len(segs) == 3
    assert segs[2]["dirty_start"] == 56  # 3 + 25 + 3 + 25 = 56


def test_compute_clean_segments():
    _, segs = build_esmfold_input("AAA:CCC", "BB")
    clean = compute_clean_segments(segs)
    assert clean[0]["clean_start"] == 0 and clean[0]["clean_end"] == 3
    assert clean[1]["clean_start"] == 3 and clean[1]["clean_end"] == 6
    assert clean[2]["clean_start"] == 6 and clean[2]["clean_end"] == 8
    assert clean[2]["dirty_start"] == 56


def test_get_clean_index_arrays():
    _, segs = build_esmfold_input("AAA:CCC", "BB")
    clean = compute_clean_segments(segs)
    idx = get_clean_index_arrays(clean)
    assert list(idx["target_idx_clean"]) == [0, 1, 2, 3, 4, 5]
    assert list(idx["binder_idx_clean"]) == [6, 7]


def test_strip_linker_correct_lengths():
    _, segs = build_esmfold_input("AAA:CCC", "BB")
    L_dirty = 3 + 25 + 3 + 25 + 2
    plddt = np.arange(L_dirty, dtype=np.float32)
    pae = np.zeros((L_dirty, L_dirty), dtype=np.float32)
    plddt_clean, pae_clean, clean = strip_linker(plddt, pae, segs)
    assert plddt_clean.shape == (8,)
    assert pae_clean.shape == (8, 8)
    # target indices [0,1,2, 28,29,30]; binder [56,57]
    assert list(plddt_clean) == [0, 1, 2, 28, 29, 30, 56, 57]


# ─── rechain ────────────────────────────────────────────────────────
def test_assign_pdb_chain_ids():
    _, raw = build_esmfold_input("AAA:CCC", "BB")
    clean = compute_clean_segments(raw)
    with_chain = _assign_pdb_chain_ids(clean)
    assert [s["pdb_chain_id"] for s in with_chain] == ["A", "B", "C"]
    assert with_chain[2]["kind"] == "binder" and with_chain[2]["pdb_chain_id"] == "C"


def test_rechain_stripped_pdb_drops_linker_and_renames():
    """Synthesize 8 standard residues all in chain A; partition into 3 segments
    (3 + 3 + 2). Verify rechain produces 3 chains with correct counts."""
    from Bio.PDB import PDBParser

    pdb = "\n".join(
        f"ATOM  {i+1:>5}  CA  ALA A{i+1:>4}    "
        f"{1.0 * i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        for i in range(8)
    ) + "\nEND\n"
    structure = PDBParser(QUIET=True).get_structure("t", StringIO(pdb))
    segs = [
        {"kind": "target_chain_0", "length": 3, "dirty_start": 0, "dirty_end": 3,
         "clean_start": 0, "clean_end": 3, "pdb_chain_id": "A"},
        {"kind": "target_chain_1", "length": 3, "dirty_start": 3, "dirty_end": 6,
         "clean_start": 3, "clean_end": 6, "pdb_chain_id": "B"},
        {"kind": "binder", "length": 2, "dirty_start": 6, "dirty_end": 8,
         "clean_start": 6, "clean_end": 8, "pdb_chain_id": "C"},
    ]
    rechained = rechain_stripped_pdb_by_segments(structure, segs)
    chain_ids = [c.id for c in rechained[0]]
    assert sorted(chain_ids) == ["A", "B", "C"]
    counts = {c.id: sum(1 for r in c if r.id[0] == " ") for c in rechained[0]}
    assert counts == {"A": 3, "B": 3, "C": 2}


# ─── GPU integration test (skip if no CUDA) ─────────────────────────
@pytest.mark.gpu
def test_run_esmfold_pdl1_design(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from eval_pipeline.esmfold_runner import (
        dump_design_artifacts,
        load_esmfold,
        run_esmfold,
    )
    from vendor.proteina_complexa.pdb_chain_utils import get_chain_sequence

    fixture_pdb = "tests/fixtures/PDL1_l85_s131547_mpnn1_model2.pdb"
    if not Path(fixture_pdb).exists():
        pytest.skip(f"missing fixture: {fixture_pdb}")
    target_seq = get_chain_sequence("tests/fixtures/PDL1_target.pdb", "A")
    binder_seq = get_chain_sequence(fixture_pdb, "B")
    esm_seq, segs = build_esmfold_input(target_seq, binder_seq)
    tok, mdl = load_esmfold("cuda:0")
    result = run_esmfold(esm_seq, tok, mdl, "cuda:0")
    assert result.plddt.shape[0] == len(esm_seq)
    assert result.pae.shape == (len(esm_seq), len(esm_seq))
    # Regression guard: HF returns plddt in [0,1]; runner must rescale to [0,100]
    # so tier thresholds (80/90) and PDB b-factors are meaningful.
    assert 1.0 < result.plddt.max() <= 100.0, (
        f"plddt max={result.plddt.max():.3f} not in (1, 100] — "
        "scaling regression (HF [0,1] vs ESMFold convention [0,100])"
    )
    meta = dump_design_artifacts(tmp_path, esm_seq, segs, result)
    assert (tmp_path / "esmfold.pdb").exists()
    assert (tmp_path / "plddt.npy").exists()
    assert (tmp_path / "pae.npy").exists()
    assert meta["binder_len"] == 85
    print(f"runtime: {result.runtime_s:.1f}s")
