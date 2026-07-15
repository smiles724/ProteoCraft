from pathlib import Path

import pytest

from eval_pipeline.hotspot_coverage import calculate_hotspot_coverage


def _atom_line(serial, atom, residue, chain, residue_number, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain:1s}{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00          {element:>2s}\n"
    )


def _write_complex(path: Path) -> None:
    records = []
    serial = 1
    for chain, residue_number, x in [
        ("A", 1, 0.0),
        ("A", 2, 20.0),
        ("B", 1, 4.0),
        ("C", 1, 1.0),
        ("C", 2, 30.0),
    ]:
        records.append(_atom_line(serial, "N", "ALA", chain, residue_number, x, 0, 0, "N"))
        serial += 1
        records.append(_atom_line(serial, "CA", "ALA", chain, residue_number, x + 0.5, 0, 0, "C"))
        serial += 1
    path.write_text("".join(records) + "END\n", encoding="utf-8")


def test_multichain_hotspot_indices_map_to_target_chains(tmp_path):
    structure_path = tmp_path / "complex.pdb"
    _write_complex(structure_path)

    result = calculate_hotspot_coverage(
        structure_path,
        target_chain_ids=["A", "B"],
        binder_chain_id="C",
        hotspot_concat_indices_0based=[0, 2],
        expected_target_chain_lengths=[2, 1],
    )

    assert result["hotspot_count"] == 2
    assert result["heavy_atom_covered_count"] == 2
    assert result["heavy_atom_coverage"] == pytest.approx(1.0)
    assert result["ca_covered_count"] == 2
    assert [item["chain_id"] for item in result["per_hotspot"]] == ["A", "B"]


def test_uncovered_hotspot_and_input_validation(tmp_path):
    structure_path = tmp_path / "complex.pdb"
    _write_complex(structure_path)

    result = calculate_hotspot_coverage(
        structure_path,
        target_chain_ids=["A", "B"],
        binder_chain_id="C",
        hotspot_concat_indices_0based=[1],
        expected_target_chain_lengths=[2, 1],
    )
    assert result["heavy_atom_coverage"] == pytest.approx(0.0)
    assert result["ca_coverage"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="outside target length"):
        calculate_hotspot_coverage(
            structure_path,
            target_chain_ids=["A", "B"],
            binder_chain_id="C",
            hotspot_concat_indices_0based=[3],
        )

    with pytest.raises(ValueError, match="target chain lengths do not match"):
        calculate_hotspot_coverage(
            structure_path,
            target_chain_ids=["A", "B"],
            binder_chain_id="C",
            hotspot_concat_indices_0based=[0],
            expected_target_chain_lengths=[1, 2],
        )
