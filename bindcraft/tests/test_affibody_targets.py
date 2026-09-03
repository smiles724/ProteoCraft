from bindcraft.affibody.targets import (
    build_pmhc_target,
    chains_from_reference_pdb,
)


def test_pmhc_target_maps_peptide_hotspots_into_concatenated_target():
    target = build_pmhc_target(
        target_id="T001",
        peptide="SLLMWITQV",
        hla_sequence="AAAA",
        b2m_sequence="GGG",
        register="P4/P5",
    )

    assert target.target_argument == "AAAA|GGG|SLLMWITQV"
    assert target.up_facing_pair == "MW"
    assert target.hotspot_positions_1based == (11, 12)
    assert target.hotspot_argument == "11,12"
    assert target.target_chain_roles == ("scaffold", "scaffold", "target")
    assert target.target_chain_roles_argument == "scaffold,scaffold,target"


def test_pdb_chain_sequence_extraction(tmp_path):
    pdb = tmp_path / "reference.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  GLY A   2      1.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  CA  SER B   1      2.000   0.000   0.000  1.00 20.00           C\n",
        encoding="utf-8",
    )

    assert chains_from_reference_pdb(pdb) == ("AG", "S")
