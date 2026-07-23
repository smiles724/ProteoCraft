from pathlib import Path

import pytest

from bindcraft.hallucination.esmfold2_cdr import (
    framework_unchanged,
    prepare_esmfold2_cdr_case,
)


def _write_case_fasta(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                ">example|Chain A|target",
                "AAAA",
                ">example|Chain B|scaffold",
                "CCCC",
                ">example|Chain C|binder",
                "GGACDEFGHIIK",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_prepare_esmfold2_cdr_case_preserves_roles_and_cdr_mask(tmp_path):
    fasta_path = tmp_path / "case.fasta"
    template_path = tmp_path / "case.cif"
    _write_case_fasta(fasta_path)
    template_path.write_text("data_case\n", encoding="utf-8")

    case = prepare_esmfold2_cdr_case(
        {
            "id": "case_alpha",
            "fasta_path": fasta_path.name,
            "template_path": template_path.name,
            "binder_chain": "C",
            "target_chains": ["A"],
            "scaffold_chains": ["B"],
            "cdr_motif": "ACDEFG",
        },
        input_dir=tmp_path,
    )

    assert case.partner_sequences == ("AAAA", "CCCC")
    assert case.target_chain_roles == ("target", "scaffold")
    assert case.target_sequence_argument == "AAAA|CCCC"
    assert case.mutable_mask_argument == "001111110000"
    assert case.cdr_start_0based == 2
    assert case.cdr_end_0based_exclusive == 8


def test_framework_guard_rejects_changes_outside_cdr():
    reference = "AAACCC"
    mutable_mask = (False, False, False, True, True, True)

    assert framework_unchanged(reference, "AAAGGG", mutable_mask)
    assert not framework_unchanged(reference, "TAAGGG", mutable_mask)


def test_prepare_esmfold2_cdr_case_rejects_missing_motif(tmp_path):
    fasta_path = tmp_path / "case.fasta"
    _write_case_fasta(fasta_path)

    with pytest.raises(ValueError, match="CDR motif not found"):
        prepare_esmfold2_cdr_case(
            {
                "id": "bad_case",
                "fasta_path": fasta_path.name,
                "binder_chain": "C",
                "target_chains": ["A"],
                "scaffold_chains": ["B"],
                "cdr_motif": "YYYY",
            },
            input_dir=tmp_path,
        )
