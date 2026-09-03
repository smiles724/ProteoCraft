import csv

from bindcraft.affibody import data


def test_cleavage_and_target_support_levels(monkeypatch):
    def fake_tables(_path, *, slide_number):
        if slide_number == 3:
            return [[
                ["", "NNYYF"],
                ["MW", "80"],
                ["EL", "70"],
            ]]
        return [[
            ["", "DSHK"],
            ["NF", "90"],
        ]]

    monkeypatch.setattr(data, "read_pptx_tables", fake_tables)
    cleavage = data.extract_cleavage_records("source.pptx")
    assert len(cleavage) == 3
    assert sum(bool(row["cleavage_binder"]) for row in cleavage) == 2

    header = [
        "Tier", "Tier meaning", "Oncogenic", "Have cleavage on this pair",
        "Epitope", "Sequence", "P2", "P9", "Register", "Up-facing pair",
        "Ile6-equivalent", "Transfer-dependent residue", "Source", "Antigen",
        "PDB", "Confidence",
    ]
    rows = [
        header,
        ["1", "", "YES", "", "exact", "SLLMWITQV", "", "", "P4/P5", "MW"],
        ["1", "", "YES", "", "transfer", "KVAELVHFL", "", "", "P4/P5", "EL"],
        ["1", "", "", "", "sequenced", "SLLAAITQV", "", "", "P4/P5", "AA"],
        ["2", "", "", "", "register", "SLLAAITQV", "", "", "P5/P6", "AI"],
        ["1", "", "", "", "compute", "SLLCGITQV", "", "", "P4/P5", "CG"],
    ]
    monkeypatch.setattr(
        data, "read_xlsx_tables", lambda _path: {"Targets_v3": rows}
    )
    targets = data.extract_target_records("targets.xlsx", cleavage_records=cleavage)

    assert [row["support_level"] for row in targets] == [
        "exact_full_peptide",
        "cleavage_pair_transfer",
        "sequencing_pair_supported",
        "register_extrapolation",
        "computational_only",
    ]


def test_sequencing_lookup_prefers_raw_data_and_computes_enrichment(tmp_path):
    raw = tmp_path / "LibB Raw data"
    raw.mkdir()
    for round_name, count, frequency in (
        ("R000", 2, 0.01),
        ("R001", 1, 0.005),
        ("R009", 20, 0.10),
        ("R010", 30, 0.15),
    ):
        path = raw / f"xxylibB_{round_name}_count_freq_pvalue.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["pep", "aff", "count", "frequency", "pvalue"])
            writer.writerow(["MW", "NNYYF", count, frequency, 0.0])

    result = data.lookup_sequencing_evidence(
        tmp_path, [("LibB", "MW", "NNYYF")]
    )[("LibB", "MW", "NNYYF")]

    assert result["R009_R010_count"] == 50
    assert result["R009_R010_frequency"] == 0.25
    assert result["log2_enrichment_vs_R000"] > 4.0
    assert result["sequencing_observed"] is True
