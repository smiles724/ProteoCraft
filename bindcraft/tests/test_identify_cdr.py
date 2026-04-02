"""Tests for CDR identification logic."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))
from identify_cdr import CDR_DEFINITIONS, is_in_cdr, build_cdr_mask


class TestCDRDefinitions:
    def test_all_definitions_have_six_regions(self):
        for name, defs in CDR_DEFINITIONS.items():
            assert set(defs.keys()) == {"L1", "L2", "L3", "H1", "H2", "H3"}, (
                f"{name} missing regions"
            )

    def test_ranges_are_valid(self):
        for name, defs in CDR_DEFINITIONS.items():
            for region, (start, end) in defs.items():
                assert start <= end, f"{name}/{region}: start {start} > end {end}"


class TestIsInCDR:
    def test_heavy_h1_contact(self):
        # Contact H1: 30-35
        assert is_in_cdr(30, "H", "contact") is True
        assert is_in_cdr(35, "H", "contact") is True
        assert is_in_cdr(29, "H", "contact") is False
        assert is_in_cdr(36, "H", "contact") is False

    def test_heavy_h3_contact(self):
        # Contact H3: 93-101
        assert is_in_cdr(93, "H", "contact") is True
        assert is_in_cdr(100, "H", "contact") is True
        assert is_in_cdr(101, "H", "contact") is True
        assert is_in_cdr(102, "H", "contact") is False

    def test_light_l3_chothia(self):
        # Chothia L3: 91-96
        assert is_in_cdr(91, "L", "chothia") is True
        assert is_in_cdr(96, "L", "chothia") is True
        assert is_in_cdr(90, "L", "chothia") is False

    def test_kappa_treated_as_light(self):
        # K (kappa) should use L CDR definitions
        assert is_in_cdr(91, "K", "chothia") is True

    def test_framework_positions(self):
        # Position 10 is framework in all definitions
        assert is_in_cdr(10, "H", "contact") is False
        assert is_in_cdr(10, "L", "contact") is False


class TestBuildCDRMask:
    def test_all_framework(self):
        # Positions 1-10 are framework for heavy chain
        numbered = [(i, " ", "A") for i in range(1, 11)]
        mask, regions = build_cdr_mask(numbered, "H", "contact")
        assert len(mask) == 10
        assert not any(mask)
        assert len(regions) == 0

    def test_h1_contact(self):
        # Positions 25-39, Contact H1: 30-35
        numbered = [(i, " ", "G") for i in range(25, 40)]
        mask, regions = build_cdr_mask(numbered, "H", "contact")
        expected = [False] * 5 + [True] * 6 + [False] * 4
        assert mask == expected
        assert "H1" in regions
        assert len(regions["H1"]) == 6

    def test_insertion_codes_included(self):
        # Position 35 + 35A should both be in H1 (Contact: 30-35)
        # Position 34 is also in H1. Position 36 is not.
        numbered = [
            (34, " ", "A"),
            (35, " ", "B"),
            (35, "A", "C"),
            (36, " ", "D"),
        ]
        mask, regions = build_cdr_mask(numbered, "H", "contact")
        assert mask == [True, True, True, False]

    def test_seq_idx_correct(self):
        numbered = [(30, " ", "S"), (31, " ", "Y"), (32, " ", "A")]
        mask, regions = build_cdr_mask(numbered, "H", "contact")
        assert all(mask)
        assert regions["H1"][0]["seq_idx"] == 0
        assert regions["H1"][2]["seq_idx"] == 2

    def test_light_chain_kappa(self):
        # Kappa chain should use L definitions
        # Contact L1: 30-36
        numbered = [(i, " ", "A") for i in range(28, 40)]
        mask, regions = build_cdr_mask(numbered, "K", "contact")
        # positions 30-36 are CDR, that's indices 2-8
        expected = [False, False] + [True] * 7 + [False] * 3
        assert mask == expected
        assert "L1" in regions
