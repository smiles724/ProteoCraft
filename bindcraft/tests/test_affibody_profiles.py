import pytest

from bindcraft.affibody.profiles import get_affibody_profile


def test_libb_profile_reconstructs_experimental_ny_eso_affibody():
    profile = get_affibody_profile("LibB")

    assert profile.length == 60
    assert profile.mutable_positions_1based == (8, 12, 15, 16, 19)
    assert profile.default_tuple == "NNYYF"
    assert profile.render() == (
        "VDNKNKFNKEFNNAYYEIFHLPNLNEEQFDAFVQSLFDDPSQSANLLAEAKKLNDAQAPK"
    )
    assert profile.extract_tuple(profile.render()) == "NNYYF"


def test_affibody_profile_rejects_framework_edits_and_outside_alphabet():
    profile = get_affibody_profile("liba")
    sequence = list(profile.render())
    sequence[0] = "A"
    with pytest.raises(ValueError, match="framework position 1"):
        profile.validate_sequence("".join(sequence))

    sequence = list(profile.render())
    sequence[profile.mutable_positions_1based[0] - 1] = "C"
    with pytest.raises(ValueError, match="outside the experimental alphabet"):
        profile.validate_sequence("".join(sequence))


def test_mutable_mask_marks_only_experimental_positions():
    profile = get_affibody_profile("libb")

    assert sum(profile.mutable_mask) == 5
    assert [
        index + 1 for index, value in enumerate(profile.mutable_mask) if value
    ] == list(profile.mutable_positions_1based)
    assert len(profile.mutable_mask_argument) == profile.length
