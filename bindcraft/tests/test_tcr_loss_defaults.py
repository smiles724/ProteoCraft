import argparse


def _empty_loss_args(keys):
    return argparse.Namespace(**{f"w_{key}": None for key in keys})


def test_tcr_runner_defaults_to_tuned_loss_weights():
    from benchmark.script import run_bindcraft_tcr_case as tcr_case
    from bindcraft.utils import TUNED_LOSS_WEIGHTS

    args = _empty_loss_args(tcr_case.LOSS_WEIGHT_KEYS)

    assert tcr_case.resolve_loss_weights(args) == TUNED_LOSS_WEIGHTS


def test_tcr_runner_explicit_loss_weight_overrides_tuned_default():
    from benchmark.script import run_bindcraft_tcr_case as tcr_case
    from bindcraft.utils import TUNED_LOSS_WEIGHTS

    args = _empty_loss_args(tcr_case.LOSS_WEIGHT_KEYS)
    args.w_i_pae = 0.7

    weights = tcr_case.resolve_loss_weights(args)

    assert weights["i_pae"] == 0.7
    assert weights["con"] == TUNED_LOSS_WEIGHTS["con"]
