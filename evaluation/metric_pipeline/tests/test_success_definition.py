"""Tests for eval_n_successes definition."""

import numpy as np
import pandas as pd

from eval_pipeline.cli import _compute_eval_n_successes


def test_success_all_valid_returns_1():
    out = {"error_msg": "", "binder_pLDDT": 88.0, "ipAE": 5.0, "binder_scRMSD": 1.2}
    assert _compute_eval_n_successes(out) == 1


def test_failure_error_msg_returns_0():
    out = {
        "error_msg": "AF3 timeout",
        "binder_pLDDT": 88.0,
        "ipAE": 5.0,
        "binder_scRMSD": 1.2,
    }
    assert _compute_eval_n_successes(out) == 0


def test_failure_python_float_nan():
    out = {"error_msg": "", "binder_pLDDT": float("nan"), "ipAE": 5.0, "binder_scRMSD": 1.2}
    assert _compute_eval_n_successes(out) == 0


def test_failure_numpy_float_nan():
    out = {"error_msg": "", "binder_pLDDT": np.float64("nan"), "ipAE": 5.0, "binder_scRMSD": 1.2}
    assert _compute_eval_n_successes(out) == 0


def test_failure_pandas_NA():
    out = {"error_msg": "", "binder_pLDDT": pd.NA, "ipAE": 5.0, "binder_scRMSD": 1.2}
    assert _compute_eval_n_successes(out) == 0
