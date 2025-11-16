# tests/test_evaluator_workflow.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluator import EValuator  
from evaluator.utils import add_judge_probability_series  


def test_invalid_alpha_raises():
    # alpha <= 0
    with pytest.raises(AssertionError):
        EValuator(alphas=[0.0])

    # alpha >= 1
    with pytest.raises(AssertionError):
        EValuator(alphas=[1.0])

    # alpha outside (0,1)
    with pytest.raises(AssertionError):
        EValuator(alphas=[-0.1, 0.05])

    # empty alphas
    with pytest.raises(AssertionError):
        EValuator(alphas=[])


def test_invalid_delta_raises():
    with pytest.raises(AssertionError):
        EValuator(delta=0.0)   # must be in (0,1)

    with pytest.raises(AssertionError):
        EValuator(delta=1.5)


def test_invalid_mt_variant_raises():
    with pytest.raises(AssertionError):
        EValuator(mt_variant="not_a_valid_variant")



def test_insufficient_data_step1_raises():
    """
    Step 1 must have >= 5 examples per class AND two classes present.
    Construct a tiny calib_df that violates this, and ensure .fit raises.
    """
    calib_df = pd.DataFrame({
        "uq_problem_idx": [f"p{i}" for i in range(4)],
        "num_steps": [1, 1, 1, 1],
        "solved": [1, 1, 1, 1],
        "judge_probability_series": [[0.5]] * 4,
    })

    ev = EValuator(mt_variant="anytime")

    with pytest.raises(AssertionError):
        ev.fit(calib_df)

def test_e_vals_carry_forward_when_no_model_for_step():
    """
    If there is no trained model for a given step, the e-value for that step
    should equal the last available e-value for that problem.

    Here we train only on step=1, then apply to a problem that has steps 1 and 2.
    There is no model for step=2, so step=2 must copy the step=1 value.
    """
    ## Calibration set: only step 1 appears, with enough data to train
    calib_df = pd.DataFrame({
        "uq_problem_idx": [f"p{i}" for i in range(10)],
        "num_steps": [1] * 10,
        "solved": [0] * 5 + [1] * 5,
        "judge_probability_series": [[0.2]] * 5 + [[0.8]] * 5,
    })

    ev = EValuator(mt_variant="anytime", alphas=[0.05])
    ev.fit(calib_df)

    ## Test set: a single problem with steps 1 and 2
    test_df = pd.DataFrame({
        "uq_problem_idx": ["p_test", "p_test"],
        "num_steps": [1, 2],
        "solved": [0, 0],
        ## Values here don't matter for step=2 since no model exists for step=2
        "judge_probability_series": [[0.3], [0.4]],
    })

    out = ev.apply(test_df, compute_rejects=False)
    vals = out["anytime_e_val"].values

    ## Step 1 should produce some positive e-value
    assert vals[0] > 0

    ## Step 2 has no model; it should copy the last ratio from step 1
    assert vals[1] == pytest.approx(vals[0])

@pytest.mark.slow
@pytest.mark.parametrize("mt_variant", ["anytime", "split", "both"])
def test_evaluator_workflow_from_csv(mt_variant):
    data_path = ROOT / "data" / "hotpotqa_cleaned_w_scores.csv"
    if not data_path.exists():
        pytest.skip(f"Data file not found: {data_path}")

    # Load raw data
    df = pd.read_csv(data_path)
    assert "uq_problem_idx" in df.columns, "Expected 'uq_problem_idx' in CSV"
    assert "num_steps" in df.columns, "Expected 'num_steps' in CSV"
    assert "solved" in df.columns, "Expected 'solved' in CSV"
    assert "judge_probability" in df.columns or "judge_probability_expanded" in df.columns

    # Add judge_probability_series
    df = add_judge_probability_series(df)
    assert "judge_probability_series" in df.columns
    # Every entry should be a non-empty list
    assert df["judge_probability_series"].apply(lambda x: isinstance(x, list) and len(x) > 0).all()

    # Split into calibration / test by uq_problem_idx
    unique_ids = df["uq_problem_idx"].unique()
    rng = np.random.default_rng(42)
    n_cal = int(0.8 * len(unique_ids))
    cal_ids = rng.choice(unique_ids, size=n_cal, replace=False)
    test_ids = np.setdiff1d(unique_ids, cal_ids)

    cal_df = df[df["uq_problem_idx"].isin(cal_ids)].reset_index(drop=True)
    test_df = df[df["uq_problem_idx"].isin(test_ids)].reset_index(drop=True)

    assert len(cal_df) > 0
    assert len(test_df) > 0

    # Instantiate evaluator
    ev = EValuator(
        model_type="logistic",
        mt_variant=mt_variant,
        alphas=[0.01, 0.05, 0.1],
        delta=0.05,
    )

    # Fit on calibration set
    ev.fit(cal_df)

    # Basic sanity on thresholds
    if mt_variant == "anytime":
        assert isinstance(ev.thresholds, dict)
        assert set(ev.thresholds.keys()) == set(ev.alphas)
        for a, thr in ev.thresholds.items():
            assert thr == pytest.approx(1.0 / a)
    elif mt_variant == "split":
        assert isinstance(ev.thresholds, dict)
        assert set(ev.thresholds.keys()) == set(ev.alphas)
        # All thresholds should be positive
        for thr in ev.thresholds.values():
            assert thr >= 0.0
    else:  # "both"
        assert isinstance(ev.thresholds, dict)
        assert set(ev.thresholds.keys()) == {"anytime", "split"}
        assert set(ev.thresholds["anytime"].keys()) == set(ev.alphas)
        assert set(ev.thresholds["split"].keys()) == set(ev.alphas)

    # Apply to test set
    test_with_scores = ev.apply(test_df)

    ## Check columns by variant
    if mt_variant in {"anytime", "both"}:
        assert "anytime_e_val" in test_with_scores.columns
        ## e-values should be positive
        assert (test_with_scores["anytime_e_val"] > 0).all()

    if mt_variant in {"split", "both"}:
        assert "split_e_val" in test_with_scores.columns
        assert (test_with_scores["split_e_val"] > 0).all()

