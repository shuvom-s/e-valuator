import pandas as pd
import numpy as np
from tqdm import tqdm


def add_judge_probability_series(
    df: pd.DataFrame,
    prob_col: str = "judge_probability",
    problem_col: str = "uq_problem_idx",
    step_col: str = "num_steps",
) -> pd.DataFrame:
    """
    Add a column `judge_probability_series`, where each entry is a list of
    judge probabilities from step 1 up to that row's step for the same problem.

    Args:
        df:
            Input DataFrame. Must contain:
                - problem_col  (default: "uq_problem_idx")
                - step_col     (default: "num_steps")
                - prob_col     (default: "judge_probability")
        prob_col:
            Column containing the per-step judge probability.
        problem_col:
            Column identifying the problem / trajectory.
        step_col:
            Column giving the step index (1, 2, 3, ...).

    Returns:
        A copy of df with an extra column:
            - judge_probability_series: list of probabilities from step 1
              up to the current step for that (problem_col).
    """
    if prob_col not in df.columns:
        raise KeyError(f"Probability column '{prob_col}' not found in df.")
    if problem_col not in df.columns:
        raise KeyError(f"problem_col '{problem_col}' not found in df.")
    if step_col not in df.columns:
        raise KeyError(f"step_col '{step_col}' not found in df.")

    df = df.copy()
    series_out = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Building judge_probability_series"
    ):
        pid = row[problem_col]
        step = row[step_col]

        prefix = (
            df[(df[problem_col] == pid) & (df[step_col] <= step)]
            .sort_values(step_col)
            .drop_duplicates(subset=[problem_col, step_col], keep="first")
        )

        series_out.append(prefix[prob_col].tolist())

    df["judge_probability_series"] = series_out
    return df

