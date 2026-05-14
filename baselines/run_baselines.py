#!/usr/bin/env python3
"""
Run baselines + e-valuator (PAC) on a dataset.

Methods:
  raw_verifier         - reject if judge_probability <= alpha at any step
  pac_verifier         - PAC lower tolerance bound on min judge_probability
                         (taken over solved cal problems); reject if score < tau
  calibrated_verifier  - score calibrated via isotonic regression (https://en.wikipedia.org/wiki/Isotonic_regression), reject <= alpha at any step
  Ville                - reject if M_t > 1/alpha at any step (Ville's inequality)
  RandVille            - reject if M_t > 1/alpha for non-final steps,
                         > U/alpha at the final step (U ~ Uniform(0,1) per problem;
                         randomized Ville's inequality)
                         > See https://arxiv.org/abs/2304.02611 for details.
  evaluator_pac        - e-valuator with PAC tolerance-bound threshold

pac_verifier and evaluator_pac use a budget split alpha'=0.9*alpha,
delta'=0.1*alpha (sum=alpha), so their marginal FAR is bounded by alpha. One can
change this by changing the budget split.

Outputs (in baselines/results/):
  {stem}_terminations.csv  - per-problem rows for all methods/alphas/runs
  {stem}_summary.csv       - power_mean/sd, type1_mean/sd aggregated over runs

Usage (from main directory):
  python baselines/run_baselines.py --input_path data/math_w_scores_compressed.csv.gz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import evaluator as e_val
from evaluator.evaluator import EValuator

ALPHAS    = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
GROUP_COL = "uq_problem_idx"
STEP_COL  = "num_steps"


# ── Helpers ────────────────────────────────────────────────────────────────────

def lower_tolerance_bound(values, alpha, delta):
    """
    Distribution-free lower bound on the alpha-quantile (confidence 1-delta).
    With prob >= 1-delta, at most an alpha fraction of future null scores fall
    below this threshold. Used for pac_verifier.
    """
    Xs = np.sort(np.asarray(values, dtype=float))
    n  = Xs.size
    if n == 0:
        return 0.0
    for k in range(n, 0, -1):
        if binom.cdf(k - 1, n, alpha) <= delta:
            return float(Xs[k - 1])
    return 0.0


def first_true_step(g_sorted, flag_col):
    """Return the first STEP_COL value where flag_col is True, or None."""
    hits = g_sorted[g_sorted[flag_col].astype(bool)]
    return int(hits.iloc[0][STEP_COL]) if len(hits) > 0 else None


def tokens_up_to(g_raw_sorted, step_k):
    """Sum of generated_tokens up to step_k (inclusive), or full sum if None."""
    if step_k is None:
        return float(g_raw_sorted["generated_tokens"].sum())
    return float(g_raw_sorted.loc[g_raw_sorted[STEP_COL] <= step_k, "generated_tokens"].sum())


def make_row(pid, alpha, method, run, step_k, g_raw_sorted, total_tokens, matches_gt, has_tokens):
    row = {
        "uq_problem_idx":   pid,
        "alpha":            alpha,
        "method":           method,
        "run_seed":         run,
        "termination_step": np.nan if step_k is None else step_k,
    }
    if has_tokens:
        row["termination_all_tokens"] = tokens_up_to(g_raw_sorted, step_k)
        row["total_token_count"]      = total_tokens
    row["matches_gt"] = matches_gt
    return row


# ── Main experiment ────────────────────────────────────────────────────────────

def run_experiment(input_path: str, n_splits: int, cal_frac: float, out_dir: Path,
                   pac_delta: float = 0.1):
    input_path = Path(input_path)
    df = pd.read_csv(input_path)
    df["solved"] = df["solved"].astype(int)
    has_tokens = "generated_tokens" in df.columns

    raw_col = "judge_probability"

    print(f"Dataset : {input_path.name}")
    print(f"Rows    : {df.shape[0]}  |  Problems: {df[GROUP_COL].nunique()}")
    print(f"Tokens  : {has_tokens}   |  Splits: {n_splits}   |  Cal frac: {cal_frac}")

    print("Building judge_probability_series...")
    df_series = e_val.utils.add_judge_probability_series(df, prob_col=raw_col)

    unique_ids = df[GROUP_COL].unique()
    termination_rows = []

    for run in tqdm(range(n_splits), desc="splits"):
        rng   = np.random.default_rng(run)
        n_cal = int(cal_frac * len(unique_ids))
        cal_ids  = rng.choice(unique_ids, size=n_cal, replace=False)
        test_ids = np.setdiff1d(unique_ids, cal_ids)

        cal_df  = df[df[GROUP_COL].isin(cal_ids)].reset_index(drop=True)
        test_df = df[df[GROUP_COL].isin(test_ids)].reset_index(drop=True)
        cal_s   = df_series[df_series[GROUP_COL].isin(cal_ids)].reset_index(drop=True)
        test_s  = df_series[df_series[GROUP_COL].isin(test_ids)].reset_index(drop=True)

        # ── Calibrated verifier (isotonic) ───────────────────────────────────
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal_df[raw_col].values, cal_df["solved"].values)
        test_df = test_df.copy()
        test_df["p_hat_cal"] = iso.transform(test_df[raw_col].values)

        # ── PAC verifier thresholds (budget split: alpha'=0.9*a, delta'=0.1*a) ──
        # Null sample: min raw score per solved cal problem.
        solved_mins = (
            cal_df[cal_df["solved"] == 1]
            .groupby(GROUP_COL)[raw_col].min().values
        )
        pac_tau = {a: lower_tolerance_bound(solved_mins, 0.9 * a, 0.1 * a) for a in ALPHAS}

        # ── RandVille EValuator: produces Ville_e_val + RandVille rejects.
        #     Ville rejects use the same e-values with threshold 1/alpha.
        ev_rv = EValuator(mt_variant="RandVille", alphas=ALPHAS, random_state=run)
        ev_rv.fit(cal_s)
        test_rv = ev_rv.apply(test_s.copy())

        # ── PAC EValuator ────────────────────────────────────────────────────
        ev_pac = EValuator(mt_variant="PAC", alphas=ALPHAS, delta=pac_delta,
                           random_state=run)
        ev_pac.fit(cal_s)
        test_pac = ev_pac.apply(test_s.copy(), compute_rejects=False)

        # Budget split: alpha'=0.9*a, delta'=0.1*a (sum=a). Marginal FAR <= a.
        # Replicate ev_pac's internal log/cal split to get the held-out cal half.
        _inner_ids = cal_s[GROUP_COL].unique()
        _inner_rng = np.random.default_rng(run)  # same seed as ev_pac.random_state
        _n_log     = max(1, int(len(_inner_ids) * 0.5))
        _log_ids   = _inner_rng.choice(_inner_ids, size=_n_log, replace=False)
        _cal_ids   = np.setdiff1d(_inner_ids, _log_ids)
        _inner_cal = cal_s[cal_s[GROUP_COL].isin(_cal_ids)].reset_index(drop=True)
        _inner_cal = ev_pac.apply(_inner_cal, compute_rejects=False)
        _solved_cal = _inner_cal[_inner_cal["solved"] == 1]
        _group_max  = [
            g["PAC_e_val"].dropna().max()
            for _, g in _solved_cal.groupby(GROUP_COL)
            if g["PAC_e_val"].dropna().size > 0
        ]
        pac_thr = {
            a: ev_pac._upper_tolerance_bound(_group_max, alpha=0.9 * a, delta=0.1 * a)
            for a in ALPHAS
        }

        # ── Per-problem rows ─────────────────────────────────────────────────
        problem_solved = test_df.groupby(GROUP_COL)["solved"].max().to_dict()
        total_tok_dict = (
            test_df.groupby(GROUP_COL)["generated_tokens"].sum().to_dict()
            if has_tokens else {}
        )

        for pid in test_ids:
            g_raw = test_df[test_df[GROUP_COL] == pid].sort_values(STEP_COL)
            g_rv  = test_rv[test_rv[GROUP_COL] == pid].sort_values(STEP_COL)
            g_pac = test_pac[test_pac[GROUP_COL] == pid].sort_values(STEP_COL)

            matches_gt = problem_solved[pid]
            total_tok  = total_tok_dict.get(pid, np.nan)

            for a in ALPHAS:
                base = str(a).replace(".", "_")

                # raw_verifier
                sk = first_true_step(g_raw.assign(_f=g_raw[raw_col] <= a), "_f")
                termination_rows.append(make_row(
                    pid, a, "raw_verifier", run, sk, g_raw, total_tok, matches_gt, has_tokens))

                # pac_verifier (budget split: alpha'=0.9*a, delta'=0.1*a)
                tau = pac_tau[a]
                sk = first_true_step(g_raw.assign(_f=g_raw[raw_col] < tau), "_f")
                termination_rows.append(make_row(
                    pid, a, "pac_verifier", run, sk, g_raw, total_tok, matches_gt, has_tokens))

                # calibrated_verifier
                sk = first_true_step(g_raw.assign(_f=g_raw["p_hat_cal"] <= a), "_f")
                termination_rows.append(make_row(
                    pid, a, "calibrated_verifier", run, sk, g_raw, total_tok, matches_gt, has_tokens))

                # Ville: M_t > 1/alpha at any step
                sk = first_true_step(g_rv.assign(_f=g_rv["Ville_e_val"] > 1.0 / a), "_f")
                termination_rows.append(make_row(
                    pid, a, "Ville", run, sk, g_raw, total_tok, matches_gt, has_tokens))

                # RandVille: use the reject column produced by EValuator
                sk = first_true_step(g_rv, f"reject_RandVille_alpha_{base}")
                termination_rows.append(make_row(
                    pid, a, "RandVille", run, sk, g_raw, total_tok, matches_gt, has_tokens))

                # evaluator_pac (budget split: alpha'=0.9*a, delta'=0.1*a)
                sk = first_true_step(g_pac.assign(_f=g_pac["PAC_e_val"] > pac_thr[a]), "_f")
                termination_rows.append(make_row(
                    pid, a, "evaluator_pac", run, sk, g_raw, total_tok, matches_gt, has_tokens))

    # ── Save outputs ───────────────────────────────────────────────────────────
    stem    = input_path.name.replace(".csv.gz", "").replace(".csv", "")
    term_df = pd.DataFrame(termination_rows).sort_values(
        ["method", "alpha", "run_seed", GROUP_COL]
    )
    term_out = out_dir / f"{stem}_terminations.csv"
    term_df.to_csv(term_out, index=False)

    # Summary: "rejected" = termination_step is not NaN
    term_df["rejected"] = term_df["termination_step"].notna()
    summary = (
        term_df
        .groupby(["method", "alpha", "run_seed"])
        .apply(lambda g: pd.Series({
            "power":       g.loc[g["matches_gt"] == 0, "rejected"].mean(),
            "type1_error": g.loc[g["matches_gt"] == 1, "rejected"].mean(),
        }))
        .reset_index()
        .groupby(["method", "alpha"])
        .agg(
            power_mean=("power",       "mean"),
            power_sd  =("power",       "std"),
            type1_mean=("type1_error", "mean"),
            type1_sd  =("type1_error", "std"),
        )
        .reset_index()
    )
    summ_out = out_dir / f"{stem}_summary.csv"
    summary.to_csv(summ_out, index=False)

    print(f"\nWrote terminations : {term_out}")
    print(f"Wrote summary      : {summ_out}")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to dataset CSV/CSV.GZ")
    parser.add_argument("--n_splits",   type=int,   default=50,  help="Number of random splits")
    parser.add_argument("--cal_frac",   type=float, default=0.2, help="Calibration fraction")
    parser.add_argument("--pac_delta",  type=float, default=0.1, help="PAC failure prob (delta)")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    run_experiment(args.input_path, args.n_splits, args.cal_frac, out_dir,
                   pac_delta=args.pac_delta)


if __name__ == "__main__":
    main()
