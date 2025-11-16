import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import binom


class EValuator:
    """
    Stepwise evaluator using density ratio estimation + e-values.

    Args:
        model_type: "logistic" or "random_forest"
        mt_variant: "anytime", "split", or "both"
        alphas: list of alpha levels
        delta: tolerance-bound failure probability for split mode (1 - confidence)
        problem_col: column name for problem identifier (default "uq_problem_idx")
        step_col: column name for step index (default "num_steps")
        split_fraction: fraction of problems used for log-training in split mode
        random_state: RNG seed for the internal log/cal split
    """

    def __init__(
        self,
        model_type: str = "logistic",
        mt_variant: str = "split",
        alphas=None,
        delta: float = 0.1,
        problem_col: str = "uq_problem_idx",
        step_col: str = "num_steps",
        split_fraction: float = 0.5,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.mt_variant = mt_variant
        self.alphas = alphas if alphas is not None else [0.05]
        self.delta = delta
        self.problem_col = problem_col
        self.step_col = step_col
        self.split_fraction = split_fraction
        self.random_state = random_state

        assert isinstance(self.problem_col, str), "problem_col must be a string."
        assert isinstance(self.step_col, str), "step_col must be a string."

        assert len(self.alphas) > 0, "Must specify at least one alpha."
        for a in self.alphas:
            assert 0 < a < 1, f"Alpha must be in (0,1). Got {a}"

        assert 0 < self.delta < 1, f"Delta must be in (0,1). Got {self.delta}"
        assert 0 < self.split_fraction < 1, f"split_fraction must be in (0,1). Got {self.split_fraction}"
        assert self.mt_variant in {"anytime", "split", "both"}, (
            "mt_variant must be 'anytime', 'split', or 'both'. "
            f"Got {self.mt_variant}"
        )

        ## Model families for anytime and split
        self.step_models_anytime = {}
        self.step_scalers_anytime = {}
        self.step_base_probs_anytime = {}
        self.max_trained_step_anytime = 0

        self.step_models_split = {}
        self.step_scalers_split = {}
        self.step_base_probs_split = {}
        self.max_trained_step_split = 0

        ## For "anytime" or "split": dict[alpha] -> threshold
        ## For "both": {"anytime": {alpha: thr}, "split": {alpha: thr}}
        self.thresholds = {}

    def _new_model(self):
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=200)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=200)
        raise ValueError(f"Unknown model type: {self.model_type}")

    def _upper_tolerance_bound(self, values, alpha, delta):
        """
        Distribution-free one-sided upper bound for the (1−alpha)-quantile
        with confidence (1−delta).
        """
        Xs = np.sort(np.asarray(values))
        n = Xs.size
        if n == 0:
            return 0.0

        p = 1.0 - alpha
        sig = delta

        for k in range(1, n + 1):
            if binom.sf(k - 1, n, p) <= sig:
                return float(Xs[k - 1])
        return float(Xs[-1])


    def _fit_step_models(self, df: pd.DataFrame, which: str):
        """
        Fit one density-ratio model per step on df, for the given variant.
        Should be one of {"anytime", "split"}.
        """
        max_steps = df[self.step_col].max()

        for step in range(1, max_steps + 1):
            step_df = df[df[self.step_col] == step]
            y = step_df["solved"].values
            uniq, counts = np.unique(y, return_counts=True)

            if step == 1:
                if len(uniq) < 2 or np.min(counts) < 5:
                    raise AssertionError("Step 1 requires ≥5 examples per class.")
            else:
                if len(uniq) < 2 or np.min(counts) < 5:
                    continue

            X = np.array([list(s) for s in step_df["judge_probability_series"].values])
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            model = self._new_model()
            model.fit(Xs, y)

            p1 = float(np.mean(y))
            p0 = 1.0 - p1

            if which == "anytime":
                self.step_models_anytime[step] = model
                self.step_scalers_anytime[step] = scaler
                self.step_base_probs_anytime[step] = {"p0": p0, "p1": p1}
                self.max_trained_step_anytime = max(self.max_trained_step_anytime, step)
            elif which == "split":
                self.step_models_split[step] = model
                self.step_scalers_split[step] = scaler
                self.step_base_probs_split[step] = {"p0": p0, "p1": p1}
                self.max_trained_step_split = max(self.max_trained_step_split, step)
            else:
                raise ValueError(f"Unknown which={which}")

    def _compute_e_vals_for_variant(self, df: pd.DataFrame, which: str) -> np.ndarray:
        """
        Compute density ratios per row for the requested variant ("anytime" or "split").
        """
        if which == "anytime":
            models = self.step_models_anytime
            scalers = self.step_scalers_anytime
            base_probs = self.step_base_probs_anytime
        elif which == "split":
            models = self.step_models_split
            scalers = self.step_scalers_split
            base_probs = self.step_base_probs_split
        else:
            raise ValueError(f"Unknown which={which}")

        e_vals = np.full(len(df), np.nan, dtype=float)
        last_ratio = {}

        for idx, row in df.iterrows():
            pid = row[self.problem_col]
            step = int(row[self.step_col])

            if pid not in last_ratio:
                last_ratio[pid] = 1.0

            if step in models:
                series = np.array(row["judge_probability_series"], dtype=float)
                X = series.reshape(1, -1)
                scaler = scalers[step]
                model = models[step]

                Xs = scaler.transform(X)
                proba = model.predict_proba(Xs)[0]
                p1_s = np.clip(proba[1], 1e-6, 1 - 1e-6)
                p0_s = np.clip(proba[0], 1e-6, 1 - 1e-6)

                base = base_probs[step]
                p0 = base["p0"]
                p1 = base["p1"]

                ## Density ratio:
                ##   Pr(S | Y=0) / Pr(S | Y=1)
                ## = (Pr(Y=1 | S) / Pr(Y=0 | S)) * (Pr(Y=0) / Pr(Y=1))
                ## and we are using the equivalent form:
                ##   (p0_s / p1_s) * (p1 / p0)
                ratio = (p0_s / p1_s) * (p1 / p0)
                last_ratio[pid] = ratio
            else:
                ratio = last_ratio[pid]

            e_vals[idx] = ratio

        return e_vals

    def _compute_split_thresholds(self, calib_df: pd.DataFrame) -> dict:
        """
        Use the held-out calibration split to get tolerance-bound thresholds
        for split_e_val.
        """
        tmp = calib_df.copy()
        tmp["split_e_val"] = self._compute_e_vals_for_variant(tmp, which="split")
        solved = tmp[tmp["solved"] == 1]

        group_max = []
        for _, g in solved.groupby(self.problem_col):
            vals = pd.to_numeric(g["split_e_val"], errors="coerce").dropna()
            if len(vals) > 0:
                group_max.append(vals.max())

        thresholds = {}
        for a in self.alphas:
            if len(group_max) == 0:
                thresholds[a] = 0.0
            else:
                thresholds[a] = self._upper_tolerance_bound(
                    group_max, alpha=a, delta=self.delta
                )
        return thresholds


    def fit(self, calib_df: pd.DataFrame):
        """
        Fit one model per step and compute per-alpha thresholds.

        calib_df must contain:
            - self.step_col
            - "solved"
            - "judge_probability_series"
            - self.problem_col (for split/both threshold computation)
        """
        calib_df = calib_df.copy()

        ## Anytime-only: train on full calib, Ville thresholds 1/alpha
        if self.mt_variant == "anytime":
            self._fit_step_models(calib_df, which="anytime")
            self.thresholds = {a: 1.0 / a for a in self.alphas}
            return

        ## For split version, first build a log/cal split by problem
        unique_ids = calib_df[self.problem_col].unique()
        rng = np.random.default_rng(self.random_state)
        n_log = max(1, int(len(unique_ids) * self.split_fraction))
        log_ids = rng.choice(unique_ids, size=n_log, replace=False)
        cal_ids = np.setdiff1d(unique_ids, log_ids)

        log_df = calib_df[calib_df[self.problem_col].isin(log_ids)].reset_index(drop=True)
        cal_df = calib_df[calib_df[self.problem_col].isin(cal_ids)].reset_index(drop=True)

        if self.mt_variant == "split":
            ## Split-only: train on one half, thresholds from held-out other half
            self._fit_step_models(log_df, which="split")
            self.thresholds = self._compute_split_thresholds(cal_df)
            return

        ## mt_variant == "both"
        ## Anytime side: train on full calib set, thresholds 1/alpha
        self._fit_step_models(calib_df, which="anytime")
        anytime_thresholds = {a: 1.0 / a for a in self.alphas}

        ## Split side: train on one half, thresholds from held-out other half
        self._fit_step_models(log_df, which="split")
        split_thresholds = self._compute_split_thresholds(cal_df)

        self.thresholds = {
            "anytime": anytime_thresholds,
            "split": split_thresholds,
        }

    def apply(self, df: pd.DataFrame, compute_rejects: bool = True) -> pd.DataFrame:
        """
        Compute e-values per row.

        df must contain:
            - self.step_col
            - self.problem_col
            - "judge_probability_series"

        Returns a copy of df with:
            - "anytime_e_val" if mt_variant in {"anytime", "both"}
            - "split_e_val"   if mt_variant in {"split", "both"}
            - reject columns (if compute_rejects=True):
                * "reject_anytime_alpha_{a}" for anytime/both
                * "reject_split_alpha_{a}"   for split/both
        """
        df = df.copy()

        has_anytime = self.mt_variant in {"anytime", "both"}
        has_split = self.mt_variant in {"split", "both"}

        if has_anytime:
            df["anytime_e_val"] = self._compute_e_vals_for_variant(df, which="anytime")

        if has_split:
            df["split_e_val"] = self._compute_e_vals_for_variant(df, which="split")

        if not compute_rejects:
            return df

        if self.mt_variant == "anytime":
            for a in self.alphas:
                thr = self.thresholds[a]
                base = str(a).replace(".", "_")
                col = f"reject_anytime_alpha_{base}"
                df[col] = df["anytime_e_val"] > thr

        elif self.mt_variant == "split":
            for a in self.alphas:
                thr = self.thresholds[a]
                base = str(a).replace(".", "_")
                col = f"reject_split_alpha_{base}"
                df[col] = df["split_e_val"] > thr

        else:  ## mt_variant == "both"
            anytime_thr = self.thresholds["anytime"]
            split_thr = self.thresholds["split"]

            for a in self.alphas:
                base = str(a).replace(".", "_")

                thr_any = anytime_thr[a]
                thr_split = split_thr[a]

                col_any = f"reject_anytime_alpha_{base}"
                col_split = f"reject_split_alpha_{base}"

                df[col_any] = df["anytime_e_val"] > thr_any
                df[col_split] = df["split_e_val"] > thr_split

        return df
