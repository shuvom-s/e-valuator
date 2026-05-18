import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.stats import binom


class EValuator:
    """
    Stepwise evaluator using density ratio estimation + e-values.

    Args:
        model_type: "logistic" (default) or "svm"
        mt_variant: "PAC" (default, recommended — the method in the paper),
            "Ville", or "RandVille".
            "PAC" uses a PAC tolerance-bound threshold calibrated on held-out data.
            "Ville" uses Ville's inequality with threshold 1/alpha.
            "RandVille" uses the randomized Ville inequality.
        alphas: list of target marginal FAR budgets (default [0.05]). For PAC,
            each target a is split internally as alpha' = w0*a, delta' = w1*a
            using alpha_delta_weights; by the union bound the marginal FAR is
            bounded by (w0 + w1) * a (= a when the weights sum to 1).
            For Ville/RandVille, alpha is used directly.
        alpha_delta_weights: pair (w0, w1) controlling the PAC budget split
            of each target alpha into (alpha', delta'). Default (0.9, 0.1).
            Must satisfy w0 > 0, w1 > 0, and w0 + w1 <= 1.
            Only used when mt_variant="PAC".
        problem_col: column name for problem identifier (default "uq_problem_idx")
        step_col: column name for step index (default "num_steps")
        split_fraction: fraction of problems used for log-training in PAC mode
        random_state: RNG seed for the internal log/cal split and RandVille draws
    """

    def __init__(
        self,
        model_type: str = "logistic",
        mt_variant: str = "PAC",
        alphas=None,
        alpha_delta_weights=(0.9, 0.1),
        problem_col: str = "uq_problem_idx",
        step_col: str = "num_steps",
        split_fraction: float = 0.5,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.mt_variant = mt_variant
        self.alphas = alphas if alphas is not None else [0.05]
        self.alpha_delta_weights = tuple(alpha_delta_weights)
        self.problem_col = problem_col
        self.step_col = step_col
        self.split_fraction = split_fraction
        self.random_state = random_state

        assert isinstance(self.problem_col, str), "problem_col must be a string."
        assert isinstance(self.step_col, str), "step_col must be a string."

        assert len(self.alphas) > 0, "Must specify at least one alpha."
        for a in self.alphas:
            assert 0 < a < 1, f"Alpha must be in (0,1). Got {a}"

        assert len(self.alpha_delta_weights) == 2, (
            f"alpha_delta_weights must be a pair (w0, w1). Got {self.alpha_delta_weights}"
        )
        w0, w1 = self.alpha_delta_weights
        assert w0 > 0 and w1 > 0, (
            f"alpha_delta_weights must be strictly positive. Got {self.alpha_delta_weights}"
        )
        assert w0 + w1 <= 1.0 + 1e-12, (
            f"alpha_delta_weights must sum to <= 1 (so marginal FAR <= alpha). "
            f"Got {self.alpha_delta_weights} (sum={w0 + w1})"
        )
        assert 0 < self.split_fraction < 1, f"split_fraction must be in (0,1). Got {self.split_fraction}"
        assert self.mt_variant in {"Ville", "PAC", "RandVille"}, (
            "mt_variant must be 'Ville', 'PAC', or 'RandVille'. "
            f"Got {self.mt_variant}"
        )

        ## Model families for Ville/RandVille (trained on full calib)
        self.step_models_anytime = {}
        self.step_scalers_anytime = {}
        self.step_base_probs_anytime = {}
        self.max_trained_step_anytime = 0

        ## Model families for PAC (trained on log half of calib)
        self.step_models_split = {}
        self.step_scalers_split = {}
        self.step_base_probs_split = {}
        self.max_trained_step_split = 0

        self.thresholds = {}

        self.base_probs_anytime = None
        self.base_probs_split = None

    ## TODO: add a way to specify the model hyperparameters, also add other model families
    def _new_model(self):
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=200)
        elif self.model_type == "svm":
            return SVC(probability=True, kernel="rbf")
        raise ValueError(f"Unknown model type: {self.model_type}")

    def _upper_tolerance_bound(self, values, alpha, delta):
        """
        Distribution-free one-sided upper bound for the (1−alpha)-quantile
        with confidence (1−delta).
        """
        Xs = np.sort(np.asarray(values))
        n = Xs.size
        if n == 0:
            return float('inf')

        p = 1.0 - alpha
        sig = delta

        for k in range(1, n + 1):
            if binom.sf(k - 1, n, p) <= sig:
                return float(Xs[k - 1])
        return float('inf')

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

            if step == 1:
                p1 = float(np.mean(y))
                p0 = 1.0 - p1
                if which == "anytime":
                    self.base_probs_anytime = {"p0": p0, "p1": p1}
                elif which == "split":
                    self.base_probs_split = {"p0": p0, "p1": p1}

            if which == "anytime":
                base = self.base_probs_anytime
                p0 = base["p0"]
                p1 = base["p1"]
                self.step_models_anytime[step] = model
                self.step_scalers_anytime[step] = scaler
                self.step_base_probs_anytime[step] = {"p0": p0, "p1": p1}
                self.max_trained_step_anytime = max(self.max_trained_step_anytime, step)
            elif which == "split":
                base = self.base_probs_split
                p0 = base["p0"]
                p1 = base["p1"]
                self.step_models_split[step] = model
                self.step_scalers_split[step] = scaler
                self.step_base_probs_split[step] = {"p0": p0, "p1": p1}
                self.max_trained_step_split = max(self.max_trained_step_split, step)
            else:
                raise ValueError(f"Unknown which={which}")

    def _compute_e_vals_for_variant(self, df: pd.DataFrame, which: str) -> np.ndarray:
        """
        Compute density ratios per row for the requested variant ("anytime" or "split").
        Anytime: uses entire calibration set to train density ratios and later applies Ville or Randomized Ville threshold.
        Split: uses half of calibration set to train density ratios and uses the other half to compute PAC thresholds.
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
                ## initialize M_0 = 1
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

            ## if no model exists for this step, just keep copying the last ratio.
            ## false alarm guarantee still holds!
            else:
                ratio = last_ratio[pid]

            e_vals[idx] = ratio

        return e_vals

    def _compute_pac_thresholds(self, calib_df: pd.DataFrame) -> dict:
        """
        Use the held-out calibration split to get PAC tolerance-bound thresholds.
        """
        tmp = calib_df.copy()
        tmp["PAC_e_val"] = self._compute_e_vals_for_variant(tmp, which="split")
        solved = tmp[tmp["solved"] == 1]

        group_max = []
        for _, g in solved.groupby(self.problem_col):
            vals = pd.to_numeric(g["PAC_e_val"], errors="coerce").dropna()
            if len(vals) > 0:
                group_max.append(vals.max())

        n = len(group_max)
        w0, w1 = self.alpha_delta_weights
        thresholds = {}
        for a in self.alphas:
            alpha_p, delta_p = w0 * a, w1 * a
            thr = self._upper_tolerance_bound(group_max, alpha=alpha_p, delta=delta_p)
            if np.isinf(thr):
                # warning on insufficient calibration set size
                min_n = int(np.ceil(np.log(delta_p) / np.log(1.0 - alpha_p)))
                warnings.warn(
                    f"The calibration set has only n={n} successful trajectories, which is "
                    f"too small for the PAC threshold at target alpha={a} (split as "
                    f"alpha'={alpha_p}, delta'={delta_p} via alpha_delta_weights={self.alpha_delta_weights}). "
                    f"The threshold is set to inf (no rejections; false alarm rate controlled "
                    f"but zero power). To obtain a finite threshold, ensure n >= "
                    f"ceil(log(delta') / log(1 - alpha')) = {min_n}, or increase the "
                    f"calibration set size. "
                    f"See https://arxiv.org/pdf/2512.03109 Appendix 8.1.2 for details.",
                    UserWarning,
                    stacklevel=3,
                )
            thresholds[a] = thr
        return thresholds

    def fit(self, calib_df: pd.DataFrame):
        """
        Fit one model per step and compute per-alpha thresholds.

        calib_df must contain:
            - self.step_col
            - "solved"
            - "judge_probability_series"
            - self.problem_col (for PAC threshold computation)
        """
        calib_df = calib_df.copy()

        if self.mt_variant in {"Ville", "RandVille"}:
            ## Train on full calib; Ville/RandVille threshold is 1/alpha
            self._fit_step_models(calib_df, which="anytime")
            self.thresholds = {a: 1.0 / a for a in self.alphas}
            return

        ## PAC (recommended): split calib into log half (model training) and cal half (threshold)
        unique_ids = calib_df[self.problem_col].unique()
        rng = np.random.default_rng(self.random_state)
        n_log = max(1, int(len(unique_ids) * self.split_fraction))
        log_ids = rng.choice(unique_ids, size=n_log, replace=False)
        cal_ids = np.setdiff1d(unique_ids, log_ids)

        log_df = calib_df[calib_df[self.problem_col].isin(log_ids)].reset_index(drop=True)
        cal_df = calib_df[calib_df[self.problem_col].isin(cal_ids)].reset_index(drop=True)

        self._fit_step_models(log_df, which="split")
        self.thresholds = self._compute_pac_thresholds(cal_df)

    def apply(self, df: pd.DataFrame, compute_rejects: bool = True) -> pd.DataFrame:
        """
        Compute e-values per row.

        df must contain:
            - self.step_col
            - self.problem_col
            - "judge_probability_series"

        Returns a copy of df with:
            - "Ville_e_val"  if mt_variant in {"Ville", "RandVille"}
            - "PAC_e_val"    if mt_variant == "PAC"
            - reject columns (if compute_rejects=True):
                * "reject_Ville_alpha_{a}"     for Ville
                * "reject_PAC_alpha_{a}"       for PAC
                * "reject_RandVille_alpha_{a}" for RandVille
        """
        df = df.copy()

        if self.mt_variant in {"Ville", "RandVille"}:
            df["Ville_e_val"] = self._compute_e_vals_for_variant(df, which="anytime")
        else:
            df["PAC_e_val"] = self._compute_e_vals_for_variant(df, which="split")

        if not compute_rejects:
            return df

        if self.mt_variant == "Ville":
            for a in self.alphas:
                thr = self.thresholds[a]
                base = str(a).replace(".", "_")
                df[f"reject_Ville_alpha_{base}"] = df["Ville_e_val"] > thr

        elif self.mt_variant == "PAC":
            for a in self.alphas:
                thr = self.thresholds[a]
                base = str(a).replace(".", "_")
                df[f"reject_PAC_alpha_{base}"] = df["PAC_e_val"] > thr

        else:  ## RandVille: standard threshold 1/alpha on non-final steps, U/alpha on final step
            max_step_by_pid = df.groupby(self.problem_col)[self.step_col].max().to_dict()
            rng_rv = np.random.default_rng(self.random_state)
            U_per_pid = {pid: rng_rv.uniform(0, 1) for pid in df[self.problem_col].unique()}

            T_col = df[self.problem_col].map(max_step_by_pid)
            U_col = df[self.problem_col].map(U_per_pid)
            is_final = df[self.step_col] == T_col

            for a in self.alphas:
                base = str(a).replace(".", "_")
                thr = np.where(is_final, U_col / a, 1.0 / a)
                df[f"reject_RandVille_alpha_{base}"] = df["Ville_e_val"] > thr

        return df
