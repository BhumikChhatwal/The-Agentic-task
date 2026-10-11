"""Tool 4 — inferential statistics.

Distinct from the others because it produces *inferences with uncertainty*: a test
statistic, a p-value, an effect size and a confidence interval, plus an
interpretation gated on the significance level. Aggregation (SQL) tells you the
means differ; this tool tells you whether that difference is distinguishable from
noise, and how big it is.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..errors import ToolError
from ..formatting import DataTable
from .base import (
    Tool,
    ToolContext,
    as_float,
    as_int,
    optional,
    require,
    require_numeric_column,
    resolve_column,
)

TESTS: dict[str, str] = {
    "ttest_ind": "Welch's t-test: compare the mean of a numeric column between two independent groups.",
    "ttest_rel": "Paired t-test: compare two numeric columns measured on the same rows.",
    "ttest_1samp": "One-sample t-test: is the mean of a numeric column different from a constant?",
    "mannwhitneyu": "Mann-Whitney U: non-parametric comparison of two independent groups.",
    "wilcoxon": "Wilcoxon signed-rank: non-parametric paired comparison of two columns.",
    "anova": "One-way ANOVA: compare the mean of a numeric column across 3+ groups.",
    "kruskal": "Kruskal-Wallis: non-parametric one-way comparison across 3+ groups.",
    "chi_square": "Chi-square test of independence between two categorical columns.",
    "pearson": "Pearson correlation + p-value between two numeric columns.",
    "spearman": "Spearman rank correlation (monotonic, robust to outliers).",
    "kendall": "Kendall tau correlation (rank-based, small samples).",
    "normality": "Shapiro-Wilk normality test for one numeric column.",
    "proportion": "Two-proportion z-test between two groups of a binary outcome.",
}

_NEEDS_GROUP = {"ttest_ind", "mannwhitneyu", "anova", "kruskal", "proportion"}
_NEEDS_TWO_NUMERIC = {"ttest_rel", "wilcoxon", "pearson", "spearman", "kendall"}
_NEEDS_ONE_NUMERIC = {"ttest_1samp", "normality"}
_NEEDS_TWO_CATEGORICAL = {"chi_square"}


class StatisticalTestTool(Tool):
    name = "statistical_test"
    category = "inference"
    description = (
        "Run a hypothesis test or correlation with a p-value, effect size and confidence "
        "interval, and a plain-language conclusion. Tests: "
        + ", ".join(f"{name} ({desc.split(':')[0]})" for name, desc in TESTS.items())
        + ". Use it to decide whether a difference or relationship is real, and to quantify "
        "how strong it is. `group_column` splits rows into two groups for ttest_ind/"
        "mannwhitneyu; `value_column` is the numeric outcome; for paired/correlation tests "
        "pass two numeric columns in `columns`."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "test": {
                "type": "string",
                "enum": list(TESTS),
                "description": "Which test to run.",
            },
            "dataset": {"type": "string", "description": "Dataset name (optional if only one)."},
            "value_column": {
                "type": "string",
                "description": "The numeric outcome column (not needed for chi_square).",
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exactly two columns for paired/correlation tests (ttest_rel, wilcoxon, "
                    "pearson, spearman, kendall) or the two categorical columns for chi_square."
                ),
            },
            "group_column": {
                "type": "string",
                "description": "Categorical column that defines the groups, for ttest_ind/mannwhitneyu/anova/kruskal/proportion.",
            },
            "group_a": {"type": "string", "description": "First group label (optional; default the two most common)."},
            "group_b": {"type": "string", "description": "Second group label (optional)."},
            "popmean": {
                "type": "number",
                "description": "The constant to compare against for ttest_1samp (default 0).",
            },
            "alpha": {"type": "number", "description": "Significance level (default 0.05)."},
            "alternative": {
                "type": "string",
                "enum": ["two-sided", "greater", "less"],
                "description": "Alternative hypothesis for t-tests (default two-sided).",
            },
        },
        "required": ["test"],
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        test = str(require(kwargs, "test")).strip().lower()
        if test not in TESTS:
            raise ToolError(f"unknown test {test!r}; choose one of: {', '.join(TESTS)}")
        dataset = ctx.dataset(optional(kwargs, "dataset"))
        frame = dataset.frame
        alpha = as_float(optional(kwargs, "alpha", 0.05), "alpha")
        alternative = str(optional(kwargs, "alternative", "two-sided"))

        if test in _NEEDS_GROUP:
            return self._group_test(ctx, dataset, frame, test, kwargs, alpha, alternative)
        if test in _NEEDS_TWO_NUMERIC:
            return self._paired_test(dataset, frame, test, kwargs, alpha)
        if test in _NEEDS_ONE_NUMERIC:
            return self._one_sample(dataset, frame, test, kwargs, alpha, alternative)
        return self._chi_square(dataset, frame, kwargs, alpha)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _conclusion(p_value: float, alpha: float, significant_text: str, null_text: str) -> str:
        if p_value is None or (isinstance(p_value, float) and math.isnan(p_value)):
            return "could not compute a p-value"
        return significant_text if p_value < alpha else null_text

    @staticmethod
    def _clean(*series: pd.Series) -> tuple[pd.Series, ...]:
        frame = pd.concat(series, axis=1).dropna()
        return tuple(frame[col] for col in frame.columns)

    # -- test families ----------------------------------------------------
    def _group_test(self, ctx, dataset, frame, test, kwargs, alpha, alternative):
        group_column = resolve_column(frame, str(require(kwargs, "group_column")))
        groups = frame[group_column].dropna()
        labels = list(pd.Series(groups).value_counts().index)

        if test == "proportion":
            value_column = resolve_column(frame, str(require(kwargs, "value_column")))
            binary = frame[value_column].dropna().unique()
            if len(binary) > 2:
                raise ToolError(
                    f"proportion test needs a binary outcome; {value_column!r} has {len(binary)} distinct values"
                )
            if len(labels) < 2:
                raise ToolError(f"{group_column!r} has fewer than two groups")
            a_label = str(optional(kwargs, "group_a", labels[0]))
            b_label = str(optional(kwargs, "group_b", labels[1]))
            a = frame.loc[frame[group_column].astype(str) == a_label, value_column].dropna()
            b = frame.loc[frame[group_column].astype(str) == b_label, value_column].dropna()
            if not len(a) or not len(b):
                raise ToolError(f"one of the groups {a_label!r}/{b_label!r} has no rows")
            success = binary[1] if len(binary) > 1 else binary[0]
            count = np.array([(a == success).sum(), (b == success).sum()], dtype=float)
            nobs = np.array([len(a), len(b)], dtype=float)
            statistic, p_value = _proportions_ztest(count, nobs, alternative=alternative)
            rate_a, rate_b = count[0] / nobs[0], count[1] / nobs[1]
            pooled = (count[0] + count[1]) / (nobs[0] + nobs[1])
            se = math.sqrt(pooled * (1 - pooled) * (1 / nobs[0] + 1 / nobs[1]))
            diff = rate_a - rate_b
            ci = (diff - 1.96 * se, diff + 1.96 * se)
            return self.result(
                self._conclusion(
                    p_value, alpha,
                    f"{success!r} rate differs significantly between {a_label!r} ({rate_a:.1%}) "
                    f"and {b_label!r} ({rate_b:.1%}), p={p_value:.4g}",
                    f"no significant difference in {success!r} rate between {a_label!r} ({rate_a:.1%}) "
                    f"and {b_label!r} ({rate_b:.1%}), p={p_value:.4g}",
                ),
                detail={
                    "test": "two-proportion z-test",
                    "dataset": dataset.name,
                    "group_a": a_label, "group_b": b_label,
                    "rate_a": rate_a, "rate_b": rate_b, "difference": diff,
                    "statistic_z": float(statistic), "p_value": float(p_value),
                    "ci95_difference": list(ci), "n_a": int(nobs[0]), "n_b": int(nobs[1]),
                    "alpha": alpha,
                },
            )

        value_column = require_numeric_column(frame, str(require(kwargs, "value_column")))
        two_group = test in {"ttest_ind", "mannwhitneyu"}
        required = 2 if two_group else 3
        if len(labels) < (2 if two_group else 2):
            raise ToolError(f"{group_column!r} needs at least two groups; found {len(labels)}")
        if not two_group and len(labels) < required:
            raise ToolError(f"{test} needs at least 3 groups; {group_column!r} has {len(labels)}")

        if two_group:
            a_label = str(optional(kwargs, "group_a", labels[0]))
            b_label = str(optional(kwargs, "group_b", labels[1]))
            a = frame.loc[frame[group_column].astype(str) == a_label, value_column].dropna()
            b = frame.loc[frame[group_column].astype(str) == b_label, value_column].dropna()
            if len(a) < 2 or len(b) < 2:
                raise ToolError(
                    f"need at least 2 observations per group; got {a_label}={len(a)}, {b_label}={len(b)}"
                )
            # Tiny or constant groups make a t-test and Cohen's d unreliable; scipy
            # emits a warning, which we replace with an explicit caveat of our own.
            caveats = _reliability_caveats(a, b, a_label, b_label)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if test == "ttest_ind":
                    statistic, p_value = sps.ttest_ind(a, b, equal_var=False, alternative=alternative)
                    effect = _cohens_d(a, b)
                    effect_name = "cohens_d"
                else:
                    statistic, p_value = sps.mannwhitneyu(a, b, alternative=alternative)
                    effect = _rank_biserial(a, b)
                    effect_name = "rank_biserial_r"
            ci = _mean_diff_ci(a, b)
            rows = [
                [a_label, len(a), round(float(a.mean()), 6), round(float(a.median()), 6), round(float(a.std(ddof=1)), 6)],
                [b_label, len(b), round(float(b.mean()), 6), round(float(b.median()), 6), round(float(b.std(ddof=1)), 6)],
            ]
            table = DataTable(["group", "n", "mean", "median", "sd"], rows)
            detail = {
                "test": test, "dataset": dataset.name, "value_column": value_column,
                "group_column": group_column, "statistic": float(statistic),
                "p_value": float(p_value), effect_name: effect, "alpha": alpha,
                "mean_difference": float(a.mean() - b.mean()),
                "ci95_mean_difference": list(ci),
                "n_a": int(len(a)), "n_b": int(len(b)),
            }
            if caveats:
                detail["caveats"] = caveats
            message = self._conclusion(
                p_value, alpha,
                f"{value_column} differs significantly between {a_label!r} (mean {a.mean():.4g}) and "
                f"{b_label!r} (mean {b.mean():.4g}): {test} p={p_value:.4g}, {effect_name}={effect:.3f}",
                f"no significant difference in {value_column} between {a_label!r} (mean {a.mean():.4g}) "
                f"and {b_label!r} (mean {b.mean():.4g}): p={p_value:.4g}",
            )
            if caveats:
                message += " Caveat: " + " ".join(caveats)
            return self.result(message, table=table, detail=detail)

        # 3+ groups
        grouped = [
            (str(label), frame.loc[frame[group_column].astype(str) == str(label), value_column].dropna())
            for label in labels
        ]
        grouped = [(label, values) for label, values in grouped if len(values) >= 2]
        if len(grouped) < 2:
            raise ToolError("fewer than two groups have enough observations")
        samples = [values.to_numpy() for _, values in grouped]
        if test == "anova":
            statistic, p_value = sps.f_oneway(*samples)
            eta_sq = _eta_squared(samples)
            effect_name, effect = "eta_squared", eta_sq
        else:
            statistic, p_value = sps.kruskal(*samples)
            k = len(samples)
            n = sum(len(s) for s in samples)
            effect = (statistic - k + 1) / (n - k) if n > k else float("nan")
            effect_name = "epsilon_squared"
        table = DataTable(
            ["group", "n", "mean", "median", "sd"],
            [
                [label, len(values), round(float(values.mean()), 6), round(float(values.median()), 6),
                 round(float(values.std(ddof=1)), 6)]
                for label, values in grouped
            ],
        )
        return self.result(
            self._conclusion(
                p_value, alpha,
                f"{value_column} differs significantly across {len(grouped)} levels of {group_column} "
                f"({test} p={p_value:.4g}, {effect_name}={effect:.3f})",
                f"no significant difference in {value_column} across levels of {group_column} "
                f"(p={p_value:.4g}); observed differences are consistent with noise",
            ),
            table=table,
            detail={
                "test": test, "dataset": dataset.name, "value_column": value_column,
                "group_column": group_column, "statistic": float(statistic), "p_value": float(p_value),
                effect_name: float(effect) if effect == effect else None,
                "groups": [label for label, _ in grouped], "alpha": alpha,
            },
        )

    def _paired_test(self, dataset, frame, test, kwargs, alpha):
        columns = kwargs.get("columns") or []
        if isinstance(columns, str):
            columns = [columns]
        if len(columns) != 2:
            raise ToolError(f"{test} requires exactly two columns in `columns`")
        left = resolve_column(frame, str(columns[0]))
        right = resolve_column(frame, str(columns[1]))
        if test in {"pearson", "spearman", "kendall"}:
            a = pd.to_numeric(frame[left], errors="coerce")
            b = pd.to_numeric(frame[right], errors="coerce")
        else:
            a = frame[left]
            b = frame[right]
        pairs = pd.concat([a, b], axis=1).dropna()
        a_vals, b_vals = pairs.iloc[:, 0], pairs.iloc[:, 1]
        if len(pairs) < 3:
            raise ToolError(f"need at least 3 complete pairs; found {len(pairs)}")

        if test == "ttest_rel":
            statistic, p_value = sps.ttest_rel(a_vals, b_vals)
            effect = _cohens_d(a_vals, b_vals)
            effect_name = "cohens_d"
            label = f"{left} vs {right} (paired)"
        elif test == "wilcoxon":
            try:
                statistic, p_value = sps.wilcoxon(a_vals, b_vals)
            except ValueError as exc:
                raise ToolError(f"wilcoxon could not run: {exc}") from exc
            effect = _rank_biserial(a_vals, b_vals)
            effect_name = "rank_biserial_r"
            label = f"{left} vs {right} (paired)"
        elif test == "pearson":
            statistic, p_value = sps.pearsonr(a_vals, b_vals)
            effect, effect_name = float(statistic), "pearson_r"
            label = f"{left} vs {right}"
        elif test == "spearman":
            statistic, p_value = sps.spearmanr(a_vals, b_vals)
            effect, effect_name = float(statistic), "spearman_rho"
            label = f"{left} vs {right}"
        else:  # kendall
            statistic, p_value = sps.kendalltau(a_vals, b_vals)
            effect, effect_name = float(statistic), "kendall_tau"
            label = f"{left} vs {right}"

        strength = _describe_strength(effect)
        if test in {"pearson", "spearman", "kendall"}:
            message = self._conclusion(
                p_value, alpha,
                f"{label} show a {strength} {effect_name}={effect:.3f} (p={p_value:.4g}, n={len(pairs)})",
                f"{label} show no significant correlation ({effect_name}={effect:.3f}, p={p_value:.4g}, n={len(pairs)})",
            )
        else:
            message = self._conclusion(
                p_value, alpha,
                f"{left} and {right} differ significantly (mean difference {float(a_vals.mean() - b_vals.mean()):.4g}, "
                f"p={p_value:.4g}, {effect_name}={effect:.3f})",
                f"no significant difference between {left} and {right} (p={p_value:.4g})",
            )
        detail = {
            "test": test, "dataset": dataset.name, "columns": [left, right],
            "statistic": float(statistic), "p_value": float(p_value),
            effect_name: float(effect) if effect == effect else None,
            "n_pairs": int(len(pairs)), "alpha": alpha,
            "interpretation_strength": strength,
        }
        if test in {"pearson", "spearman", "kendall"}:
            detail["r_squared"] = round(float(effect) ** 2, 6)
        return self.result(message, detail=detail)

    def _one_sample(self, dataset, frame, test, kwargs, alpha, alternative):
        column = require_numeric_column(frame, str(require(kwargs, "value_column")))
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values) < 3:
            raise ToolError(f"need at least 3 numeric observations; found {len(values)}")
        if test == "normality":
            if len(values) > 5000:
                values = values.sample(5000, random_state=0)
            statistic, p_value = sps.shapiro(values)
            message = self._conclusion(
                p_value, alpha,
                f"{column} departs significantly from normality (Shapiro-Wilk W={statistic:.4f}, p={p_value:.4g}); "
                "prefer non-parametric tests (mannwhitneyu / spearman / kruskal)",
                f"{column} is consistent with a normal distribution (Shapiro-Wilk W={statistic:.4f}, p={p_value:.4g})",
            )
            return self.result(
                message,
                detail={"test": "shapiro-wilk", "dataset": dataset.name, "column": column,
                        "statistic": float(statistic), "p_value": float(p_value),
                        "n": int(len(values)), "alpha": alpha},
            )
        popmean = as_float(optional(kwargs, "popmean", 0.0), "popmean")
        statistic, p_value = sps.ttest_1samp(values, popmean, alternative=alternative)
        se = values.std(ddof=1) / math.sqrt(len(values))
        ci = (values.mean() - 1.96 * se, values.mean() + 1.96 * se)
        return self.result(
            self._conclusion(
                p_value, alpha,
                f"{column} mean {values.mean():.4g} differs significantly from {popmean} "
                f"(p={p_value:.4g}, 95% CI [{ci[0]:.4g}, {ci[1]:.4g}])",
                f"{column} mean {values.mean():.4g} is not significantly different from {popmean} (p={p_value:.4g})",
            ),
            detail={"test": "one-sample t-test", "dataset": dataset.name, "column": column,
                    "mean": float(values.mean()), "popmean": popmean,
                    "statistic": float(statistic), "p_value": float(p_value),
                    "ci95_mean": list(ci), "n": int(len(values)), "alpha": alpha},
        )

    def _chi_square(self, dataset, frame, kwargs, alpha):
        columns = kwargs.get("columns") or []
        if isinstance(columns, str):
            columns = [columns]
        if len(columns) != 2:
            raise ToolError("chi_square requires exactly two categorical columns in `columns`")
        left = resolve_column(frame, str(columns[0]))
        right = resolve_column(frame, str(columns[1]))
        observed = pd.crosstab(frame[left], frame[right])
        if observed.shape[0] < 2 or observed.shape[1] < 2:
            raise ToolError("chi_square needs at least a 2x2 contingency table")
        statistic, p_value, dof, expected = sps.chi2_contingency(observed)
        n = observed.to_numpy().sum()
        cramers_v = math.sqrt(statistic / (n * min(observed.shape[0] - 1, observed.shape[1] - 1)))
        table = DataTable(
            [observed.index.name or "row"] + [str(c) for c in observed.columns],
            [[str(idx)] + [int(v) for v in row] for idx, row in observed.iterrows()],
            total_rows=len(observed),
        )
        return self.result(
            self._conclusion(
                p_value, alpha,
                f"{left} and {right} are significantly associated (chi2={statistic:.4g}, dof={dof}, "
                f"p={p_value:.4g}, Cramer's V={cramers_v:.3f})",
                f"no significant association between {left} and {right} (p={p_value:.4g}); "
                "the two columns look independent",
            ),
            table=table,
            detail={"test": "chi-square", "dataset": dataset.name, "columns": [left, right],
                    "statistic": float(statistic), "dof": int(dof), "p_value": float(p_value),
                    "cramers_v": round(cramers_v, 4), "n": int(n), "alpha": alpha},
        )


# ---------------------------------------------------------------------------
# Effect-size and interval helpers
# ---------------------------------------------------------------------------


def _reliability_caveats(a: pd.Series, b: pd.Series, a_label: str, b_label: str) -> list[str]:
    """Flag conditions under which the reported numbers should not be trusted."""
    caveats: list[str] = []
    for label, values in ((a_label, a), (b_label, b)):
        if len(values) < 5:
            caveats.append(f"`{label}` has only {len(values)} observation(s).")
        if len(values) > 1 and float(values.std(ddof=1)) == 0.0:
            caveats.append(f"every `{label}` value is identical, so the effect size is undefined.")
    if abs(float(a.mean() - b.mean())) > 0 and len(a) > 1 and len(b) > 1:
        ratio = max(abs(float(a.mean())), abs(float(b.mean()))) / max(
            1e-12, min(abs(float(a.mean())), abs(float(b.mean())))
        )
        if ratio > 100:
            caveats.append("the group means differ by orders of magnitude — check for data-entry errors.")
    return caveats


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / math.sqrt(pooled_var))


def _rank_biserial(a: pd.Series, b: pd.Series) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    try:
        u, _ = sps.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return float("nan")
    return float((2 * u) / (len(a) * len(b)) - 1)


def _mean_diff_ci(a: pd.Series, b: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0 or not math.isfinite(se):
        return (float(diff), float(diff))
    crit = sps.t.ppf(0.5 + confidence / 2, df=min(len(a), len(b)) - 1)
    return (float(diff - crit * se), float(diff + crit * se))


def _eta_squared(samples: list[np.ndarray]) -> float:
    groups = [np.asarray(s, dtype=float) for s in samples]
    all_values = np.concatenate(groups)
    grand = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((all_values - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def _proportions_ztest(count: np.ndarray, nobs: np.ndarray, alternative: str = "two-sided"):
    """Two-proportion z-test (avoids a statsmodels dependency for this one case)."""
    p1, p2 = count[0] / nobs[0], count[1] / nobs[1]
    p_pool = count.sum() / nobs.sum()
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    if alternative == "two-sided":
        p_value = 2 * (1 - sps.norm.cdf(abs(z)))
    elif alternative == "greater":
        p_value = 1 - sps.norm.cdf(z)
    else:
        p_value = sps.norm.cdf(z)
    return float(z), float(p_value)


def _describe_strength(effect: float) -> str:
    if effect != effect:  # NaN
        return "undefined"
    magnitude = abs(effect)
    if magnitude < 0.1:
        return "negligible"
    if magnitude < 0.3:
        return "weak"
    if magnitude < 0.5:
        return "moderate"
    if magnitude < 0.7:
        return "strong"
    return "very strong"
