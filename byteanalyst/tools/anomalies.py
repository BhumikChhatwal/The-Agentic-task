"""Tool 5 — anomaly detection.

Distinct because it looks for the *exceptions*, not the summary: which rows do not
fit the pattern. Five methods with genuinely different statistical machinery:

* ``iqr``               — Tukey fences on a single column (robust, univariate).
* ``zscore``            — standard deviations from the mean (assumes normality).
* ``mad``               — modified z-score on median absolute deviation (robust).
* ``isolation_forest``  — multivariate density-based detection across many columns.
* ``seasonal_residual`` — residual spikes left after removing trend + weekly seasonality.
* ``grouped``           — the same test applied *within* each group, so that
                          "unusual for the North region" is not masked by a regional
                          level difference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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

METHODS = ("iqr", "zscore", "mad", "isolation_forest", "seasonal_residual")


class AnomalyTool(Tool):
    name = "detect_anomalies"
    category = "detection"
    description = (
        "Find the unusual rows in a table: which records are outliers and why. Methods: "
        "iqr (Tukey fences), zscore, mad (median absolute deviation, robust), "
        "isolation_forest (multivariate, across several columns at once), and "
        "seasonal_residual (spikes that survive removing trend + seasonality — use this for "
        "seasonal series, where a plain threshold hides real extremes). "
        "Pass `group_by` to test within each group (so a regional level shift does not mask "
        "local outliers, and so interleaved entities are not averaged together). "
        "Pass `columns` (list) for isolation_forest; `column` (single) for the others."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "description": "Dataset name (optional if only one)."},
            "column": {"type": "string", "description": "Numeric column to test (univariate methods)."},
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple numeric columns for isolation_forest.",
            },
            "method": {"type": "string", "enum": list(METHODS), "description": "Detection method."},
            "group_by": {
                "type": "string",
                "description": "Categorical column: detect outliers within each group.",
            },
            "time_column": {
                "type": "string",
                "description": "Datetime column, required for seasonal_residual.",
            },
            "threshold": {
                "type": "number",
                "description": "IQR multiplier (default 1.5), z/MAD cutoff (default 3.0).",
            },
            "contamination": {
                "type": "number",
                "description": "Expected outlier fraction for isolation_forest (default 0.05).",
            },
            "top_k": {"type": "integer", "description": "How many anomalies to list (default 20)."},
        },
        "required": ["method"],
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        method = str(require(kwargs, "method")).strip().lower()
        if method not in METHODS:
            raise ToolError(f"unknown method {method!r}; choose one of: {', '.join(METHODS)}")
        dataset = ctx.dataset(optional(kwargs, "dataset"))
        frame = dataset.frame
        top_k = max(1, min(as_int(optional(kwargs, "top_k", 20), "top_k"), 200))
        group_by = optional(kwargs, "group_by")

        if method == "isolation_forest":
            return self._isolation_forest(dataset, frame, kwargs, top_k, group_by)
        if method == "seasonal_residual":
            return self._seasonal(dataset, frame, kwargs, top_k)

        column = require_numeric_column(frame, str(require(kwargs, "column")))
        threshold = as_float(
            optional(kwargs, "threshold", 3.0 if method in {"zscore", "mad"} else 1.5), "threshold"
        )

        if group_by:
            group_column = resolve_column(frame, str(group_by))
            flags = pd.Series(False, index=frame.index)
            scores = pd.Series(np.nan, index=frame.index)
            for _, idx in frame.groupby(group_column, dropna=True).groups.items():
                subset = frame.loc[idx, column].astype(float)
                mask, score = self._score(subset, method, threshold)
                flags.loc[idx] = mask
                scores.loc[idx] = score
            flagged = frame.loc[flags]
            scores = scores.loc[flags]
        else:
            mask, score = self._score(frame[column].astype(float), method, threshold)
            flagged = frame.loc[mask]
            scores = score[mask]

        return self._finish(
            dataset, frame, method, column, flagged, scores, top_k,
            threshold=threshold, group_by=group_by,
            extra_note=self._seasonality_hint(dataset, column),
        )

    # -- univariate scoring ----------------------------------------------
    @staticmethod
    def _score(values: pd.Series, method: str, threshold: float) -> tuple[pd.Series, pd.Series]:
        numeric = pd.to_numeric(values, errors="coerce")
        if method == "iqr":
            q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                median = numeric.median()
                score = (numeric - median).abs()
                return score > (threshold * (numeric.std(ddof=0) or 1.0)), score
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
            score = np.maximum((lower - numeric) / iqr, (numeric - upper) / iqr)
            return numeric.lt(lower) | numeric.gt(upper), score
        if method == "zscore":
            mean, sd = numeric.mean(), numeric.std(ddof=0)
            if not sd or not np.isfinite(sd):
                return pd.Series(False, index=numeric.index), pd.Series(0.0, index=numeric.index)
            score = ((numeric - mean) / sd).abs()
            return score > threshold, score
        if method == "mad":
            median = numeric.median()
            mad = (numeric - median).abs().median()
            if not mad or not np.isfinite(mad):
                return pd.Series(False, index=numeric.index), pd.Series(0.0, index=numeric.index)
            score = 0.6745 * (numeric - median).abs() / mad
            return score > threshold, score
        raise ToolError(f"unsupported univariate method {method!r}")

    # -- multivariate -----------------------------------------------------
    def _isolation_forest(self, dataset, frame, kwargs, top_k, group_by):
        columns = kwargs.get("columns") or []
        if isinstance(columns, str):
            columns = [columns]
        if not columns:
            columns = dataset.numeric_columns()
        if len(columns) < 1:
            raise ToolError("isolation_forest needs at least one numeric column")
        resolved = [require_numeric_column(frame, str(c)) for c in columns]
        if len(resolved) < 1:
            raise ToolError("no usable numeric columns")
        contamination = as_float(optional(kwargs, "contamination", 0.05), "contamination")
        contamination = max(0.001, min(contamination, 0.5))

        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        data = frame[resolved].apply(pd.to_numeric, errors="coerce")
        mask_complete = data.notna().all(axis=1)
        if int(mask_complete.sum()) < 10:
            raise ToolError(
                "isolation_forest needs at least 10 complete rows across the chosen columns"
            )
        usable = data.loc[mask_complete]
        scaled = StandardScaler().fit_transform(usable.to_numpy(dtype=float))
        model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
        )
        labels = model.fit_predict(scaled)
        scores = -model.score_samples(scaled)  # higher = more anomalous
        flagged_index = usable.index[labels == -1]
        score_series = pd.Series(scores, index=usable.index)

        if group_by:
            group_column = resolve_column(frame, str(group_by))
            # Keep an anomaly only if it is unusual for its own group too.
            keep = []
            for group, idx in frame.loc[flagged_index].groupby(group_column, dropna=True).groups.items():
                local_scores = score_series.loc[idx]
                if len(local_scores) == 0:
                    continue
                cutoff = max(local_scores.quantile(0.5), 0.0)
                keep.extend(local_scores.sort_values(ascending=False).head(max(1, len(idx) // 2)).index)
            flagged_index = pd.Index(sorted(set(keep)))

        flagged = frame.loc[flagged_index]
        return self._finish(
            dataset, frame, "isolation_forest", " + ".join(resolved), flagged,
            score_series.loc[flagged_index], top_k,
            contamination=contamination, columns=resolved, group_by=group_by,
        )

    # -- seasonal ---------------------------------------------------------
    def _seasonal(self, dataset, frame, kwargs, top_k):
        """Remove trend + seasonality, then flag the residual spikes.

        A heatwave is invisible to IQR when the column also has a large annual
        cycle: the seasonal swing is wider than the event. Decomposing first (and
        optionally doing it per group, so three cities are not interleaved into
        one meaningless series) is what makes those events detectable.
        """
        import statsmodels.api as sm

        time_column = resolve_column(frame, str(require(kwargs, "time_column")))
        value_column = require_numeric_column(frame, str(require(kwargs, "column")))
        threshold = as_float(optional(kwargs, "threshold", 3.0), "threshold")
        group_by = optional(kwargs, "group_by")
        period = _infer_period_from_span(frame[time_column])

        if group_by:
            group_column = resolve_column(frame, str(group_by))
            groups = [
                (str(label), group)
                for label, group in frame.groupby(group_column, dropna=True)
            ]
        else:
            groups = [(None, frame)]

        flagged_parts: list[pd.DataFrame] = []
        score_parts: list[pd.Series] = []
        tested = 0
        skipped: list[str] = []
        for label, group in groups:
            working = group[[time_column, value_column]].dropna().copy()
            working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
            working = working.dropna(subset=[time_column])
            if working.empty:
                skipped.append(label or value_column)
                continue
            # Collapse duplicates at the same timestamp (e.g. several rows per day)
            # so the series has one value per period.
            series = (
                working.groupby(time_column)[value_column].mean().astype(float).sort_index()
            )
            if len(series) < 2 * period:
                skipped.append(label or value_column)
                continue
            residual = None
            try:
                decomposition = _decompose(series, period)
                residual = pd.Series(decomposition.resid, index=series.index).dropna()
            except Exception:
                residual = None
            if residual is None or residual.empty:
                continue
            sd = residual.std(ddof=0)
            if not sd or not np.isfinite(sd):
                continue
            tested += len(residual)
            score = (residual / sd).abs()
            hits = set(score.index[score > threshold])
            if not hits:
                continue
            hit_rows = group.loc[working.index.intersection(group.index)]
            hit_rows = hit_rows.loc[pd.to_datetime(hit_rows[time_column]).isin(hits)]
            if hit_rows.empty:
                continue
            flagged_parts.append(hit_rows)
            score_parts.append(
                pd.Series(
                    [float(score.loc[stamp]) for stamp in pd.to_datetime(hit_rows[time_column])],
                    index=hit_rows.index,
                )
            )

        if not flagged_parts:
            detail = {
                "dataset": dataset.name, "method": "seasonal_residual",
                "column": value_column, "anomalies_found": 0, "period": period,
                "observations_tested": tested,
            }
            if skipped:
                detail["series_skipped_too_short"] = ", ".join(skipped[:8])
            return self.result(
                f"No residual spikes in `{value_column}` above {threshold} standard deviations "
                f"after removing trend and a {period}-period seasonal cycle "
                f"({tested:,} observations tested).",
                detail=detail,
            )

        flagged = pd.concat(flagged_parts)
        scores = pd.concat(score_parts)
        return self._finish(
            dataset, frame, "seasonal_residual", value_column, flagged, scores, top_k,
            threshold=threshold, time_column=time_column, period=period,
            group_by=group_by, observations_tested=tested,
        )

    # -- shared rendering -------------------------------------------------
    def _finish(
        self,
        dataset,
        frame,
        method: str,
        column_label: str,
        flagged: pd.DataFrame,
        scores: pd.Series,
        top_k: int,
        extra_note: str = "",
        **extra: Any,
    ) -> Any:
        total = len(frame)
        found = len(flagged)
        detail: dict[str, Any] = {
            "dataset": dataset.name,
            "method": method,
            "column": column_label,
            "rows_tested": total,
            "anomalies_found": found,
            "anomaly_rate": round(found / total, 5) if total else 0.0,
        }
        detail.update({k: v for k, v in extra.items() if v is not None})

        if not found:
            return self.result(
                f"No anomalies detected in `{column_label}` of `{dataset.name}` "
                f"using {method} across {total:,} rows." + extra_note,
                detail=detail,
            )

        ranked = flagged.assign(__score=scores.reindex(flagged.index)).sort_values(
            "__score", ascending=False
        )
        display_columns = [c for c in frame.columns if c != "__score"][:8]
        rows = [
            [index] + [_render(row[c]) for c in display_columns] + [round(float(row["__score"]), 4)]
            for index, row in ranked.head(top_k).iterrows()
        ]
        table = DataTable(
            ["row"] + display_columns + ["anomaly_score"],
            rows,
            total_rows=found,
            truncated=found > top_k,
        )
        method_note = {
            "iqr": "IQR fences on the middle 50% (robust to the very outliers it finds)",
            "zscore": "standard deviations from the mean",
            "mad": "modified z-score using the median absolute deviation (robust)",
            "isolation_forest": "multivariate isolation depth across all supplied columns",
            "seasonal_residual": "spikes remaining after removing trend and seasonality",
        }[method]
        summary = (
            f"{found:,} anomaly(ies) in {total:,} rows ({found / total:.2%}) of `{dataset.name}` "
            f"by {method} on `{column_label}` — {method_note}. "
            f"Top {min(top_k, found)} listed, highest score first." + extra_note
        )
        return self.result(summary, table=table, detail=detail)

    @staticmethod
    def _seasonality_hint(dataset, analysed_column: str) -> str:
        """Point at the better method when a time column makes seasonality likely."""
        dates = dataset.datetime_columns()
        if not dates:
            return ""
        return (
            f" Note: this table has a time column (`{dates[0]}`); if `{analysed_column}` follows a "
            "seasonal cycle, a plain threshold can hide real spikes — try method "
            "`seasonal_residual` (optionally with `group_by`)."
        )


def _decompose(series: pd.Series, period: int):
    """Classical decomposition with a version-tolerant ``extrapolate_trend``."""
    import statsmodels.api as sm

    try:
        return sm.tsa.seasonal_decompose(series, period=period, extrapolate_trend="period")
    except (TypeError, ValueError):
        try:
            return sm.tsa.seasonal_decompose(series, period=period, extrapolate_trend="freq")
        except Exception:
            pass
    from statsmodels.tsa.seasonal import STL

    return STL(series, period=period, robust=True).fit()


def _infer_period_from_span(dates) -> int:
    """Guess the seasonal period of a series from its sampling interval."""
    series = pd.to_datetime(pd.Series(dates), errors="coerce").dropna().sort_values()
    if len(series) < 3:
        return 7
    step = (series.diff().dropna().dt.total_seconds() / 86400.0).median()
    if not step or not np.isfinite(step):
        return 7
    if step <= 0.25:
        return 24  # sub-6-hourly -> daily cycle
    if step <= 1.5:
        return 7  # daily -> weekly cycle
    if step <= 10:
        return 52  # weekly -> annual cycle
    if step <= 45:
        return 12  # monthly -> annual cycle
    return 4  # quarterly -> annual cycle


def _render(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value
