"""Tool 7 — time series.

Distinct from the others because it treats the *order* of rows as information:
resampling to a regular frequency, separating trend from seasonality, testing
stationarity, locating change points, and extrapolating. ``detect_anomalies``
answers "which rows are odd"; this tool answers "where is this heading and what
shape is it in".
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
    as_int,
    optional,
    require,
    require_numeric_column,
    resolve_column,
)

FREQS = ("auto", "h", "D", "W", "ME", "QE", "YE")
FORECAST_METHODS = ("ets", "arima", "naive", "linear")


class TimeSeriesTool(Tool):
    name = "analyze_time_series"
    category = "temporal"
    description = (
        "Analyse a metric over time: resample to a regular frequency, measure trend and slope, "
        "quantify seasonality strength, test stationarity (ADF), find change points, and "
        "optionally forecast future periods. Use this for any question about growth, decline, "
        "seasonality, 'is it trending up', 'what will next month be', or 'when did it change'. "
        "Requires a datetime column and a numeric column."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "description": "Dataset name (optional if only one)."},
            "time_column": {"type": "string", "description": "Datetime column."},
            "value_column": {"type": "string", "description": "Numeric metric column."},
            "agg": {
                "type": "string",
                "enum": ["sum", "mean", "median", "count", "max", "min"],
                "description": "How to combine rows within each time bucket (default sum).",
            },
            "freq": {"type": "string", "enum": list(FREQS), "description": "Resample frequency (default auto)."},
            "group_by": {
                "type": "string",
                "description": "Optional categorical column; analyse each series separately.",
            },
            "rolling_window": {
                "type": "integer",
                "description": "Window (in periods) for a rolling mean, e.g. 7 for a 7-day average.",
            },
            "decompose": {"type": "boolean", "description": "Split into trend/seasonal/residual (default true)."},
            "stationarity": {"type": "boolean", "description": "Run an Augmented Dickey-Fuller test (default true)."},
            "changepoints": {"type": "boolean", "description": "Look for structural breaks (default true)."},
            "forecast_periods": {
                "type": "integer",
                "description": "How many future periods to forecast (0 disables forecasting).",
            },
            "forecast_method": {
                "type": "string",
                "enum": list(FORECAST_METHODS),
                "description": "Forecast model (default ets; naive repeats the last value; linear extrapolates the trend).",
            },
        },
        "required": ["time_column", "value_column"],
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        dataset = ctx.dataset(optional(kwargs, "dataset"))
        frame = dataset.frame
        time_column = resolve_column(frame, str(require(kwargs, "time_column")))
        value_column = require_numeric_column(frame, str(require(kwargs, "value_column")))
        agg = str(optional(kwargs, "agg", "sum"))
        freq_choice = str(optional(kwargs, "freq", "auto"))
        group_by = optional(kwargs, "group_by")
        decompose = bool(optional(kwargs, "decompose", True))
        stationarity = bool(optional(kwargs, "stationarity", True))
        changepoints = bool(optional(kwargs, "changepoints", True))
        forecast_periods = as_int(optional(kwargs, "forecast_periods", 0), "forecast_periods")
        forecast_method = str(optional(kwargs, "forecast_method", "ets"))
        rolling_window = optional(kwargs, "rolling_window")
        rolling_window = as_int(rolling_window, "rolling_window") if rolling_window else None

        if agg not in {"sum", "mean", "median", "count", "max", "min"}:
            raise ToolError(f"unsupported aggregation {agg!r}")
        if forecast_method not in FORECAST_METHODS:
            raise ToolError(f"unknown forecast_method {forecast_method!r}")

        working = frame[[time_column, value_column] + ([str(group_by)] if group_by else [])].copy()
        working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
        working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
        working = working.dropna(subset=[time_column, value_column])
        if working.empty:
            raise ToolError(f"no rows with both a valid date in {time_column!r} and a number in {value_column!r}")
        if len(working) < 4:
            raise ToolError("need at least 4 time-ordered observations")

        freq = _resolve_freq(working[time_column], freq_choice)
        detail: dict[str, Any] = {
            "dataset": dataset.name,
            "time_column": time_column,
            "value_column": value_column,
            "aggregation": agg,
            "frequency": freq,
            "observations": int(len(working)),
            "date_range": f"{working[time_column].min().date()} .. {working[time_column].max().date()}",
        }

        if group_by:
            series_map = {
                str(label): _aggregate(group, time_column, value_column, freq, agg)
                for label, group in working.groupby(str(group_by), dropna=True)
            }
            series_map = {k: v for k, v in series_map.items() if len(v) >= 4}
            if not series_map:
                raise ToolError("no group had enough observations after resampling")
            rows = []
            for label, series in series_map.items():
                change = _pct_change(series)
                slope, r2 = _trend(series)
                rows.append([
                    label, len(series), round(float(series.mean()), 6),
                    round(float(series.iloc[0]), 6), round(float(series.iloc[-1]), 6),
                    round(change, 3) if change is not None else None,
                    round(slope, 6), round(r2, 4),
                ])
            table = DataTable(
                ["series", "periods", "mean", "first", "last", "change_pct", "slope_per_period", "r_squared"],
                rows, total_rows=len(rows),
            )
            rising = [r[0] for r in rows if r[6] and r[6] > 0]
            detail["series_compared"] = len(rows)
            detail["series_rising"] = len(rising)
            return self.result(
                f"Compared {len(rows)} series of `{value_column}` by `{group_by}` at {freq} frequency. "
                f"{len(rising)} are trending upward (positive slope): {', '.join(rising[:8])}"
                + ("..." if len(rising) > 8 else "."),
                table=table, detail=detail,
            )

        series = _aggregate(working, time_column, value_column, freq, agg)
        if len(series) < 4:
            raise ToolError(
                f"only {len(series)} bucket(s) after resampling to {freq}; use a coarser `freq` or more data"
            )

        detail.update(
            periods=int(len(series)),
            first_value=round(float(series.iloc[0]), 6),
            last_value=round(float(series.iloc[-1]), 6),
            series_min=round(float(series.min()), 6),
            series_max=round(float(series.max()), 6),
            series_mean=round(float(series.mean()), 6),
        )
        change = _pct_change(series)
        if change is not None:
            detail["change_first_to_last_pct"] = round(change, 3)

        slope, r2 = _trend(series)
        detail["trend_slope_per_period"] = round(slope, 6)
        detail["trend_r_squared"] = round(r2, 4)
        # A significant slope on a nearly-flat fit is noise; say so instead of
        # reporting a direction the data does not support.
        if r2 < 0.05:
            direction = "unclear"
        else:
            direction = "upward" if slope > 0 else ("downward" if slope < 0 else "flat")
        detail["trend_direction"] = direction

        # Rolling average
        if rolling_window and rolling_window >= 2:
            window = min(rolling_window, max(2, len(series) // 2))
            rolling = series.rolling(window=window, min_periods=max(2, window // 2)).mean()
            detail["rolling_mean_window"] = window
            detail["rolling_mean_last"] = _round_or_none(rolling.iloc[-1])
            detail["rolling_mean_above_series"] = bool(
                rolling.iloc[-1] == rolling.iloc[-1] and rolling.iloc[-1] > series.iloc[-1]
            )

        # Seasonality + stationarity
        seasonal_strength = None
        period = _infer_period(freq)
        if decompose and len(series) >= 2 * period:
            try:
                from statsmodels.tsa.seasonal import STL

                result = STL(series, period=period, robust=True).fit()
                var_resid = np.nanvar(result.resid)
                var_season = np.nanvar(result.seasonal)
                var_trend = np.nanvar(result.trend)
                seasonal_strength = max(0.0, 1 - var_resid / var_season) if var_season > 0 else 0.0
                trend_strength = max(0.0, 1 - var_resid / var_trend) if var_trend > 0 else 0.0
                detail["seasonal_period"] = period
                detail["seasonal_strength"] = round(float(seasonal_strength), 4)
                detail["trend_strength"] = round(float(trend_strength), 4)
                detail["seasonal_peak_period"] = _peak_period(result.seasonal, period)
                detail["detrended_noise_sd"] = _round_or_none(np.nanstd(result.resid))
            except Exception as exc:  # decomposition is best-effort
                detail["decomposition_error"] = str(exc)

        if stationarity:
            try:
                adf_stat, p_value, critical = _adfuller(series.to_numpy())
                detail["adf_statistic"] = round(float(adf_stat), 4)
                detail["adf_p_value"] = round(float(p_value), 5)
                detail["adf_conclusion"] = (
                    "stationary (reject unit root)" if p_value < 0.05
                    else "non-stationary (cannot reject a unit root) — differences or detrending needed"
                )
                detail["adf_critical_5pct"] = round(float(critical.get("5%", float("nan"))), 4)
            except Exception as exc:
                detail["stationarity_error"] = str(exc)

        if changepoints:
            breaks = _change_points(series)
            if breaks:
                detail["changepoints"] = [f"{ts.date()} ({value:.4g})" for ts, value in breaks]

        # Forecast
        table = None
        if forecast_periods and forecast_periods > 0:
            forecast_periods = min(forecast_periods, 60)
            try:
                forecast, lower, upper, model_name = _forecast(
                    series, forecast_periods, forecast_method, period
                )
                detail["forecast_method"] = model_name
                detail["forecast"] = [
                    {"period": str(index.date()) if hasattr(index, "date") else str(index),
                     "value": round(float(value), 4)}
                    for index, value in forecast.items()
                ]
                if lower is not None:
                    detail["forecast_ci95"] = [
                        {"period": str(index.date()) if hasattr(index, "date") else str(index),
                         "low": round(float(lo), 4), "high": round(float(hi), 4)}
                        for index, lo, hi in zip(forecast.index, lower, upper)
                    ]
                table = DataTable(
                    ["period", "actual", "forecast"],
                    [
                        [str(i.date()) if hasattr(i, "date") else str(i),
                         _round_or_none(series.get(i)), _round_or_none(forecast.get(i))]
                        for i in series.index
                    ][-12:]
                    + [
                        [str(i.date()) if hasattr(i, "date") else str(i), None, round(float(v), 4)]
                        for i, v in forecast.items()
                    ],
                    total_rows=len(forecast),
                )
                detail["forecast_mean"] = round(float(forecast.mean()), 4)
            except Exception as exc:
                detail["forecast_error"] = f"{type(exc).__name__}: {exc}"

        summary = (
            f"`{value_column}` of `{dataset.name}` aggregated by {agg} at {freq} frequency over "
            f"{len(series)} periods ({detail['date_range']}): went from {_fmt(series.iloc[0])} to "
            f"{_fmt(series.iloc[-1])}"
            + (f" ({change:+.1f}%)" if change is not None else "")
        )
        if direction == "unclear":
            summary += (
                f" There is no clear linear trend (slope {slope:.4g}/period but R\u00b2={r2:.3f}, so the "
                "fit explains almost nothing);"
                + (
                    " treat the first-to-last change above as describing the endpoints rather than "
                    "a steady trend."
                    if change is not None
                    else " the series mostly moves sideways."
                )
            )
        else:
            summary += f" Trend is {direction} (slope {slope:.4g}/period, R\u00b2={r2:.3f})."
        if seasonal_strength is not None:
            summary += f" Seasonality strength {seasonal_strength:.2f} on a {period}-period cycle."
        if "adf_conclusion" in detail:
            summary += f" ADF: {detail['adf_conclusion']}."
        if detail.get("changepoints"):
            summary += f" Change points: {', '.join(detail['changepoints'][:3])}."
        if table is not None:
            summary += (
                f" Forecast for the next {forecast_periods} period(s) using "
                f"{detail.get('forecast_method', forecast_method)}: "
                f"mean {detail.get('forecast_mean')}, ending at "
                f"{list(detail['forecast'])[-1]['value'] if detail.get('forecast') else 'n/a'}."
            )
        return self.result(summary, table=table, detail=detail)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):,.4g}"
    except (TypeError, ValueError):
        return str(value)


def _round_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)


def _resolve_freq(dates: pd.Series, choice: str) -> str:
    if choice != "auto":
        return choice
    ordered = dates.sort_values()
    span_days = (ordered.iloc[-1] - ordered.iloc[0]).total_seconds() / 86400.0
    if span_days <= 2:
        return "h"
    if span_days <= 120:
        return "D"
    if span_days <= 800:
        return "W"
    if span_days <= 3300:
        return "ME"
    return "YE"


def _infer_period(freq: str) -> int:
    return {"h": 24, "D": 7, "W": 52, "ME": 12, "QE": 4, "YE": 1}.get(freq, 7)


def _aggregate(frame: pd.DataFrame, time_column: str, value_column: str, freq: str, agg: str) -> pd.Series:
    series = frame.set_index(time_column)[value_column].sort_index()
    resampled = series.resample(freq)
    return getattr(resampled, agg)().dropna()


def _pct_change(series: pd.Series) -> float | None:
    if len(series) < 2:
        return None
    first, last = float(series.iloc[0]), float(series.iloc[-1])
    if first == 0:
        return None
    return (last - first) / abs(first) * 100.0


def _trend(series: pd.Series) -> tuple[float, float]:
    y = series.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 2 or np.allclose(y, y[0]):
        return 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(slope), (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _peak_period(seasonal: np.ndarray, period: int) -> str | None:
    if period < 2 or len(seasonal) < period:
        return None
    profile = np.array([np.nanmean(seasonal[i::period]) for i in range(period)])
    if np.all(np.isnan(profile)):
        return None
    position = int(np.nanargmax(profile))
    return f"position {position + 1} of {period} within the cycle"


def _change_points(series: pd.Series, max_points: int = 3) -> list[tuple[Any, float]]:
    """Binary-segmentation change points on the mean using a variance-reduction score."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    if n < 6:
        return []

    def best_split(lo: int, hi: int) -> tuple[float, int] | None:
        if hi - lo < 4:
            return None
        total_var = values[lo:hi].var() * (hi - lo)
        best_score, best_index = 0.0, -1
        for cut in range(lo + 2, hi - 1):
            left, right = values[lo:cut], values[cut:hi]
            score = total_var - (left.var() * len(left) + right.var() * len(right))
            if score > best_score:
                best_score, best_index = score, cut
        if best_index == -1:
            return None
        return best_score, best_index

    points: list[tuple[Any, float]] = []
    segments = [(0, n)]
    for _ in range(max_points):
        candidate = None
        for lo, hi in segments:
            found = best_split(lo, hi)
            if found and (candidate is None or found[0] > candidate[0]):
                candidate = found + (lo, hi)
        if candidate is None:
            break
        score, index, lo, hi = candidate
        # Require the split to explain a meaningful share of the variance.
        relative = score / (values[lo:hi].var() * (hi - lo) + 1e-12)
        if relative < 0.35:
            break
        points.append((series.index[index], float(values[index])))
        segments = [(lo, index), (index, hi)]
    return sorted(points)


def _adfuller(values: np.ndarray) -> tuple[float, float, dict]:
    """Augmented Dickey-Fuller, tolerant of the statsmodels return-type transition."""
    from statsmodels.tsa.stattools import adfuller

    try:  # statsmodels >= 0.15 exposes result_object; keep the legacy tuple shape
        result = adfuller(values, autolag="AIC", result_object=False)
    except TypeError:  # pragma: no cover - older statsmodels
        result = adfuller(values, autolag="AIC")
    return float(result[0]), float(result[1]), dict(result[4])


def _forecast(series: pd.Series, periods: int, method: str, period: int):
    """Return (forecast, lower, upper, model_name)."""
    index = pd.date_range(start=series.index[-1], periods=periods + 1, freq=series.index.freq or "D")[1:]
    lower = upper = None
    model_name = method

    if method == "naive":
        values = np.repeat(series.iloc[-1], periods)
        residual_sd = float(np.nanstd(np.diff(series.to_numpy(dtype=float)))) or 0.0
        lower = values - 1.96 * residual_sd
        upper = values + 1.96 * residual_sd
        model_name = "naive (last value repeated)"
    elif method == "linear":
        slope, intercept = np.polyfit(np.arange(len(series)), series.to_numpy(dtype=float), 1)
        future_x = np.arange(len(series), len(series) + periods)
        values = slope * future_x + intercept
        residual_sd = float(np.std(series.to_numpy(dtype=float) - (slope * np.arange(len(series)) + intercept)))
        lower = values - 1.96 * residual_sd
        upper = values + 1.96 * residual_sd
        model_name = "linear trend extrapolation"
    elif method == "arima":
        from statsmodels.tsa.arima.model import ARIMA

        order = (1, 1, 1) if len(series) >= 12 else (1, 0, 1)
        fitted = ARIMA(series.astype(float), order=order).fit()
        result = fitted.get_forecast(steps=periods)
        values = result.predicted_mean.to_numpy()
        conf = result.conf_int(alpha=0.05)
        # statsmodels returns a DataFrame here, an ndarray for other models.
        if hasattr(conf, "iloc") and getattr(conf, "ndim", 1) == 2:
            lower = conf.iloc[:, 0].to_numpy()
            upper = conf.iloc[:, 1].to_numpy()
        else:
            array = np.asarray(conf, dtype=float)
            lower, upper = array[:, 0], array[:, 1]
        model_name = f"ARIMA{order}"
    else:  # ets / holt-winters
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        seasonal = "add" if len(series) >= 2 * period else None
        seasonal_periods = period if seasonal else None
        try:
            model = ExponentialSmoothing(
                series.astype(float), trend="add", seasonal=seasonal,
                seasonal_periods=seasonal_periods, initialization_method="estimated",
            )
            fitted = model.fit()
            values = fitted.forecast(periods).to_numpy()
            residual_sd = float(np.std(fitted.resid))
            lower = values - 1.96 * residual_sd
            upper = values + 1.96 * residual_sd
            model_name = "Holt-Winters exponential smoothing" + (" (additive seasonality)" if seasonal else " (trend only)")
        except Exception:
            slope, intercept = np.polyfit(np.arange(len(series)), series.to_numpy(dtype=float), 1)
            future_x = np.arange(len(series), len(series) + periods)
            values = slope * future_x + intercept
            model_name = "fallback linear trend (ETS did not converge)"

    values = np.asarray(values, dtype=float)
    forecast = pd.Series(values, index=index[: len(values)])
    if lower is not None:
        lower = pd.Series(np.asarray(lower, dtype=float), index=forecast.index)
        upper = pd.Series(np.asarray(upper, dtype=float), index=forecast.index)
    return forecast, lower, upper, model_name
