"""Tools that infer: statistical_test, detect_anomalies, analyze_time_series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import run_tool


# ---------------------------------------------------------------------------
# statistical_test
# ---------------------------------------------------------------------------


def test_ttest_detects_a_real_group_difference(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "statistical_test",
        test="ttest_ind", dataset="orders", value_column="revenue",
        group_column="region", group_a="North", group_b="South",
    )
    # North carries a planted +60 shift, so this must be significant.
    assert result.ok, result.error
    assert result.detail["p_value"] < 0.05
    assert result.detail["cohens_d"] > 0.3
    assert "differs significantly" in result.summary
    assert result.table.columns[:2] == ["group", "n"]


def test_ttest_respects_alpha_and_can_conclude_no_difference(tools, ctx) -> None:
    # A tiny alpha makes almost nothing significant, so the wording must switch.
    result = run_tool(
        tools, ctx, "statistical_test",
        test="ttest_ind", dataset="orders", value_column="quantity",
        group_column="region", group_a="East", group_b="South", alpha=1e-9,
    )
    assert result.ok
    assert result.detail["p_value"] > 1e-9
    assert "no significant" in result.summary


def test_mannwhitney_and_anova_and_kruskal(tools, ctx) -> None:
    mw = run_tool(
        tools, ctx, "statistical_test", test="mannwhitneyu", dataset="orders",
        value_column="revenue", group_column="region", group_a="North", group_b="South",
    )
    assert mw.ok and mw.detail["rank_biserial_r"] is not None

    anova = run_tool(
        tools, ctx, "statistical_test", test="anova", dataset="orders",
        value_column="revenue", group_column="region",
    )
    assert anova.ok and 0 <= anova.detail["eta_squared"] <= 1
    assert len(anova.table.rows) == 3

    kruskal = run_tool(
        tools, ctx, "statistical_test", test="kruskal", dataset="orders",
        value_column="revenue", group_column="region",
    )
    assert kruskal.ok and kruskal.detail["epsilon_squared"] is not None


def test_correlations_report_strength_and_r_squared(tools, ctx) -> None:
    effect_keys = {"pearson": "pearson_r", "spearman": "spearman_rho", "kendall": "kendall_tau"}
    for name, key in effect_keys.items():
        result = run_tool(
            tools, ctx, "statistical_test", test=name, dataset="orders",
            columns=["quantity", "revenue"],
        )
        assert result.ok, result.error
        assert "r_squared" in result.detail
        assert -1.0001 <= result.detail[key] <= 1.0001
        assert result.detail["interpretation_strength"]


def test_chi_square_on_two_categoricals(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "statistical_test", test="chi_square", dataset="orders",
        columns=["region", "channel"],
    )
    assert result.ok
    assert result.detail["cramers_v"] >= 0
    assert result.table.total_rows == 3


def test_normality_and_one_sample(tools, ctx) -> None:
    normality = run_tool(
        tools, ctx, "statistical_test", test="normality", dataset="orders", value_column="revenue"
    )
    assert normality.ok and "Shapiro" in normality.summary

    one = run_tool(
        tools, ctx, "statistical_test", test="ttest_1samp", dataset="orders",
        value_column="revenue", popmean=10000,
    )
    assert one.ok
    assert one.detail["p_value"] < 0.05  # nowhere near 10,000


def test_proportion_test_needs_binary_outcome(tools, ctx) -> None:
    ok = run_tool(
        tools, ctx, "statistical_test", test="proportion", dataset="orders",
        value_column="status", group_column="region", group_a="North", group_b="South",
    )
    assert ok.ok, ok.error
    assert 0 <= ok.detail["rate_a"] <= 1

    bad = run_tool(
        tools, ctx, "statistical_test", test="proportion", dataset="orders",
        value_column="quantity", group_column="region",
    )
    assert not bad.ok
    assert "binary outcome" in bad.error


def test_paired_test_requires_two_columns(tools, ctx) -> None:
    result = run_tool(tools, ctx, "statistical_test", test="ttest_rel", dataset="orders", columns=["revenue"])
    assert not result.ok
    assert "two columns" in result.error


def test_unknown_test_and_non_numeric_column(tools, ctx) -> None:
    unknown = run_tool(tools, ctx, "statistical_test", test="magic")
    assert not unknown.ok and "unknown test" in unknown.error

    wrong_type = run_tool(
        tools, ctx, "statistical_test", test="ttest_ind", dataset="orders",
        value_column="region", group_column="channel",
    )
    assert not wrong_type.ok
    assert "not numeric" in wrong_type.error


def test_group_test_missing_group_column(tools, ctx) -> None:
    result = run_tool(tools, ctx, "statistical_test", test="ttest_ind", dataset="orders", value_column="revenue")
    assert not result.ok
    assert "group_column" in result.error


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------


def test_iqr_finds_the_planted_outlier(tools, ctx) -> None:
    result = run_tool(tools, ctx, "detect_anomalies", method="iqr", dataset="orders", column="revenue")
    assert result.ok, result.error
    assert result.detail["anomalies_found"] >= 1
    flagged_rows = [int(row[0]) for row in result.table.rows]
    assert 100 in flagged_rows  # the seeded spike


def test_zscore_and_mad_agree_on_the_spike(tools, ctx) -> None:
    for method in ("zscore", "mad"):
        result = run_tool(tools, ctx, "detect_anomalies", method=method, dataset="orders", column="revenue")
        assert result.ok
        assert result.detail["anomalies_found"] >= 1


def test_grouped_detection_uses_within_group_fences(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "detect_anomalies", method="iqr", dataset="orders",
        column="revenue", group_by="region",
    )
    assert result.ok
    assert result.detail["group_by"] == "region"


def test_isolation_forest_is_multivariate(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "detect_anomalies", method="isolation_forest", dataset="orders",
        columns=["revenue", "quantity"], contamination=0.05, top_k=5,
    )
    assert result.ok, result.error
    assert result.detail["anomalies_found"] >= 1
    assert "isolation_forest" in result.summary


def test_seasonal_residual_and_its_hint(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "detect_anomalies", method="seasonal_residual", dataset="orders",
        column="revenue", time_column="order_date", group_by="region", top_k=5,
    )
    assert result.ok, result.error
    assert result.detail["period"] == 7

    # The univariate methods should point at the seasonal one when a time column exists.
    iqr = run_tool(tools, ctx, "detect_anomalies", method="iqr", dataset="orders", column="revenue")
    assert "seasonal_residual" in iqr.summary


def test_anomaly_errors(tools, ctx) -> None:
    unknown = run_tool(tools, ctx, "detect_anomalies", method="vibes", dataset="orders", column="revenue")
    assert not unknown.ok and "unknown method" in unknown.error

    missing_time = run_tool(
        tools, ctx, "detect_anomalies", method="seasonal_residual", dataset="orders", column="revenue"
    )
    assert not missing_time.ok and "time_column" in missing_time.error

    too_few = run_tool(
        tools, ctx, "detect_anomalies", method="seasonal_residual", dataset="customers",
        column="customer_id", time_column="customer_id",
    )
    assert not too_few.ok


# ---------------------------------------------------------------------------
# analyze_time_series
# ---------------------------------------------------------------------------


def test_trend_reports_slope_direction_and_forecast(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "analyze_time_series", dataset="orders",
        time_column="order_date", value_column="revenue",
        forecast_periods=14, rolling_window=7,
    )
    assert result.ok, result.error
    assert result.detail["trend_slope_per_period"] > 0
    assert result.detail["trend_direction"] in {"upward", "downward", "flat", "unclear"}
    assert len(result.detail["forecast"]) == 14
    assert result.detail["forecast_method"]
    assert "adf_conclusion" in result.detail
    assert result.detail["rolling_mean_window"] == 7


def test_time_series_group_by_compares_series(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "analyze_time_series", dataset="orders",
        time_column="order_date", value_column="revenue", group_by="region",
    )
    assert result.ok
    assert result.detail["series_compared"] == 3
    assert len(result.table.rows) == 3
    assert set(result.table.columns) >= {"series", "slope_per_period", "r_squared"}


def test_time_series_forecast_methods(tools, ctx) -> None:
    for method in ("naive", "linear", "ets", "arima"):
        result = run_tool(
            tools, ctx, "analyze_time_series", dataset="orders",
            time_column="order_date", value_column="revenue",
            forecast_periods=5, forecast_method=method, decompose=False, stationarity=False,
        )
        assert result.ok, f"{method}: {result.error}"
        assert len(result.detail["forecast"]) == 5


def test_time_series_changepoints_detected(tools, ctx) -> None:
    frame = pd.DataFrame(
        {
            "d": pd.date_range("2024-01-01", periods=120, freq="D"),
            "v": list(range(60)) + list(range(200, 260)),
        }
    )
    ctx.registry.add("shift", frame, "fixture")
    result = run_tool(
        tools, ctx, "analyze_time_series", dataset="shift",
        time_column="d", value_column="v",
    )
    assert result.ok
    assert result.detail.get("changepoints")


def test_time_series_input_validation(tools, ctx) -> None:
    bad_agg = run_tool(
        tools, ctx, "analyze_time_series", dataset="orders",
        time_column="order_date", value_column="revenue", agg="medianish",
    )
    assert not bad_agg.ok and "aggregation" in bad_agg.error

    bad_method = run_tool(
        tools, ctx, "analyze_time_series", dataset="orders",
        time_column="order_date", value_column="revenue", forecast_method="crystal-ball",
    )
    assert not bad_method.ok and "forecast_method" in bad_method.error

    non_date = run_tool(
        tools, ctx, "analyze_time_series", dataset="orders",
        time_column="order_id", value_column="revenue",
    )
    # order_id parses as no dates at all, so the tool must refuse rather than guess.
    assert not non_date.ok
