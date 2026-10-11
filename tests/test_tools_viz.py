"""The visualisation tool: real figures on disk plus a grounded numeric summary."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import run_tool

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _assert_png(path: str) -> None:
    target = Path(path)
    assert target.exists(), path
    assert target.stat().st_size > 1000
    assert target.read_bytes()[:8] == PNG_HEADER


@pytest.mark.parametrize(
    "arguments",
    [
        {"kind": "histogram", "x": "revenue"},
        {"kind": "box", "x": "region", "y": "revenue"},
        {"kind": "violin", "x": "region", "y": "revenue"},
        {"kind": "bar", "x": "region", "y": "revenue"},
        {"kind": "pie", "x": "region"},
        {"kind": "line", "x": "order_date", "y": "revenue"},
        {"kind": "time_series", "x": "order_date", "y": "revenue", "agg": "mean"},
        {"kind": "scatter", "x": "quantity", "y": "revenue"},
        {"kind": "heatmap"},
        {"kind": "area", "x": "order_date", "y": "revenue"},
    ],
)
def test_every_chart_kind_renders(tools, ctx, arguments) -> None:
    result = run_tool(tools, ctx, "visualize", dataset="orders", **arguments)
    assert result.ok, result.error
    _assert_png(result.detail["chart_path"])
    assert result.artifacts == [result.detail["chart_path"]]
    assert ctx.artifacts  # registered for the session


def test_histogram_summary_carries_the_statistics(tools, ctx) -> None:
    result = run_tool(tools, ctx, "visualize", kind="histogram", dataset="orders", x="revenue", bins=15)
    assert result.ok
    for key in ("mean", "median", "std", "skew", "p25", "p75"):
        assert key in result.detail
    assert result.detail["mean"] == pytest.approx(ctx.frame("orders")["revenue"].mean(), rel=1e-6)


def test_bar_chart_lists_categories_with_values(tools, ctx) -> None:
    result = run_tool(tools, ctx, "visualize", kind="bar", dataset="orders", x="region", y="revenue")
    assert result.ok
    assert set(result.table.columns) == {"region", "sum of revenue", "share"}
    assert result.table.total_rows == 3
    assert "top category" in result.detail["plot_note"]


def test_bar_respects_top_k(tools, ctx) -> None:
    result = run_tool(tools, ctx, "visualize", kind="bar", dataset="orders", x="customer_id", y="revenue", top_k=5)
    assert result.ok
    assert result.table.total_rows == 5


def test_heatmap_reports_the_strongest_correlation(tools, ctx) -> None:
    result = run_tool(tools, ctx, "visualize", kind="heatmap", dataset="orders")
    assert result.ok
    assert "strongest_pair" in result.detail
    assert -1 <= result.detail["strongest_r"] <= 1
    assert result.table.columns == ["column_a", "column_b", "pearson_r"]


def test_heatmap_with_three_arguments_pivots(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "visualize", kind="heatmap", dataset="orders",
        x="region", hue="channel", y="revenue", agg="sum",
    )
    assert result.ok, result.error
    _assert_png(result.detail["chart_path"])
    assert result.detail["rows"] == 3


def test_scatter_reports_correlation(tools, ctx) -> None:
    result = run_tool(tools, ctx, "visualize", kind="scatter", dataset="orders", x="quantity", y="revenue")
    assert result.ok
    assert "pearson_r" in result.detail
    assert "Pearson r" in result.detail["plot_note"]


def test_time_series_chart_reports_endpoints(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "visualize", kind="time_series", dataset="orders",
        x="order_date", y="revenue", agg="sum",
    )
    assert result.ok
    assert result.detail["points"] > 0
    assert result.detail["first"] < result.detail["last"]
    assert "change_pct" in result.detail


def test_box_plot_auto_picks_a_grouping_column(tools, ctx) -> None:
    result = run_tool(tools, ctx, "visualize", kind="box", dataset="orders", x="revenue")
    assert result.ok
    assert result.table.total_rows >= 2  # grouped by a categorical column automatically


def test_visualize_input_errors(tools, ctx) -> None:
    unknown = run_tool(tools, ctx, "visualize", kind="sankey", dataset="orders", x="region")
    assert not unknown.ok and "unknown chart kind" in unknown.error

    wrong_type = run_tool(tools, ctx, "visualize", kind="histogram", dataset="orders", x="region")
    assert not wrong_type.ok and "not numeric" in wrong_type.error

    missing = run_tool(tools, ctx, "visualize", kind="bar", dataset="orders")
    assert not missing.ok and "x" in missing.error


def test_artifact_directory_is_used(tools, ctx) -> None:
    result = run_tool(tools, ctx, "visualize", kind="histogram", dataset="orders", x="quantity")
    assert Path(result.detail["chart_path"]).parent == ctx.config.artifact_dir
