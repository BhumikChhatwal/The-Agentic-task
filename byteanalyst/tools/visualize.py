"""Tool 6 — visualisation.

Distinct because its output is a *picture*: distribution shape, a trend line, a
correlation matrix, a group comparison at a glance. Numbers can be misread;
a histogram of a bimodal column cannot. Every chart is also accompanied by a
short numeric summary of exactly what was plotted, so the agent's prose is still
grounded in the same numbers that went into the figure.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..errors import ToolError  # noqa: E402
from ..formatting import DataTable  # noqa: E402
from .base import (  # noqa: E402
    Tool,
    ToolContext,
    as_int,
    optional,
    require,
    require_numeric_column,
    resolve_column,
)

KINDS = (
    "histogram", "box", "violin", "bar", "line", "area", "scatter",
    "heatmap", "pie", "time_series",
)


class VisualizeTool(Tool):
    name = "visualize"
    category = "presentation"
    description = (
        "Render a chart to a PNG and get back a numeric summary of what was plotted. "
        f"Kinds: {', '.join(KINDS)}. Use it to show distributions (histogram, box, violin), "
        "comparisons between groups (bar), relationships (scatter, heatmap for a correlation "
        "matrix), and movement over time (line, area, time_series). The summary it returns is "
        "the evidence for anything you claim about the picture, so read it before describing "
        "the shape. Charts are shown to the user, not to you — rely on the summary."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(KINDS), "description": "Chart type."},
            "dataset": {"type": "string", "description": "Dataset name (optional if only one)."},
            "x": {"type": "string", "description": "X axis / category column (or the numeric column for histogram)."},
            "y": {"type": "string", "description": "Y axis / value column (numeric) where applicable."},
            "hue": {
                "type": "string",
                "description": "Optional categorical column to split or colour by (scatter/area/bar stacks or lines).",
            },
            "agg": {
                "type": "string",
                "enum": ["sum", "mean", "median", "count", "max", "min"],
                "description": "Aggregation for bar/area/time_series (default sum).",
            },
            "bins": {"type": "integer", "description": "Histogram bins (default 30)."},
            "top_k": {
                "type": "integer",
                "description": "For bar/pie: keep only the top K categories by value (default 15).",
            },
            "title": {"type": "string", "description": "Chart title."},
            "log_scale": {"type": "boolean", "description": "Use a log scale on the value axis."},
        },
        "required": ["kind"],
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        kind = str(require(kwargs, "kind")).strip().lower()
        if kind not in KINDS:
            raise ToolError(f"unknown chart kind {kind!r}; choose one of: {', '.join(KINDS)}")
        dataset = ctx.dataset(optional(kwargs, "dataset"))
        frame = dataset.frame
        if frame.empty:
            raise ToolError("dataset is empty; nothing to plot")

        agg = str(optional(kwargs, "agg", "sum"))
        top_k = max(1, min(as_int(optional(kwargs, "top_k", 15), "top_k"), 50))
        bins = max(2, min(as_int(optional(kwargs, "bins", 30), "bins"), 200))
        title = optional(kwargs, "title")
        log_scale = bool(optional(kwargs, "log_scale", False))

        fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
        detail: dict[str, Any] = {"dataset": dataset.name, "kind": kind}
        table = None
        try:
            table, plotted = self._render(
                ax, kind, frame, dataset, kwargs, agg, top_k, bins, log_scale, detail
            )
            ax.set_title(title or plotted or f"{kind} — {dataset.name}", fontsize=11)
            if kind not in {"pie"}:
                ax.grid(True, alpha=0.25, linestyle=":")
            fig.tight_layout()
            path = self._save(fig, ctx, dataset.name, kind)
        finally:
            plt.close(fig)

        detail["chart_path"] = path
        summary = (
            f"Rendered a {kind} of `{dataset.name}`"
            + (f" ({table.total_rows:,} plotted point(s))" if table and len(table) else "")
            + f". Figure saved as `{Path(path).name}`. "
            + (detail.get("plot_note", "") or "")
        ).strip()
        return self.result(summary, table=table, detail=detail, artifacts=[path])

    # -- rendering --------------------------------------------------------
    def _render(self, ax, kind, frame, dataset, kwargs, agg, top_k, bins, log_scale, detail):
        table = None
        note = ""

        if kind == "histogram":
            column = require_numeric_column(frame, str(require(kwargs, "x")))
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                raise ToolError(f"no numeric values in {column!r} to plot")
            hue = optional(kwargs, "hue")
            if hue:
                hue_col = resolve_column(frame, str(hue))
                for label, group in frame.groupby(hue_col, dropna=True):
                    group_values = pd.to_numeric(group[column], errors="coerce").dropna()
                    if len(group_values):
                        ax.hist(group_values, bins=bins, alpha=0.55, label=str(label))
                ax.legend(fontsize=8)
            else:
                ax.hist(values, bins=bins, color="#3b6ea5", edgecolor="white", linewidth=0.4)
            ax.set_xlabel(column)
            ax.set_ylabel("count")
            q = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            detail.update(
                n=int(values.size), mean=round(float(values.mean()), 6),
                median=round(float(values.median()), 6), std=round(float(values.std(ddof=1)), 6),
                skew=round(float(values.skew()), 4),
                p05=round(float(q.loc[0.05]), 6), p25=round(float(q.loc[0.25]), 6),
                p75=round(float(q.loc[0.75]), 6), p95=round(float(q.loc[0.95]), 6),
            )
            note = (
                f"mean {values.mean():.4g}, median {values.median():.4g}, "
                f"sd {values.std(ddof=1):.4g}, skew {values.skew():.2f}."
            )

        elif kind in {"box", "violin"}:
            # Convention: `y` is the numeric value and `x` the optional grouping column.
            # If only `x` is given it is treated as the value column.
            if kwargs.get("y"):
                value_column = require_numeric_column(frame, str(kwargs["y"]))
                group_spec = kwargs.get("x") or optional(kwargs, "hue")
            else:
                value_column = require_numeric_column(frame, str(require(kwargs, "x")))
                group_spec = optional(kwargs, "group") or optional(kwargs, "hue")
            if group_spec is None:
                group_spec = _default_group_column(dataset)
            data, labels = [], []
            if group_spec:
                group_col = resolve_column(frame, str(group_spec))
                counts = frame[group_col].value_counts().head(top_k)
                for label in counts.index:
                    values = pd.to_numeric(
                        frame.loc[frame[group_col] == label, value_column], errors="coerce"
                    ).dropna()
                    if len(values) >= 2:
                        data.append(values.to_numpy())
                        labels.append(str(label))
            else:
                values = pd.to_numeric(frame[value_column], errors="coerce").dropna()
                data, labels = [values.to_numpy()], [value_column]
            if not data:
                raise ToolError("no groups with enough observations to plot")
            if kind == "box":
                ax.boxplot(data, showmeans=True)
            else:
                ax.violinplot(data, showmeans=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            if len(labels) > 1:
                ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.set_ylabel(value_column)
            group_column = group_spec
            table = DataTable(
                [str(group_column) if group_column else "group", "n", "mean", "median", "sd"],
                [
                    [label, len(values), round(float(np.mean(values)), 6),
                     round(float(np.median(values)), 6), round(float(np.std(values, ddof=1)), 6)]
                    for label, values in zip(labels, data)
                ],
            )
            note = f"group means: {', '.join(f'{l}={np.mean(v):.4g}' for l, v in zip(labels, data))}."

        elif kind in {"bar", "pie"}:
            x_column = resolve_column(frame, str(require(kwargs, "x")))
            if kwargs.get("y"):
                y_column = resolve_column(frame, str(kwargs["y"]))
                if not pd.api.types.is_numeric_dtype(frame[y_column]):
                    y_column = require_numeric_column(frame, str(kwargs["y"]))
                grouped = (
                    frame.groupby(x_column, dropna=True)[y_column]
                    .agg(agg if agg in {"sum", "mean", "median", "count", "max", "min"} else "sum")
                    .sort_values(ascending=False)
                    .head(top_k)
                )
                value_label = f"{agg} of {y_column}"
            else:
                grouped = frame[x_column].value_counts().head(top_k)
                value_label = "count"
            if grouped.empty:
                raise ToolError(f"no values to plot for {x_column!r}")
            labels = [str(i) for i in grouped.index]
            if kind == "bar":
                ax.bar(labels, grouped.to_numpy(), color="#4c78a8")
                ax.tick_params(axis="x", rotation=30, labelsize=8)
                ax.set_ylabel(value_label)
                if log_scale:
                    ax.set_yscale("log")
            else:
                ax.pie(grouped.to_numpy(), labels=labels, autopct="%1.1f%%", textprops={"fontsize": 8})
            total = float(grouped.sum())
            table = DataTable(
                [x_column, value_label, "share"],
                [
                    [label, round(float(value), 6), f"{value / total:.1%}" if total else "0%"]
                    for label, value in grouped.items()
                ],
                total_rows=len(grouped),
            )
            note = f"top category: {labels[0]} = {grouped.iloc[0]:.4g} ({grouped.iloc[0] / total:.1%} of shown total)."

        elif kind in {"line", "area", "time_series"}:
            x_column = resolve_column(frame, str(require(kwargs, "x")))
            y_column = require_numeric_column(frame, str(require(kwargs, "y")))
            working = frame[[x_column, y_column]].dropna().copy()
            values = pd.to_numeric(working[y_column], errors="coerce")
            working = working.assign(**{y_column: values}).dropna()

            if pd.api.types.is_datetime64_any_dtype(frame[x_column]) or kind == "time_series":
                working[x_column] = pd.to_datetime(working[x_column], errors="coerce")
                working = working.dropna(subset=[x_column]).sort_values(x_column)
                if working.empty:
                    raise ToolError(f"{x_column!r} has no parseable dates")
                freq = _infer_freq(working[x_column])
                series = (
                    working.set_index(x_column)[y_column].resample(freq).mean()
                    if agg == "mean" else working.set_index(x_column)[y_column].resample(freq).sum()
                )
                ax.plot(series.index, series.to_numpy(), marker="o", markersize=3, color="#2f6f4f")
                ax.set_xlabel(x_column)
                ax.set_ylabel(f"{agg} of {y_column}")
                fig = ax.get_figure()
                fig.autofmt_xdate()
                detail.update(
                    frequency=freq, points=int(series.size),
                    first=str(series.index[0].date()), last=str(series.index[-1].date()),
                    first_value=round(float(series.iloc[0]), 6),
                    last_value=round(float(series.iloc[-1]), 6),
                    change_pct=round(
                        float((series.iloc[-1] - series.iloc[0]) / abs(series.iloc[0]) * 100), 3
                    ) if series.iloc[0] else None,
                    series_min=round(float(series.min()), 6),
                    series_max=round(float(series.max()), 6),
                )
                table = DataTable(
                    [x_column, y_column],
                    [[str(i.date()) if hasattr(i, "date") else str(i), round(float(v), 6)] for i, v in series.items()],
                    total_rows=series.size,
                    truncated=True,
                )
                note = (
                    f"from {series.index[0].date()} ({series.iloc[0]:.4g}) to "
                    f"{series.index[-1].date()} ({series.iloc[-1]:.4g})."
                )
            else:
                hue = optional(kwargs, "hue")
                if hue:
                    hue_col = resolve_column(frame, str(hue))
                    for label, group in working.groupby(hue_col, dropna=True):
                        ordered = group.sort_values(x_column)
                        ax.plot(ordered[x_column], ordered[y_column], label=str(label), linewidth=1.4)
                    ax.legend(fontsize=8)
                else:
                    ordered = working.sort_values(x_column)
                    ax.plot(ordered[x_column], ordered[y_column], color="#4c78a8", linewidth=1.4)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
                detail.update(points=int(len(working)))
                note = f"{len(working):,} points plotted."

        elif kind == "scatter":
            x_column = require_numeric_column(frame, str(require(kwargs, "x")))
            y_column = require_numeric_column(frame, str(require(kwargs, "y")))
            working = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
            if working.empty:
                raise ToolError("no complete numeric pairs to plot")
            hue = optional(kwargs, "hue")
            if hue:
                hue_col = resolve_column(frame, str(hue))
                for label, group in frame.loc[working.index].groupby(hue_col, dropna=True):
                    ax.scatter(group[x_column], group[y_column], s=14, alpha=0.65, label=str(label))
                ax.legend(fontsize=8)
            else:
                ax.scatter(working[x_column], working[y_column], s=14, alpha=0.6, color="#4c78a8")
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            if log_scale:
                ax.set_yscale("log")
            correlation = working[x_column].corr(working[y_column])
            detail.update(
                points=int(len(working)),
                pearson_r=round(float(correlation), 4) if correlation == correlation else None,
                r_squared=round(float(correlation) ** 2, 4) if correlation == correlation else None,
            )
            note = f"Pearson r = {correlation:.3f} (r²={correlation ** 2:.3f})."

        elif kind == "heatmap":
            if kwargs.get("y") and kwargs.get("hue"):
                row_col = resolve_column(frame, str(kwargs["x"])) if kwargs.get("x") else None
                col_col = resolve_column(frame, str(kwargs["hue"]))
                value_col = require_numeric_column(frame, str(kwargs["y"]))
                if row_col is None:
                    raise ToolError("heatmap needs `x` (rows), `hue` (columns) and `y` (values)")
                pivot = frame.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc=agg)
                image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlBu_r")
                ax.set_xticks(range(pivot.shape[1]))
                ax.set_xticklabels([str(c) for c in pivot.columns], rotation=30, ha="right", fontsize=8)
                ax.set_yticks(range(pivot.shape[0]))
                ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
                ax.get_figure().colorbar(image, ax=ax)
                detail.update(rows=int(pivot.shape[0]), columns=int(pivot.shape[1]))
                note = f"pivot of {value_col} by {row_col} x {col_col}."
            else:
                numeric = [c for c in dataset.numeric_columns()][:12]
                if len(numeric) < 2:
                    raise ToolError("heatmap needs at least two numeric columns (or pass x, y, hue)")
                correlation = frame[numeric].apply(pd.to_numeric, errors="coerce").corr()
                image = ax.imshow(correlation.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(numeric)))
                ax.set_xticklabels(numeric, rotation=40, ha="right", fontsize=8)
                ax.set_yticks(range(len(numeric)))
                ax.set_yticklabels(numeric, fontsize=8)
                ax.get_figure().colorbar(image, ax=ax)
                pairs = [
                    (a, b, correlation.loc[a, b])
                    for i, a in enumerate(numeric) for b in numeric[i + 1:]
                    if correlation.loc[a, b] == correlation.loc[a, b]
                ]
                pairs.sort(key=lambda t: abs(t[2]), reverse=True)
                strongest = pairs[0] if pairs else None
                table = DataTable(
                    ["column_a", "column_b", "pearson_r"],
                    [[a, b, round(float(r), 4)] for a, b, r in pairs[:15]],
                    total_rows=len(pairs),
                )
                detail.update(numeric_columns=numeric, pairs_compared=len(pairs))
                if strongest:
                    detail.update(
                        strongest_pair=f"{strongest[0]} vs {strongest[1]}",
                        strongest_r=round(float(strongest[2]), 4),
                    )
                    note = f"strongest correlation: {strongest[0]} vs {strongest[1]} r={strongest[2]:.3f}."
        else:  # pragma: no cover - enum is validated above
            raise ToolError(f"unsupported chart kind {kind!r}")

        if note:
            detail["plot_note"] = note
        return table, note

    # -- saving -----------------------------------------------------------
    @staticmethod
    def _save(fig, ctx: ToolContext, dataset_name: str, kind: str) -> str:
        ctx.config.ensure_dirs()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{dataset_name}-{kind}-{stamp}-{uuid.uuid4().hex[:6]}.png"
        path = Path(ctx.config.artifact_dir) / name
        fig.savefig(path, bbox_inches="tight")
        return str(path)


def _default_group_column(dataset) -> str | None:
    """Pick a small-cardinality categorical column to group a distribution by."""
    for column in dataset.categorical_columns():
        values = dataset.frame[column].dropna()
        if 2 <= values.nunique() <= 12:
            return column
    return None


def _infer_freq(index: pd.Series) -> str:
    """Pick a sensible resampling frequency for an irregular datetime index."""
    unique = pd.Series(pd.to_datetime(index)).dropna().sort_values().drop_duplicates()
    if len(unique) < 2:
        return "D"
    span_days = (unique.iloc[-1] - unique.iloc[0]).total_seconds() / 86400.0
    if span_days <= 2:
        return "h"
    if span_days <= 120:
        return "D"
    if span_days <= 800:
        return "W"
    if span_days <= 3000:
        return "ME"
    return "YE"
