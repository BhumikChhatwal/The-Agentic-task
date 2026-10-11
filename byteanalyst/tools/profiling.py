"""Tool 1 — profiling: what *is* this dataset?

A meta/exploratory tool. It answers "what columns exist, what do they mean, what
is broken about them, and what is worth analysing" so the agent can plan rather
than guess. Distinct from every other tool because it computes no statistic about
a relationship; it characterises the table itself.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..formatting import DataTable
from .base import Tool, ToolContext, as_int, optional, resolve_column


class ProfileTool(Tool):
    name = "profile_dataset"
    category = "understanding"
    description = (
        "Profile a table before analysing it: schema, inferred semantic role per column "
        "(numeric / datetime / categorical / identifier / text), null and cardinality "
        "stats, min/max/mean/median, top values, duplicate rows and data-quality warnings. "
        "Use this FIRST on an unfamiliar dataset, and whenever a question mentions a column "
        "or concept you have not seen yet."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "description": "Dataset (table) name. Omit if only one dataset is loaded.",
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of columns to profile.",
            },
            "max_values": {
                "type": "integer",
                "description": "How many top/sample values to list per column (default 5).",
            },
        },
        "required": [],
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        wanted = optional(kwargs, "columns")
        requested = optional(kwargs, "dataset")
        max_values = as_int(optional(kwargs, "max_values", 5), "max_values")
        max_values = max(1, min(max_values, 25))

        # Profiling every loaded table is the most useful default when the caller has
        # not chosen one yet.
        if requested:
            targets = [ctx.dataset(requested)]
        else:
            targets = ctx.registry.datasets()
        multiple = len(targets) > 1

        rows: list[list[Any]] = []
        detail: dict[str, Any] = {}
        issues: list[str] = []
        suggestions: list[str] = []
        total_rows = 0
        role_counts: dict[str, int] = {}

        for dataset in targets:
            frame = dataset.frame
            profiled = dataset.profile(refresh=True)
            if wanted:
                resolved = {resolve_column(frame, str(c)) for c in wanted}
                profiled = [row for row in profiled if row["column"] in resolved]
                if not profiled:
                    return self.fail(
                        f"none of the requested columns exist in `{dataset.name}`; "
                        f"its columns are: {', '.join(dataset.columns)}"
                    )
            prefix = f"{dataset.name}." if multiple else ""
            for row in profiled:
                stats = self._stats_text(row)
                rows.append(
                    [
                        f"{prefix}{row['column']}",
                        row["role"],
                        row["dtype"],
                        row["non_null"],
                        f"{row['null_pct']}%",
                        row["unique"],
                        stats,
                    ]
                )
                role_counts[row["role"]] = role_counts.get(row["role"], 0) + 1

            total_rows += len(frame)
            duplicates = int(frame.duplicated().sum())
            frame_issues = self._quality_issues(frame, profiled)
            if duplicates:
                frame_issues.append(
                    f"{duplicates:,} fully duplicated row(s) "
                    f"({duplicates / max(1, len(frame)):.1%} of the table)"
                )
            issues.extend(f"{prefix}{issue}" if multiple else issue for issue in frame_issues)
            suggestions.extend(self._suggestions(dataset, profiled))

            key = f"dataset_{dataset.name}" if multiple else "dataset"
            detail[key] = (
                f"{len(frame):,} rows x {len(frame.columns)} columns, "
                f"{int(frame.memory_usage(deep=True).sum() / 1024):,} KB"
            )
            if dataset.notes:
                detail[f"notes_{dataset.name}"] = dataset.notes

        table = DataTable(
            ["column", "role", "dtype", "non_null", "nulls", "unique", "summary"],
            rows,
            total_rows=len(rows),
        )
        if len(targets) == 1:
            detail["rows"] = len(targets[0].frame)
            detail["columns"] = len(targets[0].frame.columns)
        if issues:
            detail["quality_issues"] = "; ".join(issues[:12])
        if suggestions:
            detail["suggested_analyses"] = "; ".join(list(dict.fromkeys(suggestions))[:6])

        names = ", ".join(f"`{d.name}`" for d in targets)
        summary = (
            f"Profiled {names}: {total_rows:,} row(s) across {len(targets)} table(s), "
            f"{len(rows)} column(s), {len(issues)} quality warning(s). Roles: "
            + ", ".join(f"{role}={count}" for role, count in sorted(role_counts.items()))
        )
        return self.result(summary, table=table, detail=detail)

    @staticmethod
    def _stats_text(row: dict[str, Any]) -> str:
        if row.get("mean") is not None:
            return (
                f"min {row['min']} / median {row['median']} / mean {row['mean']} / "
                f"max {row['max']} (sd {row['std']})"
            )
        if row.get("min") is not None:
            return f"{row['min']} .. {row['max']}"
        if row.get("top_values"):
            return ", ".join(row["top_values"])
        if row.get("sample_values"):
            return ", ".join(row["sample_values"])
        return ""

    # -- internals --------------------------------------------------------
    @staticmethod
    def _quality_issues(frame: pd.DataFrame, profiled: list[dict[str, Any]]) -> list[str]:
        issues: list[str] = []
        for row in profiled:
            name = row["column"]
            if row["role"] == "empty":
                issues.append(f"`{name}` is entirely null")
            elif row["null_pct"] > 40:
                issues.append(f"`{name}` has {row['null_pct']}% missing values")
            elif row["role"] == "constant":
                value = (row.get("top_values") or ["?"])[0]
                issues.append(f"`{name}` is constant ({value}) — zero information")
            elif row["role"] == "identifier":
                issues.append(f"`{name}` looks like a unique identifier (not a measure)")
            elif row["role"] == "text" and len(frame) > 20:
                issues.append(f"`{name}` is free text (high cardinality)")
        return issues

    @staticmethod
    def _suggestions(dataset: Any, profiled: list[dict[str, Any]]) -> list[str]:
        roles = {row["column"]: row["role"] for row in profiled}
        numeric = [c for c, r in roles.items() if r == "numeric"]
        dates = [c for c, r in roles.items() if r == "datetime"]
        cats = [c for c, r in roles.items() if r in {"categorical", "boolean"}]
        out: list[str] = []
        if numeric and cats:
            out.append(
                f"compare `{numeric[0]}` across `{cats[0]}` groups (run_sql + statistical_test)"
            )
        if dates and numeric:
            out.append(f"trend of `{numeric[0]}` over `{dates[0]}` (analyze_time_series)")
        if len(numeric) >= 2:
            out.append(
                f"correlation between `{numeric[0]}` and `{numeric[1]}` (statistical_test pearson/spearman)"
            )
        if numeric:
            out.append(f"outliers in `{numeric[0]}` (detect_anomalies)")
        if cats and numeric:
            out.append(f"distribution of `{numeric[0]}` by `{cats[0]}` (visualize)")
        return out[:5]
