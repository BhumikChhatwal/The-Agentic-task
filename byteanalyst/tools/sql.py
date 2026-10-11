"""Tool 2 — relational querying with SQL (DuckDB).

Distinct from the other tools because it operates on the *relations*: grouping,
joining several tables, window functions, CTEs. It is the right tool whenever the
question needs aggregation across tables or a well-defined tabular result.

DuckDB is used over an in-memory database with every dataset registered as a
view, so a query can join `orders` to `customers` with no ETL step.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ..errors import ToolError
from ..formatting import DataTable
from .base import Tool, ToolContext, as_int, optional

# Statement shapes we accept; everything else is refused before DuckDB sees it.
_ALLOWED_START = re.compile(r"^\s*(with|select|describe|summarize|from)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|replace|attach|detach|copy|export|"
    r"import|install|load|pragma|call|set|grant|revoke|vacuum|begin|commit|rollback)\b",
    re.IGNORECASE,
)


class SqlTool(Tool):
    name = "run_sql"
    category = "querying"
    description = (
        "Run a read-only SQL query (DuckDB dialect, PostgreSQL-flavoured) over the loaded "
        "datasets. Every dataset is a table named after it. Use this for filtering, grouping, "
        "aggregation, ranking, window functions and joins between tables, and whenever a "
        "question needs a precise tabular answer. Call profile_dataset first if you are unsure "
        "of column names. Prefer explicit column lists over SELECT *."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A single read-only SQL SELECT (or WITH ... SELECT) statement.",
            },
            "max_rows": {
                "type": "integer",
                "description": "Maximum rows to return to you (default 50, hard cap 500).",
            },
        },
        "required": ["query"],
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        query = str(kwargs.get("query") or "").strip().rstrip(";")
        if not query:
            raise ToolError("query must not be empty")
        self._validate(query)

        max_rows = as_int(optional(kwargs, "max_rows", ctx.config.max_preview_rows), "max_rows")
        max_rows = max(1, min(max_rows, 500))

        if len(ctx.registry) == 0:
            return self.fail("no datasets are loaded, so there is nothing to query")

        con = ctx.registry.duckdb_connection()
        try:
            rel = con.execute(query)
            frame: pd.DataFrame = rel.fetchdf()
        except Exception as exc:
            tables = ", ".join(
                f"{d.sql_name}({', '.join(d.columns)})" for d in ctx.registry.datasets()
            )
            return self.fail(
                f"SQL error: {type(exc).__name__}: {exc}\n"
                f"Available tables and columns: {tables}"
            )
        finally:
            pass

        total = len(frame)
        shown = frame.head(max_rows)
        table = DataTable(
            list(shown.columns),
            shown.itertuples(index=False, name=None),
            total_rows=total,
            truncated=total > max_rows,
        )
        if total == 0:
            summary = "Query returned 0 rows. The filter may be too strict, or the value may be spelled differently."
        elif total == 1 and len(frame.columns) == 1:
            value = frame.iloc[0, 0]
            summary = f"Query returned a single value: {frame.columns[0]} = {value}"
        else:
            summary = f"Query returned {total:,} row(s) and {len(frame.columns)} column(s)."

        detail: dict[str, Any] = {"total_rows": total, "query": query}
        if len(shown.columns) == 1 and total:
            column = shown.columns[0]
            values = pd.to_numeric(shown[column], errors="coerce").dropna()
            if len(values):
                detail.update(
                    columns_returned=1,
                    min=float(values.min()),
                    max=float(values.max()),
                    sum=float(values.sum()),
                )
        return self.result(summary, table=table, detail=detail)

    @staticmethod
    def _validate(query: str) -> None:
        if ";" in query:
            raise ToolError("only a single SQL statement is allowed (no ';' inside the query)")
        if not _ALLOWED_START.match(query):
            raise ToolError(
                "only read-only queries are allowed; start with SELECT or WITH ... SELECT"
            )
        match = _FORBIDDEN.search(query)
        if match:
            raise ToolError(
                f"read-only tool: the keyword {match.group(0).upper()!r} is not allowed"
            )
