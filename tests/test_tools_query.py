"""Tools that read and reshape data: profile_dataset, run_sql, run_python."""

from __future__ import annotations

from conftest import run_tool
from byteanalyst.formatting import DataTable


# ---------------------------------------------------------------------------
# profile_dataset
# ---------------------------------------------------------------------------


def test_profile_all_datasets_when_none_named(tools, ctx) -> None:
    result = run_tool(tools, ctx, "profile_dataset")
    assert result.ok
    assert "orders" in result.summary and "customers" in result.summary
    columns = [row[0] for row in result.table.rows]
    assert any(name.startswith("orders.") for name in columns)
    assert any(name.startswith("customers.") for name in columns)


def test_profile_single_dataset_reports_roles_and_quality(tools, ctx) -> None:
    result = run_tool(tools, ctx, "profile_dataset", dataset="orders")
    assert result.ok
    rows = {row[0]: row for row in result.table.rows}
    assert rows["revenue"][1] == "numeric"
    assert rows["order_date"][1] == "datetime"
    assert rows["order_id"][1] == "identifier"
    assert "1 quality warning" in result.summary or "quality warning(s)" in result.summary
    assert "suggested_analyses" in result.detail


def test_profile_unknown_column_explains_alternatives(tools, ctx) -> None:
    result = run_tool(tools, ctx, "profile_dataset", dataset="orders", columns=["revenu"])
    # Near-miss names are resolved; a real miss must be an actionable error.
    assert result.ok
    assert result.table.rows[0][0] == "revenue"

    bad = run_tool(tools, ctx, "profile_dataset", dataset="orders", columns=["nonsense_column"])
    assert not bad.ok
    assert "revenue" in bad.error


def test_profile_unknown_dataset_lists_loaded(tools, ctx) -> None:
    result = run_tool(tools, ctx, "profile_dataset", dataset="ghost")
    assert not result.ok
    assert "orders" in result.error and "customers" in result.error


# ---------------------------------------------------------------------------
# run_sql
# ---------------------------------------------------------------------------


def test_sql_aggregates_and_joins(tools, ctx) -> None:
    grouped = run_tool(
        tools, ctx, "run_sql",
        query="select region, sum(revenue) as total, count(*) as n from orders group by 1 order by 2 desc",
    )
    assert grouped.ok
    assert grouped.table.columns == ["region", "total", "n"]
    assert sum(row[2] for row in grouped.table.rows) == 240

    joined = run_tool(
        tools, ctx, "run_sql",
        query=(
            "select c.segment, sum(o.revenue) as total from orders o "
            "join customers c using(customer_id) group by 1 order by 2 desc"
        ),
    )
    assert joined.ok
    assert {row[0] for row in joined.table.rows} <= {"Consumer", "SMB", "Enterprise"}


def test_sql_single_value_summary(tools, ctx) -> None:
    result = run_tool(tools, ctx, "run_sql", query="select count(*) as n from orders")
    assert result.ok
    assert "single value" in result.summary
    assert result.detail["total_rows"] == 1


def test_sql_rejects_writes_and_multiple_statements(tools, ctx) -> None:
    for query in [
        "drop table orders",
        "delete from orders",
        "select 1; drop table orders",
        "insert into orders values (1)",
        "attach 'x.db' as y",
    ]:
        result = run_tool(tools, ctx, "run_sql", query=query)
        assert not result.ok, query
        assert "read-only" in result.error or "single SQL statement" in result.error
    # And the table survives.
    assert run_tool(tools, ctx, "run_sql", query="select count(*) as n from orders").ok


def test_sql_error_includes_schema_hint(tools, ctx) -> None:
    result = run_tool(tools, ctx, "run_sql", query="select nope from orders")
    assert not result.ok
    assert "orders" in result.error and "revenue" in result.error


def test_sql_respects_max_rows(tools, ctx) -> None:
    result = run_tool(tools, ctx, "run_sql", query="select * from orders", max_rows=5)
    assert result.ok
    assert len(result.table.rows) == 5
    assert result.table.total_rows == 240
    assert result.table.truncated


def test_sql_empty_query_is_an_error(tools, ctx) -> None:
    result = run_tool(tools, ctx, "run_sql", query="   ")
    assert not result.ok
    assert "must not be empty" in result.error


def test_sql_window_function(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "run_sql",
        query=(
            "select region, order_date, revenue, "
            "row_number() over (partition by region order by revenue desc) as rn "
            "from orders qualify rn = 1 order by revenue desc"
        ),
    )
    assert result.ok
    assert len(result.table.rows) == 3


# ---------------------------------------------------------------------------
# run_python
# ---------------------------------------------------------------------------


def test_python_returns_dataframe_with_group_labels(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "run_python", dataset="orders",
        code="result = df.groupby('region')['revenue'].agg(['sum','mean']).round(2)",
    )
    assert result.ok, result.error
    assert result.table.columns[0] == "region"  # the group index is preserved
    assert set(result.table.columns) == {"region", "sum", "mean"}


def test_python_scalar_and_stdout(tools, ctx) -> None:
    scalar = run_tool(
        tools, ctx, "run_python", dataset="orders",
        code="result = round(float(df['revenue'].mean()), 2)",
    )
    assert scalar.ok
    assert "result" in scalar.summary

    printed = run_tool(tools, ctx, "run_python", code="print(len(tables['customers']))")
    assert printed.ok
    assert "40" in printed.summary


def test_python_can_reach_every_table(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "run_python",
        code="result = {name: frame.shape[0] for name, frame in tables.items()}",
    )
    assert result.ok
    assert "orders" in result.summary and "customers" in result.summary


def test_python_blocks_dangerous_imports_and_calls(tools, ctx) -> None:
    for code in ["import os\nresult = os.getcwd()", "import subprocess", "result = open('x')"]:
        result = run_tool(tools, ctx, "run_python", code=code)
        assert not result.ok, code


def test_python_reports_errors_without_killing_the_agent(tools, ctx) -> None:
    result = run_tool(tools, ctx, "run_python", dataset="orders", code="result = df['nope'].sum()")
    assert not result.ok
    assert "KeyError" in result.error or "nope" in result.error


def test_python_timeout_kills_the_process(tools, ctx) -> None:
    result = run_tool(tools, ctx, "run_python", code="while True:\n    pass", timeout=1)
    assert not result.ok
    assert "did not finish" in result.error


def test_python_handles_empty_code(tools, ctx) -> None:
    result = run_tool(tools, ctx, "run_python", code="   ")
    assert not result.ok
    assert "must not be empty" in result.error


def test_python_series_result_becomes_a_table(tools, ctx) -> None:
    result = run_tool(
        tools, ctx, "run_python", dataset="orders",
        code="result = df.groupby('channel')['quantity'].sum()",
    )
    assert result.ok
    assert result.table.columns == ["channel", "quantity"] or len(result.table.columns) == 2


def test_data_table_roundtrip() -> None:
    table = DataTable(["a", "b"], [[1, "x"], [2, None]], total_rows=2)
    restored = DataTable.from_dict(table.to_dict())
    assert restored.columns == ["a", "b"]
    assert restored.rows[1][1] is None
    assert "a" in table.render()
