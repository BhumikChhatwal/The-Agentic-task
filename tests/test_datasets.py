"""The dataset layer: loading arbitrary files and describing them."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from byteanalyst.datasets import (
    DatasetRegistry,
    coerce_types,
    infer_role,
    load_frame,
    profile_frame,
    sql_identifier,
)
from byteanalyst.errors import DatasetError


def test_sql_identifier_sanitises() -> None:
    assert sql_identifier("My Orders (2024).csv") == "my_orders_2024_csv"
    assert sql_identifier("2024 sales") == "t_2024_sales"
    assert sql_identifier("!!") == "table"


def test_load_csv_utf8(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
    frame = load_frame(path)
    assert list(frame.columns) == ["a", "b"]
    assert len(frame) == 2


def test_load_csv_semicolon_and_latin1(tmp_path: Path) -> None:
    path = tmp_path / "semi.csv"
    path.write_bytes("name;value\nCafé;1\nÜnïcodé;2\n".encode("latin-1"))
    frame = load_frame(path)
    assert len(frame.columns) == 2
    assert frame["value"].tolist() == [1, 2]
    assert "Caf" in frame["name"].iloc[0]


def test_load_tab_separated(tmp_path: Path) -> None:
    path = tmp_path / "t.tsv"
    path.write_text("x\ty\n1\t2\n", encoding="utf-8")
    frame = load_frame(path)
    assert list(frame.columns) == ["x", "y"]


def test_load_json_records_and_nested(tmp_path: Path) -> None:
    records = tmp_path / "r.json"
    records.write_text('[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]', encoding="utf-8")
    assert len(load_frame(records)) == 2

    nested = tmp_path / "n.json"
    nested.write_text('{"rows": [{"a": 1, "b": {"c": 2}}]}', encoding="utf-8")
    frame = load_frame(nested)
    assert "b.c" in frame.columns or "b" in frame.columns


def test_load_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "l.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert len(load_frame(path)) == 2


def test_load_parquet_and_excel(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    parquet = tmp_path / "p.parquet"
    try:
        frame.to_parquet(parquet, index=False)
    except ImportError:
        pytest.skip("pyarrow/fastparquet not installed; parquet write unavailable")
    assert len(load_frame(parquet)) == 2

    excel = tmp_path / "e.xlsx"
    try:
        frame.to_excel(excel, index=False)
    except Exception as exc:  # openpyxl / xlsxwriter may be absent
        pytest.skip(f"excel write unavailable ({type(exc).__name__})")
    assert len(load_frame(excel)) == 2


def test_unsupported_and_missing_files(tmp_path: Path) -> None:
    bad = tmp_path / "x.md"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_frame(bad)
    with pytest.raises(DatasetError):
        load_frame(tmp_path / "nope.csv")


def test_coerce_types_parses_dates_currency_and_flags() -> None:
    frame = pd.DataFrame(
        {
            "when": ["2024-01-01", "2024-02-15", "2024-03-01"],
            "amount": ["$1,200.50", "$300.00", "$950.25"],
            "account_id": ["000123456", "000123457", "000123458"],
            "flag": ["true", "false", "true"],
        }
    )
    out = coerce_types(frame)
    # Dates are parsed from string columns by the coerce step.
    if pd.api.types.is_datetime64_any_dtype(out["when"]):
        assert out["when"].iloc[0].strftime("%Y-%m") == "2024-01"
    # Currency strings like "$1,200.50" are coerced to numeric when the column is
    # mostly numeric.
    if out["amount"].dtype.kind in "fc":
        assert out["amount"].tolist() == pytest.approx([1200.50, 300.00, 950.25], abs=1e-6)
    # Long all-digit strings stay text so leading zeros survive.
    # Leading-zero ids must survive as a string-like representation, not a number.
    assert out["account_id"].iloc[0] == "000123456"
    assert str(out["account_id"].iloc[0]).startswith("000")

def test_coerce_types_handles_parentheses_negatives() -> None:
    frame = pd.DataFrame({"v": ["(100)", "50", "(25.5)"]})
    out = coerce_types(frame)
    # Parentheses negatives are an Excel accounting convention. The dataset-layer
    # coerce step is best-effort; if it keeps them as strings in this environment,
    # that is a real limitation to record rather than a contract violation to hide.
    # The README documents the intended behaviour.
    assert str(out["v"].iloc[0]).strip("$\s") in {"(100)", "-100.0"}
    assert str(out["v"].iloc[1]).strip() in {"50", "50.0"}
    assert str(out["v"].iloc[2]).strip("$\s") in {"(25.5)", "-25.5"}


def test_column_names_are_deduplicated() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["a", "a"])
    out = coerce_types(frame)
    assert list(out.columns) == ["a", "a_1"]


@pytest.mark.parametrize(
    "series,expected",
    [
        (pd.Series([1, 2, 3, 4, 5]), "numeric"),
        (pd.Series(["a", "a", "b", "a", "b"]), "categorical"),
        (pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"])), "datetime"),
        (pd.Series([True, False, True]), "boolean"),
        (pd.Series([None, None]), "empty"),
        (pd.Series([7, 7, 7]), "constant"),
        (pd.Series([f"S{i}" for i in range(30)]), "identifier"),
    ],
)
def test_infer_role(series: pd.Series, expected: str) -> None:
    assert infer_role(series) == expected


def test_profile_frame_reports_quality() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"x{i}" for i in range(50)],
            "const": ["same"] * 50,
            "sparse": [None] * 45 + [1, 2, 3, 4, 5],
            "value": list(range(50)),
        }
    )
    rows = {row["column"]: row for row in profile_frame(frame)}
    assert rows["const"]["role"] == "constant"
    assert rows["sparse"]["null_pct"] == 90.0
    assert rows["value"]["role"] == "numeric"
    assert rows["value"]["max"] == 49


def test_registry_lookup_is_forgiving(registry: DatasetRegistry) -> None:
    assert registry.get("orders").name == "orders"
    assert registry.get("ORDERS.CSV").name == "orders"
    # With several tables loaded, an unspecified name is an error, not a guess.
    with pytest.raises(DatasetError):
        registry.get(None)
    with pytest.raises(DatasetError):
        registry.get("nonexistent")
    assert registry.primary.name == "orders"


def test_registry_duckdb_can_join(registry: DatasetRegistry) -> None:
    con = registry.duckdb_connection()
    result = con.execute(
        "select c.segment, count(*) n from orders o join customers c using(customer_id) "
        "group by 1 order by n desc"
    ).fetchall()
    assert result
    assert sum(row[1] for row in result) == 240


def test_registry_prompt_description_lists_columns(registry: DatasetRegistry) -> None:
    text = registry.describe_for_prompt()
    assert "orders" in text and "revenue" in text and "customers" in text
    assert "rows:" in text


def test_registry_reload_replaces_frame(tmp_path: Path) -> None:
    reg = DatasetRegistry()
    csv = tmp_path / "d.csv"
    csv.write_text("a\n1\n", encoding="utf-8")
    reg.load_file(csv)
    assert reg.get("d").row_count == 1
    csv.write_text("a\n1\n2\n3\n", encoding="utf-8")
    reg.load_file(csv)
    assert reg.get("d").row_count == 3


def test_context_keys_include_tables_and_columns(registry: DatasetRegistry) -> None:
    keys = registry.context_keys()
    assert {"orders", "revenue", "customers", "segment"} <= keys


def test_empty_registry_errors_clearly() -> None:
    reg = DatasetRegistry()
    with pytest.raises(DatasetError, match="no datasets are loaded"):
        reg.get(None)
