"""Loading arbitrary tabular files and describing them to the agent.

Nothing here is specific to a demo dataset. A file is sniffed, parsed, typed, and
given semantic roles (numeric / datetime / categorical / identifier / text ...)
that both the profiling tool and the system prompt use.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .errors import DatasetError

SUPPORTED_SUFFIXES = {
    ".csv", ".tsv", ".txt", ".psv",
    ".json", ".jsonl", ".ndjson",
    ".parquet", ".pq",
    ".xlsx", ".xls", ".xlsm",
}

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

_ID_NAME_RE = re.compile(
    r"(^|_)(id|ids|key|code|sku|ref|reference|number|no|uuid|guid)($|_)", re.IGNORECASE
)
_DATE_NAME_RE = re.compile(
    r"(date|time|timestamp|_at$|^at$|day|month|year|week|quarter)", re.IGNORECASE
)


def sql_identifier(name: str) -> str:
    """Turn an arbitrary dataset name into a safe, quoted-friendly identifier."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
    if not cleaned:
        cleaned = "table"
    if cleaned[0].isdigit():
        cleaned = "t_" + cleaned
    return cleaned.lower()


def _unique(names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = name
        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        out.append(name)
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_csv_with_sniffing(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            raw = path.read_text(encoding=encoding)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
            continue
        # Sniff the delimiter from the first non-empty lines.
        sample = "\n".join(line for line in raw.splitlines()[:20] if line.strip())
        delimiter = ","
        if sample:
            try:
                import csv

                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except Exception:
                counts = {d: sample.count(d) for d in [",", "\t", ";", "|"]}
                delimiter = max(counts, key=counts.get) if max(counts.values()) else ","
        try:
            return pd.read_csv(
                io.StringIO(raw),
                sep=delimiter,
                engine="python",
                skipinitialspace=True,
            )
        except Exception as exc:  # try the next encoding
            last_error = exc
    raise DatasetError(f"could not parse CSV file {path.name}: {last_error}")


def _read_json_any(path: Path) -> pd.DataFrame:
    import json

    text = None
    for encoding in _ENCODINGS:
        try:
            text = path.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        raise DatasetError(f"could not decode {path.name} with any known encoding")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(io.StringIO(text), lines=True)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path.name} is not valid JSON: {exc}") from exc

    def flatten(payload) -> pd.DataFrame:
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            # Prefer a list of records nested under some key, e.g. {"rows": [...]}.
            for value in payload.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    return pd.json_normalize(value)
            return pd.json_normalize(payload)
        raise DatasetError(f"unsupported JSON structure in {path.name}: {type(payload).__name__}")

    frame = flatten(data)
    if frame.empty and len(frame.columns) == 0:
        raise DatasetError(f"{path.name} contains no tabular data")
    return frame


def load_frame(path: str | Path) -> pd.DataFrame:
    """Load any supported tabular file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DatasetError(
            f"unsupported file type {suffix!r}; supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    try:
        if suffix in {".csv", ".tsv", ".txt", ".psv"}:
            frame = _read_csv_with_sniffing(path)
        elif suffix in {".json", ".jsonl", ".ndjson"}:
            frame = _read_json_any(path)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_excel(path)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"failed to read {path.name}: {exc}") from exc
    if frame is None or frame.empty and len(frame.columns) == 0:
        raise DatasetError(f"{path.name} contains no readable tabular data")
    return frame


def load_frame_from_records(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(records))


# ---------------------------------------------------------------------------
# Typing and semantic roles
# ---------------------------------------------------------------------------


def coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Best-effort type coercion applied after load so tools get sane dtypes."""
    out = frame.copy()
    out.columns = _unique([str(c).strip() for c in out.columns])
    for column in out.columns:
        series = out[column]
        if series.dtype == object:
            non_null = series.dropna()
            if len(non_null) == 0:
                continue
            # Numeric-looking strings ("1,234.5", "$1,200")
            as_str = non_null.astype(str).str.strip()
            numericish = as_str.str.replace(r"[,\s$€£%]", "", regex=True)
            numericish = numericish.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            converted = pd.to_numeric(numericish, errors="coerce")
            if converted.notna().mean() >= 0.95 and not _looks_like_identifier(as_str):
                out[column] = pd.to_numeric(
                    series.astype(str)
                    .str.strip()
                    .str.replace(r"[,\s$€£%]", "", regex=True)
                    .str.replace(r"^\((.*)\)$", r"-\1", regex=True),
                    errors="coerce",
                )
                continue
        if series.dtype == object:
            out[column] = _maybe_datetime(series)
    return out


def _looks_like_identifier(values: pd.Series) -> bool:
    """Long *all-digit* strings are ids, not measurements (so leading zeros survive).

    Currency and grouped numbers ("$1,200.50", "1 234") must not trip this: only a
    bare run of digits, long enough to be a code, counts.
    """
    sample = values.head(200)
    if not len(sample):
        return False
    digit_runs = sample[sample.str.fullmatch(r"\d+")]
    if len(digit_runs) < 0.9 * len(sample):
        return False
    return bool(digit_runs.str.len().min() >= 6)


def _maybe_datetime(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if len(non_null) == 0:
        return series
    sample = non_null.head(500).astype(str)
    # Avoid treating plain numbers written as strings as dates.
    if sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.9:
        return series
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (TypeError, ValueError):  # older pandas
        parsed = pd.to_datetime(sample, errors="coerce")
    if parsed.notna().mean() < 0.9:
        return series
    try:
        full = pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        full = pd.to_datetime(series, errors="coerce")
    if full.notna().mean() < 0.9:
        return series
    return full


def infer_role(series: pd.Series, name: str = "") -> str:
    """Classify a column into a semantic role used for advice and memory matching."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"
    n = len(non_null)
    unique = non_null.nunique()
    if unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        if unique == n and n > 1:
            return "identifier" if _ID_NAME_RE.search(name or "") else "numeric"
        return "numeric"
    # object / string
    unique_ratio = unique / n
    if unique == n and n > 1:
        return "identifier"
    if _ID_NAME_RE.search(name or "") and unique_ratio > 0.5:
        return "identifier"
    avg_len = float(non_null.astype(str).str.len().mean())
    if unique_ratio <= 0.35 and (unique <= 60 or unique_ratio <= 0.1):
        return "categorical"
    if avg_len > 40 or unique_ratio > 0.9:
        return "text"
    return "categorical"


def profile_frame(frame: pd.DataFrame, max_values: int = 5) -> list[dict[str, Any]]:
    """Per-column summary used by the profiling tool and the prompt builder."""
    rows: list[dict[str, Any]] = []
    row_count = len(frame)
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        info: dict[str, Any] = {
            "column": str(column),
            "dtype": str(series.dtype),
            "role": infer_role(series, str(column)),
            "non_null": int(non_null.shape[0]),
            "nulls": int(series.isna().sum()),
            "null_pct": round(float(series.isna().mean() * 100.0), 2) if row_count else 0.0,
            "unique": int(non_null.nunique()) if len(non_null) else 0,
        }
        role = info["role"]
        if role in {"numeric", "constant"} and pd.api.types.is_numeric_dtype(series):
            if len(non_null):
                info.update(
                    min=_num(non_null.min()),
                    max=_num(non_null.max()),
                    mean=_num(non_null.mean()),
                    median=_num(non_null.median()),
                    std=_num(non_null.std()),
                )
        elif role == "datetime" and len(non_null):
            info["min"] = str(non_null.min())
            info["max"] = str(non_null.max())
        if role in {"categorical", "boolean", "constant"} and len(non_null):
            counts = non_null.value_counts().head(max_values)
            info["top_values"] = [f"{idx} ({int(cnt)})" for idx, cnt in counts.items()]
        elif len(non_null):
            samples = [str(v) for v in non_null.head(max_values).tolist()]
            info["sample_values"] = [_short(v) for v in samples]
        rows.append(info)
    return rows


def _num(value: Any) -> Any:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    return value


def _short(text: str, width: int = 60) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Dataset + registry
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    source: str = "<memory>"
    notes: str = ""
    _profile: list[dict[str, Any]] | None = field(default=None, repr=False)

    @property
    def sql_name(self) -> str:
        return sql_identifier(self.name)

    @property
    def row_count(self) -> int:
        return int(len(self.frame))

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    def profile(self, refresh: bool = False) -> list[dict[str, Any]]:
        if self._profile is None or refresh:
            self._profile = profile_frame(self.frame)
        return self._profile

    def roles(self) -> dict[str, str]:
        return {row["column"]: row["role"] for row in self.profile()}

    def numeric_columns(self) -> list[str]:
        return [c for c, r in self.roles().items() if r == "numeric"]

    def datetime_columns(self) -> list[str]:
        return [c for c, r in self.roles().items() if r == "datetime"]

    def categorical_columns(self) -> list[str]:
        return [c for c, r in self.roles().items() if r in {"categorical", "boolean"}]


class DatasetRegistry:
    """Holds the datasets for one session and exposes them to SQL + tools."""

    def __init__(self, frame_rows: int = 8) -> None:
        self._datasets: dict[str, Dataset] = {}
        self._order: list[str] = []
        self.prompt_sample_rows = frame_rows

    # -- mutation ---------------------------------------------------------
    def add(self, name: str, frame: pd.DataFrame, source: str = "<memory>", notes: str = "") -> Dataset:
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        key = sql_identifier(name)
        if key in self._datasets:
            existing = self._datasets[key]
            existing.frame = frame
            existing.source = source
            existing._profile = None
            return existing
        dataset = Dataset(name=key, frame=frame, source=source, notes=notes)
        self._datasets[key] = dataset
        self._order.append(key)
        return dataset

    def load_file(self, path: str | Path, name: str | None = None, notes: str = "") -> Dataset:
        path = Path(path)
        frame = coerce_types(load_frame(path))
        dataset_name = name or path.stem
        return self.add(dataset_name, frame, source=str(path), notes=notes)

    def remove(self, name: str) -> bool:
        key = sql_identifier(name)
        if key in self._datasets:
            del self._datasets[key]
            self._order.remove(key)
            return True
        return False

    # -- lookup -----------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return sql_identifier(name) in self._datasets

    def __len__(self) -> int:
        return len(self._order)

    @property
    def primary(self) -> Dataset | None:
        if not self._order:
            return None
        return self._datasets[self._order[0]]

    def names(self) -> list[str]:
        return list(self._order)

    def datasets(self) -> list[Dataset]:
        return [self._datasets[n] for n in self._order]

    def get(self, name: str | None) -> Dataset:
        """Resolve a dataset name, being generous with what the model sends."""
        if not self._datasets:
            raise DatasetError("no datasets are loaded; load one with --data <file>")
        if name is None or str(name).strip() == "":
            if len(self._order) == 1:
                return self._datasets[self._order[0]]
            raise DatasetError(
                "multiple datasets are loaded; specify one of: " + ", ".join(self._order)
            )
        key = sql_identifier(name)
        if key in self._datasets:
            return self._datasets[key]
        # tolerate table aliases like "orders.csv" or "orders_final"
        for candidate in self._order:
            if candidate in key or key in candidate:
                return self._datasets[candidate]
        raise DatasetError(
            f"unknown dataset {name!r}; loaded datasets: {', '.join(self._order)}"
        )

    def resolve(self, name: str | None) -> Dataset:
        return self.get(name)

    # -- introspection ----------------------------------------------------
    def all_columns(self) -> dict[str, list[str]]:
        return {d.name: d.columns for d in self.datasets()}

    def column_lookup(self) -> dict[str, list[str]]:
        """Reverse index: column name -> datasets that contain it (for memory keys)."""
        index: dict[str, list[str]] = {}
        for dataset in self.datasets():
            for column in dataset.columns:
                index.setdefault(str(column).lower(), []).append(dataset.name)
        return index

    def context_keys(self) -> set[str]:
        """All table/column tokens, used to decide which memories are relevant."""
        keys: set[str] = set()
        for dataset in self.datasets():
            keys.add(dataset.name.lower())
            for column in dataset.columns:
                keys.add(str(column).lower())
        return keys

    def duckdb_connection(self):
        """A fresh in-memory DuckDB connection with every dataset registered."""
        import duckdb

        con = duckdb.connect(database=":memory:")
        for dataset in self.datasets():
            con.register(dataset.sql_name, dataset.frame)
        return con

    def describe_for_prompt(self, max_columns: int = 80) -> str:
        if not self._datasets:
            return "(no datasets loaded)"
        blocks: list[str] = []
        for dataset in self.datasets():
            lines = [
                f"### dataset `{dataset.name}`  (SQL table name: {dataset.sql_name})",
                f"rows: {dataset.row_count:,}  |  columns: {len(dataset.columns)}  |  source: {dataset.source}",
            ]
            profile = dataset.profile()
            for row in profile[:max_columns]:
                bits = [row["dtype"], f"nulls {row['null_pct']}%", f"unique {row['unique']:,}"]
                if row.get("min") is not None or row.get("max") is not None:
                    bits.append(f"range [{row.get('min')} .. {row.get('max')}]")
                if row.get("top_values"):
                    bits.append("top: " + ", ".join(row["top_values"]))
                elif row.get("sample_values"):
                    bits.append("e.g. " + ", ".join(row["sample_values"][:3]))
                lines.append(f"  - {row['column']} ({row['role']}): " + "; ".join(bits))
            if len(profile) > max_columns:
                lines.append(f"  ... and {len(profile) - max_columns} more columns")
            sample = dataset.frame.head(self.prompt_sample_rows)
            if len(sample):
                lines.append("  first rows:")
                lines.append(_indent(sample.to_string(index=False, max_cols=12), "    "))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def build_registry(paths: Sequence[str | Path], names: Sequence[str] | None = None) -> DatasetRegistry:
    """Convenience loader used by the CLI and web UI."""
    registry = DatasetRegistry()
    for index, path in enumerate(paths):
        name = names[index] if names and index < len(names) else None
        registry.load_file(path, name=name)
    return registry
