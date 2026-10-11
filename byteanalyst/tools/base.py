"""The tool abstraction shared by every analysis tool.

A tool declares a JSON-Schema parameter block, receives validated keyword
arguments, and returns a :class:`~byteanalyst.formatting.ToolResult`. Tools raise
``ToolError`` for anything the *model* can fix (unknown column, bad enum value);
the executor turns that into a readable error result so the agent can retry.
"""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from ..config import Config
from ..datasets import Dataset, DatasetRegistry
from ..errors import ToolError
from ..formatting import ToolResult


@dataclass
class ToolContext:
    """Everything a tool is allowed to touch."""

    registry: DatasetRegistry
    config: Config
    memory: Any = None  # MemoryStore | None, typed loosely to avoid a cycle
    session_id: str = "default"
    artifacts: list[str] = field(default_factory=list)

    def dataset(self, name: str | None = None) -> Dataset:
        return self.registry.get(name)

    def frame(self, name: str | None = None) -> pd.DataFrame:
        return self.dataset(name).frame

    @property
    def context_keys(self) -> set[str]:
        return self.registry.context_keys()

    @property
    def primary_name(self) -> str | None:
        primary = self.registry.primary
        return primary.name if primary else None


class Tool(ABC):
    """Base class for agent tools."""

    name: ClassVar[str] = ""
    category: ClassVar[str] = "analysis"
    description: ClassVar[str] = ""
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def spec(self) -> dict[str, Any]:
        """OpenAI-style function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------
    def result(self, summary: str, **kwargs: Any) -> ToolResult:
        return ToolResult.success(summary, tool=self.name, **kwargs)

    def fail(self, error: str) -> ToolResult:
        return ToolResult.failure(error, tool=self.name)


# ---------------------------------------------------------------------------
# Argument validation helpers (the model is not always well-behaved)
# ---------------------------------------------------------------------------


def require(params: dict[str, Any], key: str, types: tuple[type, ...] = (str,)) -> Any:
    value = params.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ToolError(f"missing required argument {key!r}")
    if not isinstance(value, types):
        raise ToolError(
            f"argument {key!r} must be {'/'.join(t.__name__ for t in types)}, "
            f"got {type(value).__name__}"
        )
    return value


def optional(params: dict[str, Any], key: str, default: Any = None) -> Any:
    value = params.get(key)
    if value is None or value == "":
        return default
    return value


def as_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"argument {key!r} must be an integer, got {value!r}") from exc


def as_float(value: Any, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"argument {key!r} must be a number, got {value!r}") from exc


def resolve_column(frame: pd.DataFrame, name: str, role: str | None = None) -> str:
    """Find a column by exact, case-insensitive or near match; else explain."""
    columns = [str(c) for c in frame.columns]
    if name in columns:
        return name
    lowered = {c.lower(): c for c in columns}
    if str(name).lower() in lowered:
        return lowered[str(name).lower()]
    # tolerate `table.column` and backticks/quotes from SQL habits
    cleaned = str(name).split(".")[-1].strip('`"\'[] ')
    if cleaned in columns:
        return cleaned
    if cleaned.lower() in lowered:
        return lowered[cleaned.lower()]

    if role == "numeric":
        candidates = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    elif role == "datetime":
        candidates = [c for c in columns if pd.api.types.is_datetime64_any_dtype(frame[c])]
    elif role == "categorical":
        candidates = [
            c for c in columns
            if not pd.api.types.is_numeric_dtype(frame[c])
            and not pd.api.types.is_datetime64_any_dtype(frame[c])
        ]
    else:
        candidates = columns

    close = difflib.get_close_matches(str(name), candidates or columns, n=3, cutoff=0.6)
    if len(close) == 1:
        return close[0]
    if close:
        raise ToolError(
            f"column {name!r} is ambiguous or unknown; did you mean one of: {', '.join(close)}?"
        )
    raise ToolError(
        f"unknown column {name!r}. Available columns: {', '.join(columns)}"
    )


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]


def require_numeric_column(frame: pd.DataFrame, name: str) -> str:
    column = resolve_column(frame, name)
    if not pd.api.types.is_numeric_dtype(frame[column]):
        numeric = numeric_columns(frame)
        hint = f" Numeric columns available: {', '.join(numeric)}" if numeric else " No numeric columns."
        raise ToolError(f"column {column!r} is not numeric.{hint}")
    return column
