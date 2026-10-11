"""Small presentation helpers shared by the CLI, the web UI and the tools.

Kept dependency-free (no ``tabulate``/``rich``) so the package installs light.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "grey": "\033[90m",
}


def color(text: str, *styles: str, enabled: bool = True) -> str:
    if not enabled or not styles:
        return text
    prefix = "".join(ANSI.get(style, "") for style in styles)
    return f"{prefix}{text}{ANSI['reset']}"


def truncate(text: Any, width: int) -> str:
    """Truncate ``text`` to ``width`` characters using a single-character ellipsis."""
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\r", " ").replace("\n", "\\n")
    if len(s) <= width:
        return s
    if width <= 1:
        return s[:width]
    return s[: width - 1] + "\u2026"


def format_number(value: Any, digits: int = 4) -> str:
    """Format numbers for humans: thousands separators, sensible float precision."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,}"
        if abs(value) >= 1e6 or (0 < abs(value) < 1e-4):
            return f"{value:.{digits}g}"
        rendered = f"{value:,.{digits}f}"
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return rendered
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


class DataTable:
    """A tiny, serialisable table used for tool output and rendering."""

    __slots__ = ("columns", "rows", "total_rows", "truncated")

    def __init__(
        self,
        columns: Sequence[str],
        rows: Iterable[Sequence[Any]],
        total_rows: int | None = None,
        truncated: bool = False,
    ) -> None:
        self.columns = [str(c) for c in columns]
        self.rows = [list(r) for r in rows]
        self.total_rows = total_rows if total_rows is not None else len(self.rows)
        self.truncated = truncated

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": [[_jsonify(v) for v in row] for row in self.rows],
            "total_rows": self.total_rows,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataTable":
        return cls(
            data.get("columns", []),
            data.get("rows", []),
            total_rows=data.get("total_rows"),
            truncated=data.get("truncated", False),
        )

    def render(self, max_colwidth: int = 28, max_rows: int | None = None) -> str:
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        cells = [
            [truncate(cell, max_colwidth) for cell in row] for row in rows
        ]
        widths = [len(str(c)) for c in self.columns]
        for row in cells:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))
        header = " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(self.columns))
        sep = "-+-".join("-" * w for w in widths)
        body = [
            " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            for row in cells
        ]
        lines = [header, sep, *body]
        hidden = self.total_rows - len(rows)
        if hidden > 0:
            lines.append(f"... {hidden:,} more row(s)")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.rows)


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None if math.isnan(value) else str(value)
        return value
    try:
        import numpy as np  # local import keeps module import cheap

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            f = float(value)
            return None if math.isnan(f) or math.isinf(f) else f
        if isinstance(value, (np.bool_,)):
            return bool(value)
    except Exception:  # pragma: no cover
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover
            pass
    return str(value)


@dataclass
class ToolResult:
    """What a tool hands back: a short summary, an optional table, extras."""

    ok: bool
    summary: str
    table: DataTable | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    tool: str = ""

    @classmethod
    def success(
        cls,
        summary: str,
        table: DataTable | None = None,
        detail: dict[str, Any] | None = None,
        artifacts: Sequence[str] | None = None,
        tool: str = "",
    ) -> "ToolResult":
        return cls(
            ok=True,
            summary=summary,
            table=table,
            detail=detail or {},
            artifacts=list(artifacts or []),
            tool=tool,
        )

    @classmethod
    def failure(cls, error: str, summary: str | None = None, tool: str = "") -> "ToolResult":
        return cls(ok=False, summary=summary or error, error=error, tool=tool)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "summary": self.summary,
            "table": self.table.to_dict() if self.table else None,
            "detail": self.detail,
            "artifacts": self.artifacts,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        table = data.get("table")
        return cls(
            ok=data.get("ok", False),
            summary=data.get("summary", ""),
            table=DataTable.from_dict(table) if table else None,
            detail=data.get("detail") or {},
            artifacts=list(data.get("artifacts") or []),
            error=data.get("error"),
            tool=data.get("tool", ""),
        )

    def to_llm_text(self, max_chars: int = 6000, max_rows: int = 40) -> str:
        """Render the result as text for the model.

        The summary is always included verbatim; the table is truncated so a
        wide result cannot blow up the context window.
        """
        parts: list[str] = []
        if not self.ok:
            parts.append(f"ERROR: {self.error or self.summary}")
        else:
            parts.append(self.summary)
        if self.detail:
            for key, value in self.detail.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple)) and len(value) > 12:
                    preview = ", ".join(format_number(v) if not isinstance(v, str) else v for v in value[:12])
                    parts.append(f"{key}: [{preview}, ... {len(value) - 12} more]")
                else:
                    parts.append(f"{key}: {format_number(value) if not isinstance(value, str) else value}")
        if self.table is not None and len(self.table):
            parts.append("")
            parts.append(self.table.render(max_rows=max_rows))
        if self.artifacts:
            parts.append("")
            parts.append("artifacts (files written): " + ", ".join(self.artifacts))
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[: max_chars - 40] + f"\n... [truncated, {len(text)} chars total]"
        return text


NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


def extract_numbers(text: str) -> list[str]:
    """Pull numeric literals out of prose, normalised for comparison."""
    found: list[str] = []
    for match in NUMBER_RE.finditer(text or ""):
        raw = match.group(0)
        if raw in {"-", ""}:
            continue
        found.append(raw.replace(",", ""))
    return found


def numbers_equivalent(a: str, b: str, rel_tol: float = 0.02) -> bool:
    """Are two numeric strings the same value (allowing rounding, e.g. 1.2 vs 1.234)?"""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if fa == fb:
        return True
    scale = max(abs(fa), abs(fb), 1e-12)
    if abs(fa - fb) / scale <= rel_tol:
        return True
    # Allow a rounded rendering of the same value: 1.23 == 1.234
    for digits in range(0, 5):
        if round(fa, digits) == round(fb, digits):
            return True
    return False


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return _jsonify(value)
