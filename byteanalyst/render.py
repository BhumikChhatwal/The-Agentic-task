"""Terminal rendering.

Kept separate from ``cli.py`` so the same event stream can be rendered here, in a
notebook, or over SSE in the web UI.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .agent import AgentAnswer, AgentEvent
from .formatting import color

LEVEL_STYLE = {
    "start": ("cyan", "bold"),
    "memory": ("magenta",),
    "tool_call": ("blue", "bold"),
    "tool_result": ("green",),
    "verification": ("yellow",),
    "saved": ("magenta", "bold"),
    "answer": ("cyan", "bold"),
    "error": ("red", "bold"),
}

#: Windows consoles still default to cp1252, which cannot encode box-drawing or
#: symbol glyphs. Rather than crash, transliterate when the stream cannot cope.
_ASCII_TRANSLATION = str.maketrans(
    {
        "\u2500": "-", "\u2501": "=", "\u2502": "|", "\u251c": "|", "\u2514": "|",
        "\u25b6": ">", "\u2699": "*", "\u2713": "ok", "\u2714": "ok", "\u2717": "XX",
        "\u26a0": "!", "\u2605": "*", "\u29c9": "@", "\u2026": "...", "\u00b7": "|",
        "\u00d7": "x", "\u2192": "->", "\u2264": "<=", "\u2265": ">=", "\u2013": "-",
        "\u00b2": "2", "\u00b3": "3", "\u00b0": "deg", "\u00b5": "u", "\u03c7": "chi",
        "\u2014": "--", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
)


def asciify(text: str) -> str:
    """Best-effort ASCII rendering of text containing symbols/box characters."""
    return text.translate(_ASCII_TRANSLATION).encode("ascii", "replace").decode("ascii")


class Renderer:
    def __init__(
        self,
        stream: TextIO | None = None,
        use_color: bool | None = None,
        trace: bool = True,
        error_stream: TextIO | None = None,
    ) -> None:
        self.stream = stream or sys.stdout
        # Diagnostics go to stderr so `--json` output on stdout stays parseable.
        self.error_stream = error_stream or sys.stderr
        self.trace = trace
        if use_color is None:
            use_color = hasattr(self.stream, "isatty") and bool(self.stream.isatty())
        self.use_color = use_color

    # -- helpers ----------------------------------------------------------
    def paint(self, text: str, *styles: str) -> str:
        return color(text, *styles, enabled=self.use_color)

    def can_encode(self, text: str) -> bool:
        encoding = getattr(self.stream, "encoding", None) or "utf-8"
        try:
            text.encode(encoding, "strict")
            return True
        except (UnicodeEncodeError, LookupError, TypeError):
            return False

    def write(self, text: str = "") -> None:
        out = text + "\n"
        if not self.can_encode(out):
            out = asciify(out)
        try:
            self.stream.write(out)
        except UnicodeEncodeError:  # last resort: never crash on a glyph
            self.stream.write(out.encode("ascii", "replace").decode("ascii"))
        self.stream.flush()

    # -- events -----------------------------------------------------------
    def event(self, event: AgentEvent) -> None:
        handler = getattr(self, f"_on_{event.type}", None)
        if handler is not None:
            handler(event.data)

    def _on_start(self, data: dict[str, Any]) -> None:
        if not self.trace:
            return
        self.write(
            self.paint("▶ ", "cyan")
            + self.paint(data.get("question", ""), "bold")
            + self.paint(f"   [{data.get('provider')} · {data.get('model')}]", "grey")
        )

    def _on_memory(self, data: dict[str, Any]) -> None:
        if not self.trace:
            return
        self.write(self.paint(f"  ⧉ {data.get('count', 0)} memory(ies) recalled", "magenta"))
        for item in data.get("items", []):
            self.write(
                self.paint(
                    f"      [{item['id']}] ({item['kind']}, {item['score']}) {item['text']}",
                    "grey",
                )
            )

    def _on_tool_call(self, data: dict[str, Any]) -> None:
        if not self.trace:
            return
        arguments = _compact_arguments(data.get("arguments") or {})
        self.write(
            self.paint(f"  ⚙ {data.get('name')}", "blue", "bold")
            + self.paint(f"({arguments})", "grey")
        )

    def _on_tool_result(self, data: dict[str, Any]) -> None:
        if not self.trace:
            return
        mark = "✓" if data.get("ok") else "✗"
        summary = (data.get("summary") or "").strip().replace("\n", " ")
        if not data.get("ok"):
            summary = (data.get("error") or summary)[:300]
        self.write(
            self.paint(f"  {mark} ", "green" if data.get("ok") else "red")
            + self.paint(f"{summary[:280]}", "grey")
            + self.paint(f"   ({data.get('duration_ms', 0)} ms)", "grey")
        )
        table = data.get("table")
        if table and table.get("rows"):
            headers = table.get("columns", [])
            for row in table["rows"][:5]:
                cells = " | ".join(str(c) for c in row[:6])
                self.write(self.paint(f"      {cells}", "grey"))
            hidden = (table.get("total_rows") or 0) - min(5, len(table["rows"]))
            if hidden > 0:
                self.write(self.paint(f"      ... {hidden:,} more row(s)", "grey"))
        for artifact in data.get("artifacts") or []:
            self.write(self.paint(f"      🖼  {artifact}", "cyan"))

    def _on_verification(self, data: dict[str, Any]) -> None:
        if not self.trace:
            return
        if data.get("grounded"):
            self.write(self.paint("  ✔ every number traced to a tool result", "green"))
        else:
            numbers = ", ".join(data.get("unsupported") or [])
            self.write(
                self.paint(f"  ⚠ unverified number(s) in the draft answer: {numbers}", "yellow")
            )

    def _on_saved(self, data: dict[str, Any]) -> None:
        if not self.trace:
            return
        self.write(
            self.paint(f"  ★ memory {data.get('action')}", "magenta", "bold")
            + self.paint(f" [{data.get('id')}] ({data.get('kind')}) {data.get('text')}", "grey")
        )
        for superseded in data.get("superseded") or []:
            self.write(self.paint(f"      superseded: {superseded}", "grey"))

    def _on_error(self, data: dict[str, Any]) -> None:
        try:
            self.error_stream.write(f"  ! {data.get('where')}: {data.get('message')}\n")
            self.error_stream.flush()
        except (UnicodeEncodeError, OSError):  # never fail while reporting a failure
            pass

    def _on_answer(self, data: dict[str, Any]) -> None:
        return  # printed by answer() so ordering is explicit

    # -- final block ------------------------------------------------------
    def answer(self, result: AgentAnswer) -> None:
        self.write()
        self.write(self.paint("─" * 72, "grey"))
        self.write(result.answer.strip())
        self.write(self.paint("─" * 72, "grey"))
        footer = [
            f"tools: {', '.join(result.tools_used) or 'none'}",
            f"steps: {result.iterations}",
            f"{result.duration_ms} ms",
        ]
        if result.grounded:
            footer.append("grounded: yes")
        else:
            footer.append("grounded: no → " + ", ".join(result.unsupported_numbers[:6]))
        self.write(self.paint("  " + "  ·  ".join(footer), "grey"))
        if result.artifacts:
            self.write(self.paint(f"  artifacts: {len(result.artifacts)}", "grey"))
            for path in result.artifacts:
                self.write(self.paint(f"    {path}", "grey"))


def _compact_arguments(arguments: dict[str, Any], width: int = 160) -> str:
    parts = []
    for key, value in arguments.items():
        if isinstance(value, str):
            text = value.replace("\n", "\\n")
            if len(text) > 90:
                text = text[:87] + "..."
            parts.append(f"{key}={text!r}")
        else:
            parts.append(f"{key}={json.dumps(value, default=str)[:60]}")
    joined = ", ".join(parts)
    return joined if len(joined) <= width else joined[: width - 3] + "..."
