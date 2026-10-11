"""Provider-neutral chat + tool-calling types.

The agent loop only ever sees these types, so adding a provider means writing one
adapter rather than touching the agent. ``Message`` mirrors the OpenAI wire shape
because it is the de-facto interchange format for tool calling; the Anthropic
adapter translates it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


@dataclass
class AssistantTurn:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient(Protocol):
    """Minimal surface the agent needs from a provider."""

    model: str

    def chat(self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None) -> AssistantTurn:
        ...


def parse_json_array(text: str) -> list[Any]:
    """Extract the first JSON array from a model reply, tolerating prose and fences."""
    if not text:
        return []
    cleaned = re.sub(r"```(?:json)?", "", str(text)).strip()
    candidates = [cleaned]
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end > start:
        candidates.insert(0, cleaned[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, list):
                    return value
    return []


def parse_memory_json(text: str) -> list[dict[str, Any]]:
    """Parse a model's memory-extraction reply into validated candidate dicts."""
    allowed_kinds = {"definition", "preference", "correction", "filter", "fact", "note"}
    out: list[dict[str, Any]] = []
    for item in parse_json_array(text):
        if not isinstance(item, dict):
            continue
        text_value = str(item.get("text") or "").strip()
        if len(text_value) < 6 or len(text_value) > 240:
            continue
        kind = str(item.get("kind") or "fact").strip().lower()
        if kind not in allowed_kinds:
            kind = "fact"
        keys = item.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        keys = [str(k).strip().lower() for k in keys if str(k).strip()][:8]
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        try:
            confidence = float(item.get("confidence", 0.85))
        except (TypeError, ValueError):
            confidence = 0.85
        out.append(
            {
                "text": text_value,
                "kind": kind,
                "keys": keys,
                "payload": payload,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
    return out[:5]


def parse_tool_arguments(raw: Any) -> tuple[dict[str, Any], str]:
    """Tool arguments arrive as a JSON string (or already-decoded dict). Be lenient."""
    if isinstance(raw, dict):
        return raw, json.dumps(raw)
    text = raw if isinstance(raw, str) else ""
    if not text.strip():
        return {}, ""
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, text
        return {"value": parsed}, text
    # Nothing parseable: hand the model an empty call so it gets a schema reminder
    # rather than a crash, and keep the raw text for the transcript.
    return {}, text


_BARE_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]\s])")


def _json_candidates(text: str) -> list[str]:
    """Progressively more forgiving readings of a model's tool arguments.

    Well-behaved models emit strict JSON. Others emit JavaScript-flavoured
    objects (single quotes, unquoted keys, trailing commas); recovering those is
    worth a few lines because a lost tool call wastes a whole turn.
    """
    stripped = text.strip()
    candidates = [stripped]

    no_trailing = _TRAILING_COMMA_RE.sub(r"\1", stripped)
    candidates.append(no_trailing)

    double_quoted = no_trailing.replace("'", '"')
    candidates.append(double_quoted)

    quoted_keys = _BARE_KEY_RE.sub(r'\1"\2"\3', double_quoted)
    candidates.append(quoted_keys)
    candidates.append(_TRAILING_COMMA_RE.sub(r"\1", quoted_keys))

    # De-duplicate while preserving priority order.
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered
