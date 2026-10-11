"""Memory data model."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: The kinds of durable fact worth remembering. Each has a distinct lifetime and
#: a distinct effect on the agent's behaviour.
KINDS: dict[str, str] = {
    # What a column/metric actually means to the user (changes interpretation).
    "definition": "A meaning for a column, metric or term used by the user.",
    # How the user wants results presented (cadence, units, format).
    "preference": "A lasting preference about how answers should be produced.",
    # The user told the agent it was wrong; must override prior beliefs.
    "correction": "A correction that overrides an earlier belief or answer.",
    # A reusable, named filter (kept as a structured predicate where possible).
    "filter": "A named filter or population definition to apply going forward.",
    # A stable fact about the data discovered and confirmed during analysis.
    "fact": "A durable, verified fact about the dataset.",
    # Lower-stakes context that is useful but not authoritative.
    "note": "Background context; lower priority than the kinds above.",
}

_PUNCT_RE = re.compile(r"[^\w\s]+")


def normalize_text(text: str) -> str:
    """Normalise for equality comparisons between memories."""
    lowered = str(text or "").lower().strip()
    lowered = _PUNCT_RE.sub(" ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class MemoryRecord:
    id: str
    kind: str
    text: str
    keys: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    scope: str = "global"
    confidence: float = 0.8
    source: str = "user"
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    uses: int = 0
    last_used_at: str | None = None
    superseded_by: str | None = None
    active: bool = True

    @property
    def subject(self) -> str:
        return "|".join(sorted(k.lower() for k in self.keys)) or "-"

    @property
    def normalized(self) -> str:
        return normalize_text(self.text)

    def to_row(self) -> tuple:
        return (
            self.id,
            self.kind,
            self.text,
            json.dumps(self.keys),
            json.dumps(self.payload),
            self.scope,
            float(self.confidence),
            self.source,
            self.created_at,
            self.updated_at,
            int(self.uses),
            self.last_used_at,
            self.superseded_by,
            1 if self.active else 0,
        )

    @classmethod
    def from_row(cls, row: tuple) -> "MemoryRecord":
        (
            id_, kind, text, keys, payload, scope, confidence, source,
            created_at, updated_at, uses, last_used_at, superseded_by, active,
        ) = row
        try:
            keys_list = json.loads(keys) if keys else []
        except (TypeError, ValueError):
            keys_list = []
        try:
            payload_dict = json.loads(payload) if payload else {}
        except (TypeError, ValueError):
            payload_dict = {}
        if not isinstance(payload_dict, dict):
            payload_dict = {"value": payload_dict}
        return cls(
            id=id_,
            kind=kind,
            text=text,
            keys=list(keys_list),
            payload=payload_dict,
            scope=scope or "global",
            confidence=float(confidence or 0.0),
            source=source or "user",
            created_at=created_at or utcnow(),
            updated_at=updated_at or utcnow(),
            uses=int(uses or 0),
            last_used_at=last_used_at,
            superseded_by=superseded_by,
            active=bool(active),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "keys": list(self.keys),
            "payload": dict(self.payload),
            "scope": self.scope,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "uses": self.uses,
            "last_used_at": self.last_used_at,
            "superseded_by": self.superseded_by,
            "active": self.active,
        }

    def describe(self) -> str:
        bits = [f"[{self.id}] ({self.kind}, confidence {self.confidence:.2f}"]
        if self.scope and self.scope != "global":
            bits.append(f", scope {self.scope}")
        if self.keys:
            bits.append(f", keys: {', '.join(self.keys)}")
        bits.append(")")
        line = "".join(bits) + f" {self.text}"
        if self.payload:
            line += f"  [payload: {json.dumps(self.payload, default=str)}]"
        return line


@dataclass
class SaveOutcome:
    """Result of writing a memory, so the caller can report what happened."""

    action: str  # created | reinforced | superseded | updated
    record: MemoryRecord
    superseded: list[MemoryRecord] = field(default_factory=list)
    note: str = ""

    @property
    def changed(self) -> bool:
        return self.action in {"created", "superseded", "updated"}
