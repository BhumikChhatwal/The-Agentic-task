"""Memory tools — the agent's own interface to the persistent store.

These are deliberately *tools* rather than hidden prompt plumbing: the model
decides what is durable enough to write, can search what it already knows, and can
correct or delete. Everything the agent writes this way is inspectable and
reversible from the CLI (``byteanalyst memory ...``).
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolError
from ..formatting import DataTable
from ..memory.models import KINDS
from .base import Tool, ToolContext, as_float, as_int, optional, require


def _memory(ctx: ToolContext):
    if ctx.memory is None:
        raise ToolError("memory is disabled for this session (start the agent with memory enabled)")
    return ctx.memory


class RememberTool(Tool):
    name = "remember"
    category = "memory"
    description = (
        "Save something durable so it changes future sessions. Use it when the user defines a "
        "term (`active customers` excludes cancelled accounts), states a preference (weekly "
        "rather than daily summaries), corrects you, or names a reusable filter. Do NOT use it "
        "for per-question facts, transient numbers, or anything already in the data. Keep `text` "
        "short and self-contained, always set `keys` to the columns/tables it applies to, and put "
        "a machine-usable predicate in `payload` (e.g. {\"sql\": \"status <> 'cancelled'\"}) "
        "whenever the memory is a filter or definition."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The fact, in one short sentence and in the user's own terms.",
            },
            "kind": {
                "type": "string",
                "enum": list(KINDS),
                "description": "definition | preference | correction | filter | fact | note",
            },
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns and/or table names this applies to. Used for retrieval.",
            },
            "payload": {
                "type": "object",
                "description": "Optional structured form, e.g. {\"sql\": \"status = 'active'\"}.",
            },
            "scope": {
                "type": "string",
                "description": "Dataset name this applies to, or 'global' (default).",
            },
            "confidence": {
                "type": "number",
                "description": "0-1. Use 1.0 for facts the user stated, lower for your own inferences.",
            },
        },
        "required": ["text", "kind"],
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        store = _memory(ctx)
        text = str(require(kwargs, "text"))
        kind = str(require(kwargs, "kind")).strip().lower()
        keys = kwargs.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        payload = kwargs.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        scope = str(optional(kwargs, "scope", "global") or "global")
        default_conf = 0.95 if kind in {"correction", "filter", "definition", "preference"} else 0.8
        confidence = as_float(optional(kwargs, "confidence", default_conf), "confidence")

        outcome = store.save(
            text=text, kind=kind, keys=keys, payload=payload, scope=scope,
            confidence=confidence, source="agent",
        )
        detail = {
            "memory_id": outcome.record.id,
            "action": outcome.action,
            "kind": outcome.record.kind,
            "scope": outcome.record.scope,
            "keys": outcome.record.keys,
            "confidence": outcome.record.confidence,
        }
        superseded = [c.describe() for c in outcome.superseded]
        if superseded:
            detail["superseded"] = superseded
        message = {
            "created": f"Saved a new {kind} memory [{outcome.record.id}]: \"{text}\"",
            "reinforced": f"Confirmed an existing memory [{outcome.record.id}] (confidence raised): \"{text}\"",
            "superseded": (
                f"Saved [{outcome.record.id}] and superseded {len(superseded)} older memory(ies): \"{text}\""
            ),
        }.get(outcome.action, f"Saved [{outcome.record.id}]: \"{text}\"")
        return self.result(message, detail=detail)


class RecallMemoryTool(Tool):
    name = "recall"
    category = "memory"
    description = (
        "Search persistent memory for relevant facts, definitions, preferences and filters. "
        "Use it when a question uses a term you have not defined, when the user asks what you "
        "remember, or before re-deriving something that may already be settled."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "kind": {"type": "string", "enum": list(KINDS), "description": "Restrict to one kind."},
            "limit": {"type": "integer", "description": "Maximum memories to return (default 8)."},
        },
        "required": ["query"],
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        store = _memory(ctx)
        query = str(require(kwargs, "query"))
        kind = optional(kwargs, "kind")
        limit = max(1, min(as_int(optional(kwargs, "limit", 8), "limit"), 50))
        results = store.retrieve(
            query, context_keys=ctx.context_keys, scope=ctx.primary_name,
            limit=limit, kind=kind, touch=True,
        )
        if not results:
            return self.result(
                f"Nothing in persistent memory matches {query!r} "
                f"({store.count()} memory(ies) stored in total).",
                detail={"matches": 0, "total_memories": store.count()},
            )
        table = DataTable(
            ["id", "kind", "score", "scope", "keys", "text"],
            [
                [r.record.id, r.record.kind, round(r.score, 3), r.record.scope,
                 ", ".join(r.record.keys), r.record.text]
                for r in results
            ],
        )
        detail = {
            "matches": len(results),
            "total_memories": store.count(),
            "results": [r.to_dict() for r in results],
        }
        return self.result(
            f"Found {len(results)} relevant memory(ies) for {query!r}. Apply any definition or "
            "filter you find here, and say explicitly that you did.",
            table=table, detail=detail,
        )


class ForgetMemoryTool(Tool):
    name = "manage_memory"
    category = "memory"
    description = (
        "Inspect or correct persistent memory. Actions: `list` every stored memory, "
        "`search` without changing anything, `forget` one by id, `edit` a memory's text or "
        "payload, `clear` all memories (needs confirm=true). Use this when the user asks what "
        "you remember, wants something forgotten, or when you spot a stale memory."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "forget", "edit", "clear"],
                "description": "What to do.",
            },
            "memory_id": {"type": "string", "description": "Target memory id (forget/edit)."},
            "text": {"type": "string", "description": "Replacement text (edit)."},
            "payload": {"type": "object", "description": "Replacement payload (edit)."},
            "kind": {"type": "string", "description": "Filter `list` by kind."},
            "confirm": {"type": "boolean", "description": "Required true for `clear`."},
        },
        "required": ["action"],
    }

    def run(self, ctx: ToolContext, **kwargs: Any) -> Any:
        store = _memory(ctx)
        action = str(require(kwargs, "action")).strip().lower()

        if action == "list":
            kind = optional(kwargs, "kind")
            records = store.list(kind=kind, include_inactive=False)
            if not records:
                return self.result("Memory is empty.", detail={"count": 0})
            table = DataTable(
                ["id", "kind", "scope", "keys", "uses", "confidence", "text"],
                [
                    [r.id, r.kind, r.scope, ", ".join(r.keys), r.uses,
                     round(r.confidence, 2), r.text]
                    for r in records
                ],
                total_rows=len(records),
            )
            return self.result(
                f"{len(records)} active memory(ies). Stats: {store.stats()}", table=table
            )

        if action == "forget":
            memory_id = str(require(kwargs, "memory_id"))
            record = store.get(memory_id)
            if record is None or store.forget(memory_id) is False:
                raise ToolError(f"no memory with id {memory_id!r}")
            return self.result(
                f"Forgotten: [{memory_id}] \"{record.text}\". It is deactivated (kept in history).",
                detail={"memory_id": memory_id, "action": "forgotten"},
            )

        if action == "edit":
            memory_id = str(require(kwargs, "memory_id"))
            record = store.edit(
                memory_id,
                text=optional(kwargs, "text"),
                payload=kwargs.get("payload"),
            )
            return self.result(
                f"Updated [{record.id}]: \"{record.text}\"",
                detail={"memory_id": record.id, "text": record.text, "payload": record.payload},
            )

        if action == "clear":
            if not optional(kwargs, "confirm", False):
                raise ToolError("clearing all memory requires confirm=true")
            count = store.clear(hard=False)
            return self.result(
                f"Deactivated {count} memory(ies). They remain in history and can be restored by id.",
                detail={"cleared": count},
            )

        raise ToolError(f"unknown action {action!r}")
