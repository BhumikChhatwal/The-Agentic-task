"""The persistent memory store.

Storage is a single SQLite file (default ``.byteanalyst/memory.db``), so memory
survives process restarts and is easy to inspect with any SQLite client. Every
write is also recorded in an append-only ``memory_events`` audit log.

Written deliberately to be *small and curated*: nothing is saved automatically
unless it looks durable (a definition, a preference, a correction, a reusable
filter), and superseded records are kept as history rather than deleted.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import MemoryError_
from .models import KINDS, MemoryRecord, SaveOutcome, new_id, normalize_text, utcnow
from .retrieval import MemoryRetriever, Retrieved, tokenize

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    text          TEXT NOT NULL,
    keys          TEXT NOT NULL DEFAULT '[]',
    payload       TEXT NOT NULL DEFAULT '{}',
    scope         TEXT NOT NULL DEFAULT 'global',
    confidence    REAL NOT NULL DEFAULT 0.8,
    source        TEXT NOT NULL DEFAULT 'user',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    uses          INTEGER NOT NULL DEFAULT 0,
    last_used_at  TEXT,
    superseded_by TEXT,
    active        INTEGER NOT NULL DEFAULT 1,
    embedding     TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);

CREATE TABLE IF NOT EXISTS memory_events (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL,
    memory_id TEXT,
    action    TEXT NOT NULL,
    note      TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_COLUMNS = (
    "id, kind, text, keys, payload, scope, confidence, source, created_at, "
    "updated_at, uses, last_used_at, superseded_by, active"
)


def _similarity(a: str, b: str) -> float:
    """Token Jaccard similarity between two memory texts."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:", embedder=None) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()
        self.retriever = MemoryRetriever(embedder=embedder)

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low level --------------------------------------------------------
    def _log(self, memory_id: str | None, action: str, note: str = "") -> None:
        self._conn.execute(
            "INSERT INTO memory_events(at, memory_id, action, note) VALUES(?,?,?,?)",
            (utcnow(), memory_id, action, note),
        )

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.retriever.embedder is None or not texts:
            return []
        try:
            return [list(v) for v in self.retriever.embedder(list(texts))]
        except Exception:  # pragma: no cover - provider failures are non-fatal
            return []

    # -- reads ------------------------------------------------------------
    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return MemoryRecord.from_row(tuple(row)) if row else None

    def list(
        self,
        kind: str | None = None,
        scope: str | None = None,
        include_inactive: bool = False,
        limit: int = 200,
    ) -> list[MemoryRecord]:
        clauses, params = [], []
        if not include_inactive:
            clauses.append("active = 1")
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if scope:
            clauses.append("scope = ?")
            params.append(scope)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM memories {where} "
                "ORDER BY active DESC, updated_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [MemoryRecord.from_row(tuple(r)) for r in rows]

    def active_records(self) -> list[MemoryRecord]:
        return self.list(include_inactive=False, limit=5000)

    def count(self, active_only: bool = True) -> int:
        query = "SELECT COUNT(*) FROM memories" + (" WHERE active = 1" if active_only else "")
        with self._lock:
            return int(self._conn.execute(query).fetchone()[0])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            by_kind = {
                row["kind"]: row["n"]
                for row in self._conn.execute(
                    "SELECT kind, COUNT(*) AS n FROM memories WHERE active = 1 GROUP BY kind"
                )
            }
            inactive = int(
                self._conn.execute("SELECT COUNT(*) FROM memories WHERE active = 0").fetchone()[0]
            )
        return {"active": self.count(True), "superseded": inactive, "by_kind": by_kind}

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, at, memory_id, action, note FROM memory_events "
                "ORDER BY seq DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- writes -----------------------------------------------------------
    def save(
        self,
        text: str,
        kind: str = "fact",
        keys: Iterable[str] | None = None,
        payload: dict[str, Any] | None = None,
        scope: str = "global",
        confidence: float = 0.8,
        source: str = "user",
        embed: bool = True,
    ) -> SaveOutcome:
        """Write a memory, resolving conflicts with what is already known.

        * identical text  -> reinforce (confidence up, usage up), nothing duplicated
        * same kind+keys  -> supersede the older record (kept as history)
        * a correction    -> supersedes prior beliefs with overlapping keys/subject
        """
        text = str(text or "").strip()
        if not text:
            raise MemoryError_("memory text must not be empty")
        kind = (kind or "fact").strip().lower()
        if kind not in KINDS:
            raise MemoryError_(f"unknown memory kind {kind!r}; expected one of {', '.join(KINDS)}")
        keys = [str(k).strip().lower() for k in (keys or []) if str(k).strip()]
        payload = dict(payload or {})
        scope = (scope or "global").strip() or "global"
        confidence = max(0.0, min(1.0, float(confidence)))
        normalized = normalize_text(text)

        with self._lock:
            existing = self.list(include_inactive=False, limit=5000)
            same_scope = [r for r in existing if r.scope == scope]
            key_set = set(keys)

            for record in same_scope:
                if record.normalized == normalized:
                    record.confidence = min(1.0, max(record.confidence, confidence) + 0.05)
                    record.uses += 1
                    record.updated_at = utcnow()
                    if keys:
                        record.keys = sorted(set(record.keys) | key_set)
                    if payload:
                        record.payload = {**record.payload, **payload}
                    self._update_record(record)
                    self._log(record.id, "reinforced", text)
                    self._conn.commit()
                    return SaveOutcome(action="reinforced", record=record, note="already known")

            conflicts: list[MemoryRecord] = []
            for record in same_scope:
                if record.id in {c.id for c in conflicts}:
                    continue
                overlapping_keys = bool(key_set & set(record.keys))
                similar = _similarity(record.text, text) >= 0.55
                if kind == "correction":
                    # A correction overrides the same subject or a near-restatement.
                    if overlapping_keys or similar:
                        conflicts.append(record)
                elif record.kind == kind and (overlapping_keys or similar):
                    conflicts.append(record)

            vector = self._embed([text]) if embed else []
            record = MemoryRecord(
                id=new_id(),
                kind=kind,
                text=text,
                keys=sorted(key_set),
                payload=payload,
                scope=scope,
                confidence=confidence,
                source=source,
            )
            with self._conn:
                self._conn.execute(
                    f"INSERT INTO memories ({_COLUMNS}, embedding) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (*record.to_row(), json.dumps(vector[0]) if vector else None),
                )
                for old in conflicts:
                    old.active = False
                    old.superseded_by = record.id
                    old.updated_at = utcnow()
                    self._update_record(old)
                    self._log(old.id, "superseded", f"replaced by {record.id}: {text}")
                self._log(record.id, "created", text)

            note = ""
            if conflicts:
                note = f"superseded {len(conflicts)} earlier memory(ies)"
            return SaveOutcome(
                action="superseded" if conflicts else "created",
                record=record,
                superseded=conflicts,
                note=note,
            )

    def _update_record(self, record: MemoryRecord) -> None:
        self._conn.execute(
            "UPDATE memories SET kind=?, text=?, keys=?, payload=?, scope=?, confidence=?, "
            "source=?, created_at=?, updated_at=?, uses=?, last_used_at=?, superseded_by=?, "
            "active=? WHERE id=?",
            (
                record.kind,
                record.text,
                json.dumps(record.keys),
                json.dumps(record.payload),
                record.scope,
                float(record.confidence),
                record.source,
                record.created_at,
                record.updated_at,
                int(record.uses),
                record.last_used_at,
                record.superseded_by,
                1 if record.active else 0,
                record.id,
            ),
        )

    def edit(
        self,
        memory_id: str,
        text: str | None = None,
        payload: dict[str, Any] | None = None,
        keys: Sequence[str] | None = None,
        confidence: float | None = None,
    ) -> MemoryRecord:
        record = self.get(memory_id)
        if record is None:
            raise MemoryError_(f"no memory with id {memory_id!r}")
        with self._lock:
            if text is not None:
                record.text = text.strip()
            if payload is not None:
                record.payload = dict(payload)
            if keys is not None:
                record.keys = sorted({str(k).strip().lower() for k in keys if str(k).strip()})
            if confidence is not None:
                record.confidence = max(0.0, min(1.0, float(confidence)))
            record.updated_at = utcnow()
            self._update_record(record)
            self._log(record.id, "edited")
            self._conn.commit()
        return record

    def forget(self, memory_id: str, hard: bool = False) -> bool:
        """Deactivate (default) or delete a memory. Returns True if it existed."""
        record = self.get(memory_id)
        if record is None:
            return False
        with self._lock:
            if hard:
                self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                self._log(None, "deleted", memory_id)
            else:
                record.active = False
                record.updated_at = utcnow()
                self._update_record(record)
                self._log(memory_id, "forgotten")
            self._conn.commit()
        return True

    def clear(self, hard: bool = False) -> int:
        with self._lock:
            count = self.count(active_only=False)
            if hard:
                self._conn.execute("DELETE FROM memories")
            else:
                self._conn.execute("UPDATE memories SET active = 0, updated_at = ?", (utcnow(),))
            self._log(None, "cleared", "hard" if hard else "soft")
            self._conn.commit()
        return count

    def mark_used(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        if not ids:
            return
        now = utcnow()
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET uses = uses + 1, last_used_at = ? WHERE id = ?",
                [(now, i) for i in ids],
            )
            self._conn.commit()

    # -- retrieval --------------------------------------------------------
    def embeddings(self) -> dict[str, list[float]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding FROM memories WHERE active = 1 AND embedding IS NOT NULL"
            ).fetchall()
        out: dict[str, list[float]] = {}
        for row in rows:
            try:
                out[row["id"]] = json.loads(row["embedding"])
            except (TypeError, ValueError):
                continue
        return out

    def retrieve(
        self,
        query: str,
        context_keys: Iterable[str] | None = None,
        scope: str | None = None,
        limit: int = 6,
        kind: str | None = None,
        touch: bool = False,
    ) -> list[Retrieved]:
        """Relevance-ranked memories for this turn's question and loaded data."""
        records = self.active_records()
        if kind:
            records = [r for r in records if r.kind == kind]
        if not records:
            return []
        results = self.retriever.rank(
            records,
            query or "",
            context_keys=set(context_keys or ()),
            scope=scope,
            limit=limit,
            embeddings=self.embeddings() if self.retriever.embedder else None,
        )
        if touch and results:
            self.mark_used([r.record.id for r in results])
        return results

    # -- import / export --------------------------------------------------
    def export(self, include_inactive: bool = True) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.list(include_inactive=include_inactive, limit=10000)]

    def import_records(self, rows: Sequence[dict[str, Any]]) -> int:
        imported = 0
        for row in rows:
            if not row.get("text"):
                continue
            self.save(
                text=row["text"],
                kind=row.get("kind", "fact"),
                keys=row.get("keys") or [],
                payload=row.get("payload") or {},
                scope=row.get("scope", "global"),
                confidence=float(row.get("confidence", 0.8)),
                source=row.get("source", "import"),
                embed=False,
            )
            imported += 1
        return imported
