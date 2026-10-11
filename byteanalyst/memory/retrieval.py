"""Deciding which memories are relevant *right now*.

Pure lexical search misses the most important memory type: "the column
`status` means ..." is relevant whenever a dataset containing `status` is loaded,
even if the question never says "status". So retrieval blends three signals:

1. **Lexical overlap** — IDF-weighted token overlap between the question and the
   memory text (a BM25-flavoured score, computed over the small in-memory corpus).
2. **Context key match** — overlap between the memory's declared keys and the
   tables/columns actually loaded in this session.
3. **Authoritative-kind prior** — corrections, filters and definitions outrank
   notes, and are weighted by confidence, recency and past usefulness.

An optional embedder can be supplied to blend in cosine similarity; the default
path is fully offline and deterministic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from .models import MemoryRecord

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "how", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "so", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your", "do", "does", "did", "can", "could",
    "should", "would", "please", "show", "give", "tell", "about", "us",
}

_WORD_RE = re.compile(r"[a-z0-9_]+")

KIND_WEIGHT = {
    "correction": 1.25,
    "filter": 1.20,
    "definition": 1.12,
    "preference": 1.10,
    "fact": 1.00,
    "note": 0.85,
}

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with a very light plural stem and stopword removal."""
    tokens: list[str] = []
    for match in _WORD_RE.finditer(str(text or "").lower()):
        token = match.group(0)
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token in STOPWORDS or len(token) < 2:
            continue
        tokens.append(token)
    return tokens


@dataclass
class Retrieved:
    record: MemoryRecord
    score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 4), "reasons": self.reasons, **self.record.to_dict()}


def _recency_factor(iso_time: str | None, half_life_days: float = 45.0) -> float:
    if not iso_time:
        return 1.0
    try:
        then = datetime.fromisoformat(iso_time)
    except ValueError:
        return 1.0
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0)
    return 0.85 + 0.15 * math.exp(-age_days / half_life_days)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class MemoryRetriever:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder

    # -- idf ---------------------------------------------------------------
    @staticmethod
    def _idf(records: Iterable[MemoryRecord]) -> dict[str, float]:
        records = list(records)
        total = max(1, len(records))
        df: dict[str, int] = {}
        for record in records:
            for token in set(tokenize(record.text)):
                df[token] = df.get(token, 0) + 1
        return {token: math.log(1 + total / (1 + count)) for token, count in df.items()}

    # -- scoring -----------------------------------------------------------
    def score(
        self,
        record: MemoryRecord,
        query_tokens: Sequence[str],
        idf: dict[str, float],
        context_keys: set[str],
        scope: str | None,
        raw_query: str,
        embeddings: dict[str, list[float]] | None = None,
        query_vector: list[float] | None = None,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []

        memory_tokens = set(tokenize(record.text))
        overlap = [t for t in query_tokens if t in memory_tokens]
        default_idf = math.log(1 + max(1, len(idf)) / 2) if idf else 1.0
        lexical = sum(idf.get(t, default_idf) for t in overlap)
        if lexical:
            lexical_norm = lexical / (1.0 + lexical)
            reasons.append("lexical: " + ", ".join(sorted(set(overlap))[:6]))
        else:
            lexical_norm = 0.0

        # Context keys: dataset/column names loaded in this session.
        memory_keys = {k.lower() for k in record.keys}
        key_hits = memory_keys & {k.lower() for k in context_keys}
        key_score = 0.0
        if memory_keys:
            key_score = len(key_hits) / len(memory_keys)
            if key_hits:
                reasons.append("keys loaded: " + ", ".join(sorted(key_hits)[:6]))

        # Verbatim mention of a memory key in the question — strong evidence.
        phrase_bonus = 0.0
        lowered_query = (raw_query or "").lower()
        for key in memory_keys:
            if key and len(key) > 2 and key in lowered_query:
                phrase_bonus += 0.25
                reasons.append(f"question mentions '{key}'")
        phrase_bonus = min(phrase_bonus, 0.6)

        # Scope: a memory about a loaded dataset is more relevant than a global one.
        scope_factor = 1.0
        if record.scope and record.scope != "global":
            if scope and record.scope.lower() == str(scope).lower():
                scope_factor = 1.15
                reasons.append(f"scoped to {record.scope}")
            elif context_keys and record.scope.lower() in {k.lower() for k in context_keys}:
                scope_factor = 1.08

        base = 0.60 * lexical_norm + 0.40 * key_score + phrase_bonus

        # Semantic similarity is computed before the relevance gate so that a memory
        # with no lexical overlap at all can still be recalled ("churn" vs
        # "attrition"), which is the whole point of the optional embedder.
        vector_sim = 0.0
        if self.embedder is not None and embeddings and query_vector:
            vector = embeddings.get(record.id)
            if vector:
                vector_sim = _cosine(query_vector, vector)
                if vector_sim > 0.35:
                    reasons.append(f"semantic {vector_sim:.2f}")
        if vector_sim > 0.0:
            base = 0.55 * base + 0.45 * max(0.0, vector_sim)

        if base <= 0.0:
            return 0.0, reasons

        confidence_factor = 0.65 + 0.35 * max(0.0, min(1.0, record.confidence))
        usage_factor = 1.0 + 0.04 * math.log1p(max(0, record.uses))
        score = (
            base
            * KIND_WEIGHT.get(record.kind, 1.0)
            * confidence_factor
            * usage_factor
            * _recency_factor(record.updated_at)
            * scope_factor
        )
        return score, reasons

    # -- ranking -----------------------------------------------------------
    def rank(
        self,
        records: Sequence[MemoryRecord],
        query: str,
        context_keys: set[str] | None = None,
        scope: str | None = None,
        limit: int = 6,
        min_score: float = 0.02,
        embeddings: dict[str, list[float]] | None = None,
    ) -> list[Retrieved]:
        context_keys = context_keys or set()
        query_tokens = tokenize(query)
        idf = self._idf(records)

        query_vector = None
        if self.embedder is not None and query.strip():
            try:
                vectors = self.embedder([query])
                if vectors:
                    query_vector = list(vectors[0])
            except Exception:  # pragma: no cover - embedding provider failures are non-fatal
                query_vector = None

        ranked: list[Retrieved] = []
        for record in records:
            score, reasons = self.score(
                record,
                query_tokens,
                idf,
                context_keys,
                scope,
                query,
                embeddings,
                query_vector,
            )
            if score >= min_score:
                ranked.append(Retrieved(record=record, score=score, reasons=reasons))
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:limit]
