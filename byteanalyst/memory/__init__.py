"""Persistent memory.

Design in one paragraph: memory is a small, *typed*, *inspectable* store of
durable facts about the user's data and preferences — not a chat log and not a
data dump. Each record has a kind (definition / preference / correction / filter
/ fact / note), the keys it applies to (dataset or column names), an optional
structured payload (for example a SQL predicate), a confidence, and provenance.
Records are retrieved per turn by a hybrid lexical + key-match scorer, injected
into the system prompt as *binding* context, and written back when the user
states or corrects something durable. Conflicts supersede rather than overwrite,
so history is auditable and nothing is silently lost.
"""

from .models import KINDS, MemoryRecord, SaveOutcome
from .store import MemoryStore

__all__ = ["KINDS", "MemoryRecord", "SaveOutcome", "MemoryStore"]
