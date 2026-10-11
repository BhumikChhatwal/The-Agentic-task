"""Persistent memory: what is kept, how conflicts resolve, and how it is retrieved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from byteanalyst.errors import MemoryError_
from byteanalyst.memory import MemoryStore
from byteanalyst.memory.retrieval import MemoryRetriever, tokenize


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def test_save_and_list(memory: MemoryStore) -> None:
    outcome = memory.save(
        "Active customers exclude cancelled accounts", kind="filter",
        keys=["status"], payload={"sql": "status <> 'cancelled'"}, confidence=0.95,
    )
    assert outcome.action == "created"
    assert outcome.record.id
    records = memory.list()
    assert len(records) == 1
    assert records[0].payload["sql"] == "status <> 'cancelled'"
    assert memory.count() == 1


def test_identical_text_is_reinforced_not_duplicated(memory: MemoryStore) -> None:
    first = memory.save("Revenue is recorded net of tax", kind="definition", keys=["revenue"])
    second = memory.save("revenue   is recorded NET of tax.", kind="definition", keys=["revenue"])
    assert second.action == "reinforced"
    assert second.record.id == first.record.id
    assert memory.count() == 1
    assert second.record.confidence > first.record.confidence


def test_same_kind_and_keys_supersedes_the_older_record(memory: MemoryStore) -> None:
    old = memory.save("Active means status is active", kind="filter", keys=["status"])
    new = memory.save("Active means status is not cancelled", kind="filter", keys=["status"])
    assert new.action == "superseded"
    assert [c.id for c in new.superseded] == [old.record.id]
    assert memory.get(old.record.id).active is False
    assert memory.get(old.record.id).superseded_by == new.record.id
    assert memory.count(active_only=True) == 1
    assert memory.count(active_only=False) == 2  # history is kept


def test_correction_overrides_an_unrelated_kind(memory: MemoryStore) -> None:
    memory.save("Churn is measured monthly", kind="fact", keys=["churn"])
    correction = memory.save("Churn is measured weekly", kind="correction", keys=["churn"])
    assert correction.action == "superseded"
    assert memory.count(active_only=True) == 1
    assert memory.list()[0].kind == "correction"


def test_memories_about_different_subjects_coexist(memory: MemoryStore) -> None:
    memory.save("Revenue is gross", kind="definition", keys=["revenue"])
    memory.save("Margin is after discounts", kind="definition", keys=["margin"])
    assert memory.count() == 2
    assert len(memory.list()) == 2


def test_scoped_memories_do_not_collide(memory: MemoryStore) -> None:
    a = memory.save("Total means sum of revenue", kind="definition", keys=["revenue"], scope="orders")
    b = memory.save("Total means sum of amount", kind="definition", keys=["amount"], scope="invoices")
    assert memory.count() == 2
    assert {r.scope for r in memory.list()} == {"orders", "invoices"}
    assert a.record.id != b.record.id


def test_write_validation(memory: MemoryStore) -> None:
    with pytest.raises(MemoryError_):
        memory.save("   ")
    with pytest.raises(MemoryError_):
        memory.save("something", kind="gossip")
    assert memory.count() == 0


def test_confidence_is_clamped(memory: MemoryStore) -> None:
    record = memory.save("Note", confidence=5.0).record
    assert record.confidence == 1.0
    assert memory.save("Other", confidence=-1.0).record.confidence == 0.0


# ---------------------------------------------------------------------------
# editing and control
# ---------------------------------------------------------------------------


def test_edit_forget_and_clear(memory: MemoryStore) -> None:
    record = memory.save("Cancelled means status = cancelled", kind="definition", keys=["status"]).record
    edited = memory.edit(record.id, text="Cancelled means status = 'cancelled' or 'void'", confidence=1.0)
    assert "void" in edited.text
    assert edited.confidence == 1.0

    assert memory.forget(record.id) is True
    assert memory.count(active_only=True) == 0
    assert memory.count(active_only=False) == 1
    assert memory.get(record.id).active is False
    assert memory.forget("does-not-exist") is False


def test_clear_and_history(memory: MemoryStore) -> None:
    memory.save("A", kind="fact")
    memory.save("B", kind="fact", keys=["x"])
    assert memory.clear(hard=False) == 2
    assert memory.count(active_only=True) == 0
    actions = [row["action"] for row in memory.history()]
    assert "cleared" in actions and "created" in actions


def test_stats_and_export_import(memory: MemoryStore) -> None:
    memory.save("Revenue is net", kind="definition", keys=["revenue"])
    memory.save("Weekly summaries please", kind="preference")
    stats = memory.stats()
    assert stats["active"] == 2
    assert stats["by_kind"] == {"definition": 1, "preference": 1}

    dumped = json.loads(json.dumps(memory.export()))
    other = MemoryStore(":memory:")
    try:
        assert other.import_records(dumped) == 2
        assert other.count() == 2
    finally:
        other.close()


def test_usage_counters_track_surfacing(memory: MemoryStore) -> None:
    record = memory.save("Active excludes cancelled", kind="filter", keys=["status"]).record
    memory.mark_used([record.id])
    memory.mark_used([record.id])
    assert memory.get(record.id).uses == 2
    assert memory.get(record.id).last_used_at is not None


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------


def test_retrieval_prefers_memories_whose_columns_are_loaded(memory: MemoryStore) -> None:
    memory.save("Active customers exclude cancelled accounts", kind="filter",
                keys=["status"], payload={"sql": "status <> 'cancelled'"})
    memory.save("Currency is EUR", kind="fact", keys=["currency"])

    results = memory.retrieve("how many active customers", context_keys={"orders", "status", "revenue"})
    assert results
    assert results[0].record.keys == ["status"]
    assert "keys loaded: status" in results[0].reasons


def test_retrieval_ignores_irrelevant_memories(memory: MemoryStore) -> None:
    memory.save("Currency is EUR", kind="fact", keys=["currency"])
    results = memory.retrieve("average delivery time", context_keys={"orders", "delivery_days"})
    assert results == []


def test_retrieval_ranks_corrections_above_notes(memory: MemoryStore) -> None:
    memory.save("Revenue might be gross", kind="note", keys=["revenue"])
    results = memory.retrieve("how is revenue measured", context_keys={"revenue"})
    assert results[0].record.kind == "note"

    memory.save("Revenue is actually net of refunds", kind="correction", keys=["revenue"])
    results = memory.retrieve("how is revenue measured", context_keys={"revenue"})
    assert results[0].record.kind == "correction"


def test_superseded_memories_are_not_retrieved(memory: MemoryStore) -> None:
    old = memory.save("Active means status = active", kind="filter", keys=["status"]).record
    memory.save("Active excludes cancelled", kind="filter", keys=["status"])
    ids = {r.record.id for r in memory.retrieve("active customers", context_keys={"status"})}
    assert old.id not in ids


def test_retrieval_can_filter_by_kind_and_touch_usage(memory: MemoryStore) -> None:
    memory.save("Active excludes cancelled", kind="filter", keys=["status"])
    memory.save("Weekly summary please", kind="preference", keys=["status"])
    only_filter = memory.retrieve("status", kind="filter", context_keys={"status"})
    assert len(only_filter) == 1

    touched = memory.retrieve("status", context_keys={"status"}, touch=True)
    assert any(memory.get(r.record.id).uses >= 1 for r in touched)


def test_scope_boosts_matching_dataset(memory: MemoryStore) -> None:
    memory.save("Total means net revenue", kind="definition", keys=["total"], scope="orders")
    results = memory.retrieve("what does total mean", context_keys={"orders"}, scope="orders")
    assert any("scoped to orders" in reason for reason in results[0].reasons)


def test_tokenizer_and_empty_corpus(memory: MemoryStore) -> None:
    assert "customer" in tokenize("customers")
    assert "the" not in tokenize("the customers and a")
    assert memory.retrieve("anything") == []


def test_optional_embedding_hook_blends_semantic_similarity() -> None:
    """A pluggable embedder can add semantic recall without changing anything else."""
    # A stand-in for a real embedding model: "churn" and "attrition" land in the
    # same bucket, so a query using one word can recall a memory using the other.
    buckets = {"revenue": 0, "sales": 0, "churn": 1, "attrition": 1}

    def embedder(texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = [0.0, 0.0]
            for word, index in buckets.items():
                if word in lowered:
                    vector[index] = 1.0
            vectors.append(vector)
        return vectors

    store = MemoryStore(":memory:", embedder=embedder)
    try:
        store.save("Churn is measured monthly", kind="fact", keys=["churn"])
        # No lexical overlap at all, so only the embedding can surface this.
        results = store.retrieve("attrition rate", context_keys=set())
        assert results
        assert results[0].record.text.startswith("Churn")
        assert any("semantic" in reason for reason in results[0].reasons)
    finally:
        store.close()


def test_retriever_is_deterministic() -> None:
    retriever = MemoryRetriever()
    assert retriever.embedder is None


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_memory_survives_a_new_store_instance(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = MemoryStore(path)
    first.save("Active customers exclude cancelled accounts", kind="filter",
               keys=["status"], payload={"sql": "status <> 'cancelled'"})
    first.close()

    second = MemoryStore(path)
    try:
        assert second.count() == 1
        assert second.retrieve("active customers", context_keys={"status"})[0].record.kind == "filter"
        # And the schema/history came across too.
        assert any(row["action"] == "created" for row in second.history())
    finally:
        second.close()


def test_conflict_resolution_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    store.save("Active means status = active", kind="filter", keys=["status"])
    store.close()

    reopened = MemoryStore(path)
    try:
        outcome = reopened.save("Active excludes cancelled", kind="filter", keys=["status"])
        assert outcome.action == "superseded"
        assert reopened.count(active_only=True) == 1
        assert reopened.count(active_only=False) == 2
    finally:
        reopened.close()
