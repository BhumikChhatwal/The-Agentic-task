"""The agent loop.

Both drivers are covered: the deterministic rule-based planner (no API key) and
the model-driven loop (via a scripted client, so the real dispatch, result
rendering, verification and memory write-back code paths all run).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from byteanalyst.agent import Agent, find_unsupported_numbers
from byteanalyst.agent.core import _regex_memory_candidates
from byteanalyst.config import Config
from byteanalyst.datasets import DatasetRegistry
from byteanalyst.errors import ByteAnalystError, ConfigError
from byteanalyst.llm.offline import RuleBasedPlanner
from byteanalyst.llm.scripted import ScriptedClient
from byteanalyst.memory import MemoryStore
from byteanalyst.tools.registry import ToolExecution


@pytest.fixture
def offline_agent(config: Config, registry: DatasetRegistry, memory: MemoryStore) -> Agent:
    return Agent(config, registry, memory=memory)


def llm_agent(config: Config, registry: DatasetRegistry, turns, memory=None) -> Agent:
    return Agent(config, registry, memory=memory, client=ScriptedClient(turns))


# ---------------------------------------------------------------------------
# offline (rule-based) driver
# ---------------------------------------------------------------------------


def test_offline_run_uses_tools_and_stays_grounded(offline_agent: Agent) -> None:
    answer = offline_agent.ask("total revenue by region")
    assert answer.ok
    assert answer.tools_used == ["run_sql"]
    assert answer.grounded
    assert answer.provider == "offline"
    assert answer.model == "offline-rule-based-planner"
    assert "Here is what the data shows" in answer.answer
    assert "run_sql" not in answer.answer or "Query returned" in answer.answer
    # The numbers in the answer must be traceable to the tool output.
    totals = offline_agent.registry.get("orders").frame.groupby("region")["revenue"].sum()
    assert any(f"{total:.0f}"[:4] in answer.answer for total in totals)


def test_offline_question_routing(offline_agent: Agent) -> None:
    cases = {
        "what's in this data": "profile_dataset",
        "profile the orders table": "profile_dataset",
        "how many orders are there": "run_sql",
        "are there any outliers in revenue": "detect_anomalies",
        "how does revenue trend over time": "analyze_time_series",
        "plot the distribution of revenue": "visualize",
        "is revenue different between North and South": "statistical_test",
    }
    for question, expected in cases.items():
        answer = offline_agent.ask(question)
        assert expected in answer.tools_used, f"{question!r} -> {answer.tools_used}"


def test_offline_bare_count_is_a_total_not_a_breakdown(offline_agent: Agent) -> None:
    answer = offline_agent.ask("how many orders are there")
    query = answer.tool_executions[0].arguments["query"]
    assert "group by" not in query.lower()
    assert "count(*)" in query.lower()


def test_offline_answer_reports_tool_failures() -> None:
    planner = RuleBasedPlanner(DatasetRegistry())
    execution = ToolExecution(
        name="run_sql",
        arguments={"query": "select nope"},
        result=__import__("byteanalyst.formatting", fromlist=["ToolResult"]).ToolResult.failure(
            "Binder Error: column nope not found"
        ),
        duration_ms=3,
    )
    text = planner.compose("anything", [execution])
    assert "could not answer" in text or "failed" in text
    assert "nope" in text
    assert "will not guess" in text or "What I tried" in text


def test_offline_memory_definition_changes_the_sql(config, registry, memory) -> None:
    memory.save(
        "Active customers exclude cancelled accounts",
        kind="filter", keys=["status"], payload={"sql": "status <> 'cancelled'"},
    )
    agent = Agent(config, registry, memory=memory)
    answer = agent.ask("how many active customers are there by region")
    query = answer.tool_executions[0].arguments["query"]
    assert "status <> 'cancelled'" in query
    assert answer.memories_used


def test_memory_about_another_table_is_not_applied(config, registry, memory) -> None:
    """A filter remembered for one dataset must not break a query on another."""
    memory.save(
        "Invoices exclude voided rows", kind="filter", keys=["invoice_status"],
        payload={"sql": "invoice_status <> 'void'"},
    )
    agent = Agent(config, registry, memory=memory)
    answer = agent.ask("how many orders are there by region")
    assert answer.ok
    query = answer.tool_executions[0].arguments["query"]
    assert "invoice_status" not in query


def test_offline_writes_memory_from_a_stated_definition(config, registry, memory) -> None:
    agent = Agent(config, registry, memory=memory)
    answer = agent.ask("remember that active customers exclude cancelled accounts")
    assert answer.memories_saved
    saved = answer.memories_saved[0]
    assert saved["kind"] == "filter"
    assert "active customers" in saved["text"]
    assert "remember that" not in saved["text"]  # conversational cue stripped
    assert memory.count() == 1


def test_offline_correction_supersedes(config, registry, memory) -> None:
    memory.save("Active means status = active", kind="filter", keys=["status"])
    agent = Agent(config, registry, memory=memory)
    answer = agent.ask("actually active means status is not cancelled")
    assert any(item["kind"] == "correction" for item in answer.memories_saved)
    assert memory.count(active_only=True) == 1


def test_regex_memory_extraction_is_conservative(registry: DatasetRegistry) -> None:
    assert _regex_memory_candidates("what is the revenue?", registry, None) == []
    assert _regex_memory_candidates("hi", registry, None) == []
    filters = _regex_memory_candidates("active customers exclude cancelled accounts", registry, None)
    assert len(filters) == 1
    assert filters[0]["kind"] == "filter"
    # One sentence must not become two near-duplicate memories.
    both = _regex_memory_candidates("remember that active customers exclude cancelled accounts", registry, None)
    assert len(both) == 1


# ---------------------------------------------------------------------------
# model-driven driver (scripted)
# ---------------------------------------------------------------------------


def test_llm_loop_dispatches_tools_then_answers(config, registry) -> None:
    agent = llm_agent(config, registry, [
        {"tool": "run_sql", "arguments": {"query": "select count(*) as n from orders"}},
        {"content": "There are 240 orders.\n\nEvidence:\n- run_sql: n = 240"},
    ])
    answer = agent.ask("how many orders are there?")
    assert answer.ok
    assert answer.tools_used == ["run_sql"]
    assert answer.iterations == 2
    assert answer.grounded
    assert "240" in answer.answer


def test_llm_loop_sends_the_tool_result_back_to_the_model(config, registry) -> None:
    client = ScriptedClient([
        {"tool": "profile_dataset", "arguments": {"dataset": "orders"}},
        {"content": "The table has 9 columns.\n\nEvidence:\n- profile_dataset: 9 columns"},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    agent.ask("what columns are there?")
    second_call_messages = client.calls[1]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    assert tool_messages and "revenue" in tool_messages[0].content
    assert any(m.role == "assistant" and m.tool_calls for m in second_call_messages)


def test_llm_loop_escalates_unverified_numbers_once(config, registry) -> None:
    client = ScriptedClient([
        {"tool": "run_sql", "arguments": {"query": "select count(*) as n from orders"}},
        {"content": "There are 240 orders and total revenue is 987654."},
        {"content": "There are 240 orders."},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    answer = agent.ask("how many orders?")
    assert answer.grounded
    assert answer.iterations == 3
    assert "987654" not in answer.answer
    # The repair instruction must be visible in what the model was sent.
    repair_prompt = [m for m in client.calls[2] if m.role == "user"][-1].content
    assert "987654" in repair_prompt and "not produced by any" in repair_prompt


def test_llm_loop_reports_numbers_it_cannot_fix(config, registry) -> None:
    client = ScriptedClient([
        {"content": "Revenue was 555111 in 2023."},
        {"content": "Revenue was 555111."},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    answer = agent.ask("what was revenue?")
    assert not answer.grounded
    assert "555111" in answer.unsupported_numbers


def test_llm_loop_with_no_tools_flags_every_number(config, registry) -> None:
    # No tool ran, so every figure is unsupported even after the repair attempt.
    agent = llm_agent(config, registry, [{"content": "The answer is 42."}, {"content": "It is 42."}])
    answer = agent.ask("anything")
    assert answer.tool_executions == []
    assert "42" in answer.unsupported_numbers
    assert not answer.grounded


def test_llm_loop_stops_at_the_iteration_budget(config, registry) -> None:
    config.max_iterations = 2
    client = ScriptedClient([
        {"tool": "run_sql", "arguments": {"query": "select 1"}},
        {"tool": "run_sql", "arguments": {"query": "select 2"}},
        {"tool": "run_sql", "arguments": {"query": "select 3"}},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    answer = agent.ask("loop forever")
    assert answer.iterations == 2
    assert "limit" in answer.answer
    assert len(answer.tool_executions) == 2


def test_llm_loop_suppresses_identical_repeated_calls(config, registry) -> None:
    """A deterministic model repeating the same call must not re-run the tool."""
    client = ScriptedClient([
        {"tool": "run_sql", "arguments": {"query": "select count(*) as n from orders"}},
        {"tool": "run_sql", "arguments": {"query": "select count(*) as n from orders"}},
        {"content": "There are 240 orders.\n\nEvidence:\n- run_sql: n = 240"},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    answer = agent.ask("how many orders?")
    assert answer.ok
    # The identical second call is answered without executing the tool again.
    assert len(answer.tool_executions) == 1
    assert answer.iterations == 3
    # The model is told why, in the tool result of the repeated call.
    followup_tool = [m for m in client.calls[2] if m.role == "tool"]
    assert followup_tool and "already called" in followup_tool[-1].content


def test_llm_loop_reprompts_after_an_empty_turn(config, registry) -> None:
    """An empty assistant turn is nudged, not reported as the answer."""
    client = ScriptedClient([
        {"content": ""},
        {"tool": "run_sql", "arguments": {"query": "select count(*) as n from orders"}},
        {"content": "There are 240 orders.\n\nEvidence:\n- run_sql: n = 240"},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    answer = agent.ask("how many orders?")
    assert answer.ok
    assert "240" in answer.answer
    nudge = [m for m in client.calls[1] if m.role == "user"][-1].content
    assert "empty" in nudge.lower()


def test_llm_loop_reports_persistent_empty_turns(config, registry) -> None:
    """If the model keeps returning silence, say so instead of answering blank."""
    client = ScriptedClient([{"content": ""}, {"content": ""}])
    agent = Agent(config, registry, memory=None, client=client)
    answer = agent.ask("how many orders?")
    assert "empty response" in answer.answer


def test_llm_loop_rescues_text_shaped_tool_calls(config, registry) -> None:
    """Small models write calls like ``run_sql: SELECT ...`` as the body."""
    client = ScriptedClient([
        {"content": "run_sql: select count(*) as n from orders"},
        {"content": "There are 240 orders.\n\nEvidence:\n- run_sql: n = 240"},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    answer = agent.ask("how many orders?")
    assert answer.ok
    assert answer.tools_used == ["run_sql"]
    assert answer.iterations == 2
    assert "240" in answer.answer


def test_llm_loop_rescues_braced_json_tool_calls(config, registry) -> None:
    client = ScriptedClient([
        {"content": 'run_sql {"query": "select count(*) as n from orders"}'},
        {"content": "There are 240 orders.\n\nEvidence:\n- run_sql: n = 240"},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    answer = agent.ask("how many orders?")
    assert answer.tools_used == ["run_sql"]
    assert answer.tool_executions[0].arguments == {"query": "select count(*) as n from orders"}
    assert "240" in answer.answer


def test_llm_loop_rescues_bare_sql(config, registry) -> None:
    client = ScriptedClient([
        {"content": "select count(*) as n from orders"},
        {"content": "There are 240 orders.\n\nEvidence:\n- run_sql: n = 240"},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    answer = agent.ask("how many orders?")
    assert answer.tools_used == ["run_sql"]
    assert "240" in answer.answer


def test_llm_loop_does_not_rescue_prose_mentioning_tools(config, registry) -> None:
    client = ScriptedClient([
        {"content": "run_sql is the tool you would use here: it runs queries."},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    answer = agent.ask("how many orders?")
    assert answer.tool_executions == []


# ---------------------------------------------------------------------------
# tool allowlist (smaller toolboxes for local models)
# ---------------------------------------------------------------------------


def test_tool_allowlist_scopes_registry_specs_and_prompt(config, registry) -> None:
    config.tool_allowlist = ("run_sql", "profile_dataset")
    client = ScriptedClient([
        {"tool": "run_sql", "arguments": {"query": "select count(*) as n from orders"}},
        {"content": "There are 240 orders.\n\nEvidence:\n- run_sql: n = 240"},
    ])
    agent = Agent(config, registry, memory=None, client=client)
    assert agent.tools.names() == ["run_sql", "profile_dataset"]
    spec_names = [spec["function"]["name"] for spec in agent.tools.specs()]
    assert spec_names == ["run_sql", "profile_dataset"]
    answer = agent.ask("how many orders?")
    assert answer.ok and "240" in answer.answer
    system = [m for m in client.calls[0] if m.role == "system"][0].content
    assert "run_sql" in system and "profile_dataset" in system
    # Tools the model cannot call must not be advertised, not even in prose.
    assert "manage_memory" not in system
    assert "Memory policy" not in system
    assert "`recall`" not in system


def test_tool_allowlist_rejects_unknown_tools(config, registry) -> None:
    config.tool_allowlist = ("run_sql", "telepathy")
    with pytest.raises(ConfigError, match="telepathy"):
        Agent(config, registry, memory=None)


def test_llm_writes_memory_from_the_extraction_pass(config, registry, memory) -> None:
    client = ScriptedClient([
        {"tool": "run_sql", "arguments": {"query": "select count(*) as n from orders"}},
        {"content": "There are 240 orders, excluding cancelled ones per your rule.\n\nEvidence:\n- run_sql: 240"},
        {"content": json.dumps([
            {"text": "Active customers exclude cancelled accounts", "kind": "filter",
             "keys": ["status"], "payload": {"sql": "status <> 'cancelled'"}, "confidence": 1.0}
        ])},
    ])
    agent = Agent(config, registry, memory=memory, client=client)
    answer = agent.ask("how many active customers?")
    assert answer.memories_saved
    assert memory.count() == 1
    assert memory.list()[0].payload["sql"] == "status <> 'cancelled'"


def test_memory_extraction_failure_does_not_break_the_answer(config, registry, memory) -> None:
    client = ScriptedClient([
        {"tool": "run_sql", "arguments": {"query": "select count(*) as n from orders"}},
        {"content": "There are 240 orders.\n\nEvidence:\n- run_sql: 240"},
        # The script is exhausted here: the extraction call raises inside the client.
    ])
    agent = Agent(config, registry, memory=memory, client=client)
    answer = agent.ask("how many orders?")
    assert answer.ok
    assert answer.memories_saved == []


def test_memory_can_be_disabled(config, registry) -> None:
    config.memory_enabled = False
    agent = Agent(config, registry, memory=MemoryStore(":memory:"))
    assert agent.memory is None
    assert "remember" not in agent.tools.names()
    answer = agent.ask("how many orders are there by region")
    assert answer.ok and answer.memories_used == [] and answer.memories_saved == []


def test_memory_tools_are_absent_without_a_store(config, registry) -> None:
    agent = Agent(config, registry, memory=None)
    assert "remember" not in agent.tools.names()
    assert "recall" not in agent.tools.names()
    assert len(agent.tools.names()) == 7


def test_provider_failure_is_reported_not_raised(config, registry) -> None:
    config.provider = "openai_compatible"
    config.api_key = None
    config.base_url = "https://api.openai.com/v1"
    agent = Agent(config, registry, memory=None)
    answer = agent.ask("how many orders?")
    assert not answer.ok
    assert answer.error
    assert "offline" in answer.answer  # it suggests the keyless route


def test_empty_question_is_rejected(offline_agent: Agent) -> None:
    with pytest.raises(ByteAnalystError):
        offline_agent.ask("   ")


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def test_events_are_emitted_in_order(offline_agent: Agent) -> None:
    events: list[str] = []
    offline_agent.ask("total revenue by region", on_event=lambda event: events.append(event.type))
    assert events[0] == "start"
    assert "tool_call" in events and "tool_result" in events
    assert events[-1] == "answer"
    assert events.index("tool_call") < events.index("tool_result")


def test_a_broken_event_listener_cannot_break_the_run(offline_agent: Agent) -> None:
    def boom(event):
        raise RuntimeError("listener exploded")

    answer = offline_agent.ask("total revenue by region", on_event=boom)
    assert answer.ok


def test_answer_is_json_serialisable(offline_agent: Agent) -> None:
    answer = offline_agent.ask("how many orders are there")
    payload = json.dumps(answer.to_dict(), default=str)
    restored = json.loads(payload)
    assert restored["tools_used"] == ["run_sql"]
    assert restored["grounded"] is True


def test_history_is_kept_across_turns(offline_agent: Agent) -> None:
    offline_agent.ask("how many orders are there")
    offline_agent.ask("and by region?")
    assert len(offline_agent.history) == 4  # two user / assistant pairs
    offline_agent.reset()
    assert offline_agent.history == []


def test_from_paths_loads_files_and_opens_memory(tmp_path: Path) -> None:
    csv = tmp_path / "things.csv"
    csv.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
    config = Config.from_env(root=tmp_path, provider="offline")
    config.state_dir = tmp_path / "state"
    agent = Agent.from_paths([csv], config=config)
    try:
        assert agent.registry.get("things").row_count == 2
        assert agent.memory is not None
        assert agent.memory.path == str(config.memory_path)
        answer = agent.ask("profile the data")
        assert answer.ok
    finally:
        agent.close()


def test_from_paths_with_one_dataset_needs_no_name(tmp_path: Path) -> None:
    csv = tmp_path / "solo.csv"
    csv.write_text("a\n1\n2\n3\n", encoding="utf-8")
    agent = Agent.from_paths([csv], config=Config.from_env(root=tmp_path), use_memory=False)
    assert agent.memory is None
    assert agent.ask("how many rows are there").ok


def test_agent_context_manager_closes_memory(tmp_path: Path) -> None:
    csv = tmp_path / "solo.csv"
    csv.write_text("a\n1\n2\n3\n", encoding="utf-8")
    agent = Agent.from_paths([csv], config=Config.from_env(root=tmp_path))
    with agent:
        assert agent.ask("how many rows are there").ok
    with pytest.raises(Exception):
        agent.memory.count()


# ---------------------------------------------------------------------------
# grounding check unit tests
# ---------------------------------------------------------------------------


def _execution(summary: str, table=None) -> ToolExecution:
    from byteanalyst.formatting import ToolResult

    return ToolExecution(
        name="run_sql", arguments={}, result=ToolResult.success(summary, table=table), duration_ms=1
    )


def test_find_unsupported_numbers() -> None:
    executions = [_execution("Query returned 1,234 rows with total 56.7")]
    assert find_unsupported_numbers("We saw 1234 rows (56.7 total).", executions) == []
    assert find_unsupported_numbers("We saw 1234 rows and 99.9 average.", executions) == ["99.9"]
    # Rounding at the displayed precision is tolerated.
    assert find_unsupported_numbers("Total was 56.70.", executions) == []


def test_find_unsupported_numbers_ignores_years_and_ordinals() -> None:
    executions = [_execution("count 500")]
    assert find_unsupported_numbers("In 2024 there were 500 rows; step 3 of 4.", executions) == []


def test_find_unsupported_numbers_with_no_tools() -> None:
    assert find_unsupported_numbers("Answer is 17.", []) == ["17"]
