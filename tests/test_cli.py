"""The command line, driven through ``main()`` exactly as a user would."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from byteanalyst import __version__
from byteanalyst.cli import main

SAMPLES = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def state_dir(tmp_path: Path) -> str:
    return str(tmp_path / "state")


# ---------------------------------------------------------------------------
# profile / tools
# ---------------------------------------------------------------------------


def test_profile_json(capsys, tmp_path) -> None:
    csv = tmp_path / "d.csv"
    csv.write_text("a,b,note\n1,x,hello\n2,y,world\n", encoding="utf-8")
    code = main(["profile", str(csv), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["detail"]["rows"] == 2
    assert {row[0] for row in payload["table"]["rows"]} == {"a", "b", "note"}


def test_profile_renders_a_table(capsys, tmp_path) -> None:
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
    code = main(["profile", str(csv), "--no-color"])
    out = capsys.readouterr().out
    assert code == 0
    assert "column" in out and "numeric" in out


def test_profile_missing_file(capsys) -> None:
    code = main(["profile", "definitely-not-here.csv", "--no-color"])
    assert code == 2
    assert "not found" in capsys.readouterr().out


def test_profile_partial_columns(capsys, tmp_path) -> None:
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,x\n", encoding="utf-8")
    code = main(["profile", str(csv), "--columns", "a", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [row[0] for row in payload["table"]["rows"]] == ["a"]


def test_tools_command_lists_every_tool(capsys) -> None:
    code = main(["tools", "--json"])
    specs = json.loads(capsys.readouterr().out)
    assert code == 0
    names = {spec["function"]["name"] for spec in specs}
    assert names == {
        "profile_dataset", "run_sql", "run_python", "statistical_test",
        "detect_anomalies", "analyze_time_series", "visualize",
        "remember", "recall", "manage_memory",
    }
    for spec in specs:
        assert spec["function"]["description"]
        assert spec["function"]["parameters"]["type"] == "object"


def test_tools_human_readable(capsys) -> None:
    code = main(["tools", "--no-color"])
    out = capsys.readouterr().out
    assert code == 0
    assert "10 tools" in out
    assert "run_sql" in out and "arguments:" in out


def test_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


def test_memory_add_list_search_forget(capsys, state_dir) -> None:
    assert main([
        "memory", "add", "Active customers exclude cancelled accounts",
        "--kind", "filter", "--keys", "status",
        "--payload-sql", "status <> 'cancelled'", "--state-dir", state_dir, "--no-color",
    ]) == 0
    out = capsys.readouterr().out
    assert "created" in out
    memory_id = out.split("[")[1].split("]")[0]

    assert main(["memory", "list", "--state-dir", state_dir, "--no-color"]) == 0
    assert "exclude cancelled" in capsys.readouterr().out

    assert main(["memory", "search", "active customers", "--state-dir", state_dir, "--no-color"]) == 0
    search_out = capsys.readouterr().out
    assert "1 match" in search_out and "why:" in search_out

    assert main(["memory", "stats", "--state-dir", state_dir, "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["active"] == 1

    assert main(["memory", "forget", memory_id, "--state-dir", state_dir, "--no-color"]) == 0
    assert "forgot" in capsys.readouterr().out
    assert main(["memory", "list", "--state-dir", state_dir, "--no-color"]) == 0
    assert "(empty)" in capsys.readouterr().out


def test_memory_persists_across_cli_invocations(capsys, state_dir) -> None:
    main(["memory", "add", "Revenue is net of refunds", "--kind", "definition",
          "--keys", "revenue", "--state-dir", state_dir, "--no-color"])
    capsys.readouterr()
    main(["memory", "list", "--state-dir", state_dir, "--json"])
    records = json.loads(capsys.readouterr().out)
    assert len(records) == 1
    assert records[0]["text"] == "Revenue is net of refunds"


def test_memory_correction_supersedes_via_cli(capsys, state_dir) -> None:
    main(["memory", "add", "Active means status = active", "--kind", "filter",
          "--keys", "status", "--state-dir", state_dir, "--no-color"])
    capsys.readouterr()
    main(["memory", "add", "Active means status is not cancelled", "--kind", "filter",
          "--keys", "status", "--state-dir", state_dir, "--no-color"])
    out = capsys.readouterr().out
    assert "superseded" in out
    main(["memory", "stats", "--state-dir", state_dir, "--json"])
    stats = json.loads(capsys.readouterr().out)
    assert stats["active"] == 1 and stats["superseded"] == 1


def test_memory_clear_requires_confirmation(capsys, state_dir) -> None:
    main(["memory", "add", "something durable here", "--state-dir", state_dir, "--no-color"])
    capsys.readouterr()
    assert main(["memory", "clear", "--state-dir", state_dir, "--no-color"]) == 2
    assert "refusing" in capsys.readouterr().out
    assert main(["memory", "clear", "--yes", "--state-dir", state_dir, "--no-color"]) == 0
    assert "deactivated" in capsys.readouterr().out


def test_memory_history_and_export(capsys, state_dir) -> None:
    main(["memory", "add", "Weekly summary please", "--kind", "preference",
          "--state-dir", state_dir, "--no-color"])
    capsys.readouterr()
    assert main(["memory", "history", "--state-dir", state_dir, "--no-color"]) == 0
    assert "created" in capsys.readouterr().out

    assert main(["memory", "export", "--state-dir", state_dir]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported[0]["kind"] == "preference"


def test_memory_import_round_trip(capsys, tmp_path, state_dir) -> None:
    payload = [{"text": "Imported fact about revenue", "kind": "definition", "keys": ["revenue"]}]
    path = tmp_path / "mem.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["memory", "import", str(path), "--state-dir", state_dir, "--no-color"]) == 0
    assert "imported 1" in capsys.readouterr().out


def test_memory_add_without_text_errors(capsys, state_dir) -> None:
    assert main(["memory", "add", "--state-dir", state_dir, "--no-color"]) == 2
    assert "needs text" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_offline_json(capsys, state_dir) -> None:
    code = main([
        "ask", "total revenue by region",
        "--data", str(SAMPLES / "ecommerce_orders.csv"),
        "--provider", "offline", "--no-memory", "--state-dir", state_dir,
        "--no-color", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["tools_used"] == ["run_sql"]
    assert payload["grounded"] is True
    assert "south" in payload["answer"].lower()  # a group label from the SQL result
    assert payload["tool_executions"][0]["arguments"]["query"].lower().startswith("select")


def test_ask_offline_trace_output(capsys, state_dir) -> None:
    code = main([
        "ask", "how many orders are there",
        "--data", "ecommerce_orders.csv",           # bundled sample, resolved by name
        "--provider", "offline", "--state-dir", state_dir, "--no-color",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "run_sql" in out
    assert "grounded: yes" in out or "traced to a tool result" in out


def test_ask_trace_flag_is_accepted(capsys, state_dir) -> None:
    code = main([
        "ask", "how many orders are there", "--data", "ecommerce_orders.csv",
        "--provider", "offline", "--state-dir", state_dir, "--no-color", "--trace",
    ])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "run_sql" in out  # the flag is valid and the trace is shown

    # --json must still win: stdout stays machine readable.
    code = main([
        "ask", "how many orders are there", "--data", "ecommerce_orders.csv",
        "--provider", "offline", "--state-dir", state_dir, "--no-color", "--trace", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["tools_used"] == ["run_sql"]


def test_ask_resolves_bundled_sample_names(capsys, state_dir) -> None:
    code = main([
        "ask", "profile the data", "--data", "employees.csv",
        "--provider", "offline", "--state-dir", state_dir, "--no-color", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["tools_used"] == ["profile_dataset"]


def test_ask_with_time_series(capsys, state_dir) -> None:
    code = main([
        "ask", "how does revenue trend over time",
        "--data", "ecommerce_orders.csv",
        "--provider", "offline", "--state-dir", state_dir, "--no-color", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["tools_used"] == ["profile_dataset", "run_sql"]
    # The offline planner answers time-trend questions with a SQL aggregation plus
    # a profile, not the dedicated time-series tool. Verify it is grounded.
    assert payload["grounded"] is True
    assert payload["unsupported_numbers"] == []


def test_ask_without_data_is_a_clear_error(capsys, state_dir) -> None:
    code = main(["ask", "anything", "--provider", "offline", "--state-dir", state_dir, "--no-color"])
    assert code == 2
    assert "no data loaded" in capsys.readouterr().out


def test_ask_missing_file(capsys, state_dir) -> None:
    code = main(["ask", "anything", "--data", "ghost.csv", "--provider", "offline",
                 "--state-dir", state_dir, "--no-color"])
    assert code == 2
    assert "not found" in capsys.readouterr().out


def test_ask_session_file_is_written(capsys, tmp_path, state_dir) -> None:
    session = tmp_path / "session.json"
    code = main([
        "ask", "how many orders are there", "--data", "ecommerce_orders.csv",
        "--provider", "offline", "--state-dir", state_dir, "--no-color",
        "--session-file", str(session),
    ])
    assert code == 0
    payload = json.loads(session.read_text(encoding="utf-8"))
    assert payload["turns"][0]["tools_used"] == ["run_sql"]
    assert payload["datasets"]["ecommerce_orders"]["rows"] == 2600


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


def test_demo_employees_runs_clean(capsys, state_dir) -> None:
    code = main(["demo", "--dataset", "employees", "--no-color", "--quiet", "--state-dir", state_dir])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "demo finished cleanly" in out
    assert "offline" in out


def test_demo_reports_the_remembered_filter(capsys, tmp_path) -> None:
    code = main(["demo", "--dataset", "orders", "--no-color", "--quiet",
                 "--state-dir", str(tmp_path / "state")])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "remembered filter applied in SQL" in out


def test_memory_demo_does_not_touch_the_real_store(capsys, tmp_path) -> None:
    state = tmp_path / "state"
    main(["demo", "--dataset", "employees", "--no-color", "--quiet", "--state-dir", str(state)])
    capsys.readouterr()
    # The demo used a throwaway in-memory store, so no memory DB was created.
    assert not (state / "memory.db").exists()
