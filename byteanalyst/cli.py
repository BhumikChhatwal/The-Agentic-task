"""The ``byteanalyst`` command line.

    byteanalyst ask "total revenue by region" --data data/ecommerce_orders.csv
    byteanalyst ask "..." --data a.csv --data b.csv        # multiple tables, joinable
    byteanalyst chat --data data/employees.csv
    byteanalyst profile data/weather_daily.csv
    byteanalyst memory list
    byteanalyst demo
    byteanalyst serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .agent import Agent
from .config import PROVIDERS, Config
from .datasets import DatasetRegistry
from .errors import ByteAnalystError, ConfigError
from .formatting import color
from .memory import KINDS, MemoryStore
from .render import Renderer
from .tools import ToolContext, build_default_registry

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data"


# ---------------------------------------------------------------------------
# argument plumbing
# ---------------------------------------------------------------------------


def _add_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d", "--data", action="append", default=[], metavar="FILE",
        help="A tabular file to load (CSV/TSV/JSON/JSONL/Parquet/Excel). Repeat for multiple tables.",
    )
    parser.add_argument(
        "--name", action="append", default=[], metavar="NAME",
        help="Override the table name for the corresponding --data (order matters).",
    )
    parser.add_argument("--provider", choices=PROVIDERS, help="LLM backend. Default: auto-detect.")
    parser.add_argument("--model", help="Model name (e.g. gpt-4o-mini, claude-3-5-sonnet-latest).")
    parser.add_argument("--base-url", dest="base_url", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", dest="api_key", help="API key (prefer the env var or .env).")
    parser.add_argument("--max-iterations", type=int, dest="max_iterations", help="Tool-call budget.")
    parser.add_argument(
        "--state-dir", dest="state_dir", metavar="DIR",
        help="Where memory, artifacts and transcripts live (default ./.byteanalyst).",
    )
    parser.add_argument("--no-memory", action="store_true", help="Disable persistent memory entirely.")
    parser.add_argument("--no-auto-memory", action="store_true", help="Do not write memories automatically.")
    parser.add_argument(
        "--trace", action="store_true",
        help="Show the tool trace (this is already the default; use it to override --quiet).",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide the tool trace.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colours.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="byteanalyst",
        description="An LLM agent that analyses arbitrary tabular datasets with real analysis tools.",
    )
    parser.add_argument("--version", action="version", version=f"byteanalyst {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask one question about the loaded data.")
    _add_source_options(ask)
    ask.add_argument("question", help="The question to answer.")
    ask.add_argument("--session-file", help="Write the full transcript JSON to this path.")

    chat = sub.add_parser("chat", help="Interactive multi-turn session.")
    _add_source_options(chat)
    chat.add_argument("--session-file", help="Append the transcript JSON here on exit.")

    profile = sub.add_parser("profile", help="Profile a file without any model (no API key needed).")
    profile.add_argument("file", help="Tabular file to profile.")
    profile.add_argument("--columns", nargs="*", help="Restrict to these columns.")
    profile.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    profile.add_argument("--no-color", action="store_true")

    memory = sub.add_parser("memory", help="Inspect and control persistent memory.")
    memory.add_argument(
        "action",
        choices=["list", "search", "add", "forget", "edit", "clear", "export", "import", "stats", "history"],
    )
    memory.add_argument("value", nargs="?", help="Query / memory id / text, depending on the action.")
    memory.add_argument("--kind", choices=list(KINDS), help="Kind for `add` or a filter for `list`.")
    memory.add_argument("--text", help="Replacement text for `edit`.")
    memory.add_argument("--keys", nargs="*", default=[], help="Keys (columns/tables) for `add`.")
    memory.add_argument("--payload-sql", help="Structured SQL predicate for `add` (filters).")
    memory.add_argument("--scope", default="global", help="Dataset scope for `add`.")
    memory.add_argument("--confidence", type=float, default=0.9)
    memory.add_argument("--yes", action="store_true", help="Confirm destructive actions.")
    memory.add_argument("--state-dir", dest="state_dir", metavar="DIR")
    memory.add_argument("--json", action="store_true")
    memory.add_argument("--no-color", action="store_true")

    demo = sub.add_parser("demo", help="Run the bundled datasets with no API key (offline planner).")
    demo.add_argument("--dataset", choices=["orders", "employees", "weather", "tickets", "all"], default="all")
    demo.add_argument("--provider", choices=PROVIDERS, help="Force a provider (default: offline).")
    demo.add_argument("--model")
    demo.add_argument("--base-url", dest="base_url")
    demo.add_argument("--state-dir", dest="state_dir", metavar="DIR", help="Where artifacts are written.")
    demo.add_argument("--no-color", action="store_true")
    demo.add_argument("--quiet", action="store_true")

    tools = sub.add_parser("tools", help="List the tools the agent can call.")
    tools.add_argument("--json", action="store_true")
    tools.add_argument("--no-color", action="store_true")

    serve = sub.add_parser("serve", help="Start the local web UI.")
    _add_source_options(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {}
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "base_url", None):
        overrides["base_url"] = args.base_url
    if getattr(args, "api_key", None):
        overrides["api_key"] = args.api_key
    if getattr(args, "max_iterations", None):
        overrides["max_iterations"] = args.max_iterations
    if getattr(args, "no_memory", False):
        overrides["memory_enabled"] = False
    if getattr(args, "no_auto_memory", False):
        overrides["auto_memory"] = False
    state_dir = getattr(args, "state_dir", None)
    if state_dir:
        overrides["state_dir"] = Path(state_dir).expanduser().resolve()
    config = Config.from_env(root=ROOT, **overrides)
    config.ensure_dirs()
    return config


def _make_agent(args: argparse.Namespace) -> Agent:
    config = _config_from_args(args)
    paths = _resolve_paths(args.data)
    if not paths:
        raise ByteAnalystError(
            "no data loaded. Pass at least one file with --data (or run `byteanalyst demo`)."
        )
    agent = Agent.from_paths(paths, names=getattr(args, "name", None) or None, config=config)
    return agent


def _resolve_paths(paths: Sequence[str]) -> list[Path]:
    resolved = []
    for raw in paths:
        candidate = Path(raw).expanduser()
        if not candidate.exists():
            # A bare name may refer to a bundled sample.
            fallback = SAMPLE / raw
            if fallback.exists():
                candidate = fallback
            else:
                raise ByteAnalystError(f"file not found: {raw}")
        resolved.append(candidate)
    return resolved


def _renderer(args: argparse.Namespace) -> Renderer:
    as_json = bool(getattr(args, "json", False))
    if as_json:
        # --json means "machine readable stdout", so the human trace is suppressed
        # even if --trace was also passed.
        trace = False
    elif getattr(args, "trace", False):
        trace = True
    else:
        trace = not getattr(args, "quiet", False)
    return Renderer(
        use_color=False if getattr(args, "no_color", False) else None,
        trace=trace,
    )


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> int:
    from .errors import LLMError

    renderer = _renderer(args)
    try:
        agent = _make_agent(args)
    except (ByteAnalystError, ConfigError) as exc:
        renderer.write(renderer.paint(f"error: {exc}", "red"))
        return 2

    with agent:
        try:
            result = agent.ask(args.question, on_event=renderer.event)
        except LLMError as exc:
            renderer.write(renderer.paint(f"model error: {exc}", "red"))
            return 3

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        renderer.answer(result)
    if getattr(args, "session_file", None):
        _write_session(args.session_file, agent, [result])
    return 0 if result.ok else 1


def cmd_chat(args: argparse.Namespace) -> int:
    from .errors import LLMError

    renderer = _renderer(args)
    try:
        agent = _make_agent(args)
    except (ByteAnalystError, ConfigError) as exc:
        renderer.write(renderer.paint(f"error: {exc}", "red"))
        return 2

    results = []
    tokens = ", ".join(f"`{d.name}`" for d in agent.registry.datasets())
    renderer.write(renderer.paint(f"Byte Analyst · {agent.model_name}", "cyan", "bold"))
    renderer.write(renderer.paint(f"tables: {tokens}", "grey"))
    renderer.write(
        renderer.paint(
            "type a question, or /tools /memory /reset /quit. Ctrl-D also exits.", "grey"
        )
    )
    with agent:
        while True:
            try:
                line = input(renderer.paint("\n> ", "cyan", "bold")).strip()
            except (EOFError, KeyboardInterrupt):
                renderer.write()
                break
            if not line:
                continue
            if line in {"/quit", "/exit", ":q"}:
                break
            if line == "/tools":
                for tool in agent.tools.tools():
                    renderer.write(renderer.paint(f"  {tool.name}", "blue") + f" — {tool.description.splitlines()[0]}")
                continue
            if line == "/memory":
                _print_memory(_memory_store_from_agent(agent), renderer, None, None)
                continue
            if line == "/reset":
                agent.reset()
                renderer.write(renderer.paint("conversation reset (memory kept)", "grey"))
                continue
            try:
                result = agent.ask(line, on_event=renderer.event)
            except LLMError as exc:
                renderer.write(renderer.paint(f"model error: {exc}", "red"))
                continue
            except ByteAnalystError as exc:
                renderer.write(renderer.paint(f"error: {exc}", "red"))
                continue
            renderer.answer(result)
            results.append(result)
    if getattr(args, "session_file", None) and results:
        _write_session(args.session_file, agent, results)
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    renderer = Renderer(use_color=False if args.no_color else None)
    try:
        registry = DatasetRegistry()
        registry.load_file(_resolve_paths([args.file])[0])
        config = Config.from_env(root=ROOT)
        config.ensure_dirs()
        tools = build_default_registry(include_memory=False)
        ctx = ToolContext(registry=registry, config=config, memory=None)
        execution = tools.execute(
            "profile_dataset",
            {"columns": args.columns} if args.columns else {},
            ctx,
        )
    except ByteAnalystError as exc:
        renderer.write(renderer.paint(f"error: {exc}", "red"))
        return 2

    result = execution.result
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.ok else 1
    renderer.write(renderer.paint(result.summary, "cyan", "bold"))
    if result.table is not None:
        renderer.write()
        renderer.write(result.table.render())
    for key, value in result.detail.items():
        renderer.write(renderer.paint(f"  {key}: {value}", "grey"))
    return 0 if result.ok else 1


def cmd_tools(args: argparse.Namespace) -> int:
    registry = build_default_registry()
    if args.json:
        print(json.dumps([tool.spec() for tool in registry.tools()], indent=2))
        return 0
    renderer = Renderer(use_color=False if args.no_color else None)
    renderer.write(renderer.paint(f"{len(registry.tools())} tools", "cyan", "bold"))
    for tool in registry.tools():
        renderer.write()
        renderer.write(renderer.paint(f"  {tool.name}  [{tool.category}]", "blue", "bold"))
        for line in tool.description.splitlines():
            renderer.write(f"    {line}")
        properties = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        if properties:
            renderer.write(renderer.paint("    arguments:", "grey"))
            for name, spec in properties.items():
                flag = " (required)" if name in required else ""
                kind = spec.get("type", "any")
                renderer.write(renderer.paint(f"      {name}: {kind}{flag}", "grey"))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    renderer = Renderer(use_color=False if args.no_color else None)
    config = _config_from_args(argparse.Namespace(state_dir=getattr(args, "state_dir", None)))
    store = MemoryStore(config.memory_path)
    try:
        return _memory_action(store, args, renderer)
    finally:
        store.close()


def _memory_action(store: MemoryStore, args: argparse.Namespace, renderer: Renderer) -> int:
    action = args.action

    if action == "list":
        _print_memory(store, renderer, args.kind, args.json)
        return 0

    if action == "search":
        query = args.value or ""
        results = store.retrieve(query, limit=10, touch=False)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
            return 0
        if not results:
            renderer.write(renderer.paint(f"nothing matches {query!r}", "yellow"))
            return 0
        renderer.write(renderer.paint(f"{len(results)} match(es) for {query!r}", "cyan", "bold"))
        for item in results:
            renderer.write(f"  {item.score:.3f}  {item.record.describe()}")
            renderer.write(renderer.paint(f"         why: {'; '.join(item.reasons) or '-'}", "grey"))
        return 0

    if action == "add":
        text = args.value
        if not text:
            renderer.write(renderer.paint("error: `memory add` needs text", "red"))
            return 2
        payload = {"sql": args.payload_sql} if args.payload_sql else {}
        outcome = store.save(
            text=text,
            kind=args.kind or "fact",
            keys=args.keys,
            payload=payload,
            scope=args.scope,
            confidence=args.confidence,
            source="user",
        )
        renderer.write(
            renderer.paint(f"{outcome.action}: [{outcome.record.id}] {outcome.record.text}", "green")
        )
        for superseded in outcome.superseded:
            renderer.write(renderer.paint(f"  superseded [{superseded.id}] {superseded.text}", "yellow"))
        return 0

    if action == "forget":
        if not args.value:
            renderer.write(renderer.paint("error: `memory forget` needs an id", "red"))
            return 2
        ok = store.forget(args.value)
        renderer.write(
            renderer.paint(f"forgot [{args.value}]" if ok else f"no memory [{args.value}]",
                           "green" if ok else "yellow")
        )
        return 0 if ok else 1

    if action == "edit":
        if not args.value:
            renderer.write(renderer.paint("error: `memory edit` needs an id", "red"))
            return 2
        record = store.edit(args.value, text=args.text, payload={"sql": args.payload_sql} if args.payload_sql else None)
        renderer.write(renderer.paint(f"updated [{record.id}] {record.text}", "green"))
        return 0

    if action == "clear":
        if not args.yes:
            renderer.write(renderer.paint("refusing to clear without --yes", "yellow"))
            return 2
        count = store.clear(hard=False)
        renderer.write(renderer.paint(f"deactivated {count} memory(ies) (history kept)", "green"))
        return 0

    if action == "export":
        print(json.dumps(store.export(), indent=2, default=str))
        return 0

    if action == "import":
        if not args.value:
            renderer.write(renderer.paint("error: `memory import` needs a JSON file path", "red"))
            return 2
        rows = json.loads(Path(args.value).read_text(encoding="utf-8"))
        count = store.import_records(rows)
        renderer.write(renderer.paint(f"imported {count} memory(ies)", "green"))
        return 0

    if action == "stats":
        stats = store.stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            renderer.write(renderer.paint(str(stats), "cyan"))
        return 0

    if action == "history":
        for row in store.history(limit=30):
            renderer.write(f"  {row['at']}  {row['action']:<12} {row['memory_id'] or '-':<14} {row['note'] or ''}")
        return 0

    renderer.write(renderer.paint(f"unknown action {action!r}", "red"))
    return 2


def _print_memory(store: MemoryStore, renderer: Renderer, kind: str | None, as_json: bool | None) -> None:
    records = store.list(kind=kind, include_inactive=False)
    if as_json:
        print(json.dumps([r.to_dict() for r in records], indent=2, default=str))
        return
    stats = store.stats()
    renderer.write(
        renderer.paint(
            f"memory: {stats['active']} active, {stats['superseded']} superseded  {stats['by_kind']}",
            "cyan", "bold",
        )
    )
    if not records:
        renderer.write(renderer.paint("  (empty)", "grey"))
        return
    for record in records:
        renderer.write(f"  {record.describe()}")


def _memory_store_from_agent(agent: Agent) -> MemoryStore:
    return agent.memory  # type: ignore[return-value]


def _write_session(path: str, agent: Agent, results: Sequence[Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": agent.session_id,
        "provider": agent.provider,
        "model": agent.model_name,
        "datasets": {
            d.name: {"source": d.source, "rows": d.row_count, "columns": d.columns}
            for d in agent.registry.datasets()
        },
        "turns": [r.to_dict() for r in results],
    }
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

DEMO_PLAN: dict[str, list[str]] = {
    "orders": [
        "what's in this data",
        "total revenue by region",
        "how does revenue trend over time and what does the next 6 months look like",
        "are there any outliers in revenue",
        "is the revenue difference between the North and South regions significant",
    ],
    "employees": [
        "profile the data",
        "is salary significantly different between Engineering and Sales",
        "does attrition depend on department",
    ],
    "weather": [
        "how does temp_c trend over time by city",
        "are there any unusual temperature spikes in temp_c over time",
    ],
    "tickets": [
        "profile the data",
        "what is the average resolution time by priority",
    ],
}

DEMO_FILES = {
    "orders": ["ecommerce_orders.csv", "customers.csv"],
    "employees": ["employees.csv"],
    "weather": ["weather_daily.csv"],
    "tickets": ["support_tickets.csv"],
}


def cmd_demo(args: argparse.Namespace) -> int:
    renderer = Renderer(use_color=False if args.no_color else None, trace=not args.quiet)
    config = _config_from_args(args)
    if not args.provider:
        config.provider = "offline"
        config.api_key = None
    config.memory_enabled = True
    config.ensure_dirs()

    wanted = list(DEMO_PLAN) if args.dataset == "all" else [args.dataset]
    renderer.write(
        renderer.paint(
            "Byte Analyst — offline demo (deterministic rule-based planner, no API key)",
            "cyan", "bold",
        )
    )
    renderer.write(
        renderer.paint(
            "Memory is exercised in-process only: nothing is written to your real memory DB.",
            "grey",
        )
    )

    failures = 0
    for name in wanted:
        paths = [_resolve_paths([DEMO_FILES[name][0]])[0]]
        if len(DEMO_FILES[name]) > 1:
            paths.extend(_resolve_paths(DEMO_FILES[name][1:]))
        # A throwaway in-memory store: the demo never touches the real memory DB.
        registry = DatasetRegistry()
        for path in paths:
            registry.load_file(path)
        memory = MemoryStore(":memory:")
        agent = Agent(config, registry, memory=memory)

        renderer.write()
        renderer.write(renderer.paint(f"{'=' * 72}", "grey"))
        renderer.write(renderer.paint(f"DATASET: {name}  ({', '.join(DEMO_FILES[name])})", "cyan", "bold"))
        renderer.write(renderer.paint(f"{'=' * 72}", "grey"))

        for question in DEMO_PLAN[name]:
            result = agent.ask(question, on_event=renderer.event)
            renderer.answer(result)
            if not result.ok:
                failures += 1
        memory.close()
        agent.close()

    if args.dataset in {"all", "orders"}:
        failures += _demo_memory(config, renderer)

    renderer.write()
    if failures:
        renderer.write(renderer.paint(f"demo finished with {failures} failed turn(s)", "yellow"))
        return 1
    renderer.write(renderer.paint("demo finished cleanly", "green", "bold"))
    return 0


def _demo_memory(config: Config, renderer: Renderer) -> int:
    """Show that a remembered definition changes a later answer.

    Runs against the orders table, where the remembered filter actually applies.
    """
    renderer.write()
    renderer.write(renderer.paint(f"{'=' * 72}", "grey"))
    renderer.write(renderer.paint("MEMORY: a definition that outlives the session", "magenta", "bold"))
    renderer.write(renderer.paint(f"{'=' * 72}", "grey"))

    registry = DatasetRegistry()
    registry.load_file(_resolve_paths([DEMO_FILES["orders"][0]])[0])
    memory = MemoryStore(":memory:")
    agent = Agent(config, registry, memory=memory)
    failures = 0
    try:
        renderer.write(renderer.paint('user: "how many active customers are there by region"', "grey"))
        before = agent.ask("how many active customers are there by region", on_event=renderer.event)
        renderer.answer(before)
        renderer.write(
            renderer.paint(
                "  note the SQL above: nothing yet defines 'active', so cancelled orders are counted",
                "grey",
            )
        )

        renderer.write()
        renderer.write(
            renderer.paint(
                'user: "remember that active customers exclude cancelled accounts"', "grey"
            )
        )
        agent.ask("remember that active customers exclude cancelled accounts", on_event=renderer.event)

        # A brand-new agent object, same memory file: this is the cross-session part.
        agent2 = Agent(config, registry, memory=memory)
        try:
            renderer.write(renderer.paint('new agent, same memory — user: "how many active customers?"', "grey"))
            after = agent2.ask("how many active customers are there by region", on_event=renderer.event)
            renderer.answer(after)
            query = after.tool_executions[0].arguments.get("query", "") if after.tool_executions else ""
            applied = bool(after.tool_executions) and after.tool_executions[0].result.ok and "'cancelled'" in query
            if applied:
                renderer.write(renderer.paint(f"  remembered filter applied in SQL: {query}", "magenta"))
            else:
                failures += 1
                renderer.write(renderer.paint(f"  the remembered filter was NOT applied: {query}", "yellow"))
        finally:
            agent2.close()
    finally:
        agent.close()
        memory.close()
    return failures


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(color("the web UI needs extra packages: pip install 'byteanalyst[web]'", "red"))
        return 2
    from .web.app import create_app

    config = _config_from_args(args)
    try:
        datasets = _resolve_paths(args.data) if args.data else []
    except ByteAnalystError as exc:
        print(color(f"error: {exc}", "red"))
        return 2
    app = create_app(config, datasets)
    print(color(f"Byte Analyst web UI → http://{args.host}:{args.port}", "cyan", "bold"))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# ---------------------------------------------------------------------------


COMMANDS = {
    "ask": cmd_ask,
    "chat": cmd_chat,
    "profile": cmd_profile,
    "memory": cmd_memory,
    "demo": cmd_demo,
    "tools": cmd_tools,
    "serve": cmd_serve,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
    except ByteAnalystError as exc:
        Renderer(use_color=False if getattr(args, "no_color", False) else None).write(
            f"error: {exc}"
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
