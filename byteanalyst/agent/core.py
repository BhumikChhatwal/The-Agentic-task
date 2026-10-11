"""The agent loop.

One turn looks like this:

1. **Retrieve memory** relevant to the question and the loaded tables (hybrid
   lexical + key-match ranking). Retrieved memories are injected as binding
   context and their usage counters are bumped.
2. **Reason and act.** The model is given the tool specs; every tool call is
   executed, its result rendered back into the conversation, and the loop repeats
   until the model answers instead of calling another tool.
3. **Verify grounding.** Every number in the answer is checked against the numbers
   the tools actually produced. Anything unsupported triggers exactly one repair
   turn, in which the model must either call a tool for the figure or drop it.
4. **Write back memory.** Durable definitions/preferences/corrections spotted in
   the exchange are stored so they affect later sessions.

When no provider is configured the same pipeline is driven by the deterministic
rule-based planner instead of a model; the memory and verification stages are
identical.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import Config
from ..datasets import DatasetRegistry
from ..errors import ByteAnalystError, LLMError
from ..formatting import numbers_equivalent, ToolResult
from ..llm import AssistantTurn, Message, build_client
from ..llm.base import ToolCall, parse_memory_json
from ..llm.offline import RuleBasedPlanner
from ..memory import MemoryStore
from ..tools import ToolContext, ToolRegistry, build_default_registry
from ..tools.registry import ToolExecution
from .prompt import build_system_prompt

MAX_REPAIRS = 1
# Numbers that are structural rather than evidential: list ordinals, small counts,
# years, and anything that appeared as a bare digit in a tool summary.
_IGNORABLE_NUMBERS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "24"}
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


@dataclass
class AgentEvent:
    """A step worth showing the user (streamed by the CLI and the web UI)."""

    type: str  # start | memory | tool_call | tool_result | answer | verification | saved | error
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}


@dataclass
class AgentAnswer:
    question: str
    answer: str
    ok: bool = True
    tool_executions: list[ToolExecution] = field(default_factory=list)
    memories_used: list[Any] = field(default_factory=list)
    memories_saved: list[dict[str, Any]] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    iterations: int = 0
    provider: str = ""
    model: str = ""
    duration_ms: int = 0
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def tools_used(self) -> list[str]:
        return [execution.name for execution in self.tool_executions]

    @property
    def grounded(self) -> bool:
        return not self.unsupported_numbers

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "ok": self.ok,
            "error": self.error,
            "tools_used": self.tools_used,
            "tool_executions": [e.to_dict() for e in self.tool_executions],
            "memories_used": [m.to_dict() if hasattr(m, "to_dict") else m for m in self.memories_used],
            "memories_saved": self.memories_saved,
            "unsupported_numbers": self.unsupported_numbers,
            "grounded": self.grounded,
            "iterations": self.iterations,
            "provider": self.provider,
            "model": self.model,
            "duration_ms": self.duration_ms,
            "artifacts": self.artifacts,
        }


def _number_index(executions: Sequence[ToolExecution]) -> set[str]:
    """Every numeric literal any tool produced, normalised for comparison."""
    found: set[str] = set()
    for execution in executions:
        result: ToolResult = execution.result
        haystacks = [result.summary, json.dumps(result.detail, default=str)]
        if result.table is not None:
            haystacks.append(json.dumps(result.table.to_dict(), default=str))
        for text in haystacks:
            for match in _NUMBER_RE.finditer(text or ""):
                found.add(match.group(0).replace(",", "").lstrip("+"))
    return found


def find_unsupported_numbers(answer: str, executions: Sequence[ToolExecution]) -> list[str]:
    """Numbers in the prose that no tool produced (allowing rounding)."""
    supported = _number_index(executions)
    unsupported: list[str] = []
    for match in _NUMBER_RE.finditer(answer or ""):
        raw = match.group(0).replace(",", "").lstrip("+")
        if raw in _IGNORABLE_NUMBERS:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if 1900 <= value <= 2100 and value == int(value):
            continue  # a year
        if supported and any(numbers_equivalent(raw, candidate) for candidate in supported):
            continue
        unsupported.append(raw)
    # de-duplicate, keeping order
    return list(dict.fromkeys(unsupported))


class Agent:
    def __init__(
        self,
        config: Config,
        registry: DatasetRegistry,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        client: Any = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.memory = memory if (memory is not None and config.memory_enabled) else None
        self.tools = tools or build_default_registry(include_memory=self.memory is not None)
        self.ctx = ToolContext(registry=registry, config=config, memory=self.memory)
        self.session_id = uuid.uuid4().hex[:8]
        self.history: list[Message] = []
        self.artifacts: list[str] = []
        self.provider = config.provider
        self._client = client
        # An explicitly injected client (tests, notebooks, a custom adapter) means the
        # model-driven loop runs even when no provider is configured.
        self.uses_llm = client is not None or config.provider != "offline"
        self._planner = None if self.uses_llm else RuleBasedPlanner(registry, self.memory)
        self._closed = False

    # -- construction -----------------------------------------------------
    @classmethod
    def from_paths(
        cls,
        paths: Sequence[str | Path],
        names: Sequence[str] | None = None,
        config: Config | None = None,
        use_memory: bool | None = None,
        client: Any = None,
    ) -> "Agent":
        config = config or Config.from_env()
        config.ensure_dirs()
        registry = DatasetRegistry()
        for index, path in enumerate(paths):
            name = names[index] if names and index < len(names) else None
            registry.load_file(path, name=name)
        memory = None
        if use_memory is None:
            use_memory = config.memory_enabled
        if use_memory:
            memory = MemoryStore(config.memory_path)
        return cls(config, registry, memory=memory, client=client)

    def client(self) -> Any:
        if self._client is None:
            self._client = build_client(self.config)
        return self._client

    @property
    def model_name(self) -> str:
        if not self.uses_llm:
            return "offline-rule-based-planner"
        return getattr(self._client, "model", self.config.model)

    # -- memory -----------------------------------------------------------
    def _retrieve_memories(self, question: str, on_event: Callable | None) -> list[Any]:
        if self.memory is None:
            return []
        try:
            memories = self.memory.retrieve(
                question,
                context_keys=self.registry.context_keys(),
                scope=self.ctx.primary_name,
                limit=6,
                touch=True,
            )
        except ByteAnalystError as exc:  # memory problems must never break analysis
            self._emit(on_event, AgentEvent("error", {"where": "memory", "message": str(exc)}))
            return []
        if memories:
            self._emit(
                on_event,
                AgentEvent(
                    "memory",
                    {
                        "count": len(memories),
                        "items": [
                            {"id": m.record.id, "kind": m.record.kind, "text": m.record.text,
                             "score": round(m.score, 3), "reasons": m.reasons}
                            for m in memories
                        ],
                    },
                ),
            )
        return memories

    def _extract_memories(
        self, question: str, answer: str, executions: Sequence[ToolExecution], on_event
    ) -> list[dict[str, Any]]:
        if self.memory is None or not self.config.auto_memory:
            return []
        candidates: list[dict[str, Any]] = []
        if not self.uses_llm:
            candidates = _regex_memory_candidates(question, self.registry, self.ctx.primary_name)
        else:
            try:
                candidates = self._llm_memory_candidates(question, answer, executions)
            except Exception as exc:  # memory is best-effort: never fail the answer over it
                self._emit(
                    on_event,
                    AgentEvent("error", {"where": "memory_extraction", "message": str(exc)[:300]}),
                )
                candidates = []

        saved: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                outcome = self.memory.save(
                    text=candidate["text"],
                    kind=candidate.get("kind", "fact"),
                    keys=candidate.get("keys") or [],
                    payload=candidate.get("payload") or {},
                    scope=candidate.get("scope") or "global",
                    confidence=float(candidate.get("confidence", 0.85)),
                    source="agent",
                )
            except ByteAnalystError as exc:
                self._emit(on_event, AgentEvent("error", {"where": "memory_save", "message": str(exc)}))
                continue
            record = {
                "id": outcome.record.id,
                "kind": outcome.record.kind,
                "text": outcome.record.text,
                "keys": outcome.record.keys,
                "action": outcome.action,
                "superseded": [c.id for c in outcome.superseded],
            }
            saved.append(record)
            self._emit(on_event, AgentEvent("saved", record))
        return saved

    def _llm_memory_candidates(self, question, answer, executions) -> list[dict[str, Any]]:
        from .prompt import render_memory_prompt

        prompt = render_memory_prompt(
            question=question,
            tools=", ".join(sorted({e.name for e in executions})) or "none",
            answer=answer[:2500],
        )
        turn: AssistantTurn = self.client().chat([Message(role="user", content=prompt)], tools=None)
        return parse_memory_json(turn.content)

    # -- main entry point -------------------------------------------------
    def ask(
        self,
        question: str,
        on_event: Callable[[AgentEvent], None] | None = None,
        use_memory: bool = True,
    ) -> AgentAnswer:
        """Answer one question.

        ``use_memory=False`` disables both memory retrieval and memory write-back
        for this turn only, without discarding anything already stored.
        """
        question = (question or "").strip()
        if not question:
            raise ByteAnalystError("the question must not be empty")
        started = time.perf_counter()
        self._emit(on_event, AgentEvent("start", {"question": question, "provider": self.provider,
                                                  "model": self.model_name}))
        memories = self._retrieve_memories(question, on_event) if use_memory else []

        executions: list[ToolExecution] = []
        unsupported: list[str] = []
        error: str | None = None
        answer = ""
        iterations = 0

        try:
            if not self.uses_llm:
                iterations, answer = self._run_offline(question, executions, memories, on_event)
                unsupported = find_unsupported_numbers(answer, executions)
                self._emit(
                    on_event,
                    AgentEvent("verification", {"grounded": not unsupported, "unsupported": unsupported}),
                )
            else:
                iterations, answer, unsupported = self._run_llm(
                    question, executions, memories, on_event
                )
        except LLMError as exc:
            error = str(exc)
            answer = (
                f"I could not reach the model provider, so I did not analyse the data.\n\n"
                f"Provider error: {error}\n\n"
                "Either fix the API configuration or run with `--provider offline` to use the "
                "built-in rule-based planner."
            )
            self._emit(on_event, AgentEvent("error", {"where": "llm", "message": error}))
        except ByteAnalystError as exc:
            error = str(exc)
            answer = f"I could not complete the analysis: {error}"
            self._emit(on_event, AgentEvent("error", {"where": "agent", "message": error}))

        memories_saved = (
            self._extract_memories(question, answer, executions, on_event)
            if (not error and use_memory)
            else []
        )
        self.history.extend([
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ])
        self.artifacts = list(self.ctx.artifacts)
        self._emit(on_event, AgentEvent("answer", {"answer": answer, "unsupported_numbers": unsupported}))

        return AgentAnswer(
            question=question,
            answer=answer,
            ok=error is None,
            tool_executions=executions,
            memories_used=memories,
            memories_saved=memories_saved,
            unsupported_numbers=unsupported,
            iterations=iterations,
            provider=self.provider,
            model=self.model_name,
            duration_ms=int((time.perf_counter() - started) * 1000),
            artifacts=list(self.ctx.artifacts),
            error=error,
        )

    # -- the LLM loop -----------------------------------------------------
    def _run_llm(
        self,
        question: str,
        executions: list[ToolExecution],
        memories: Sequence[Any],
        on_event: Callable | None,
    ) -> tuple[int, str, list[str]]:
        system = build_system_prompt(
            self.registry,
            memories,
            tools_catalogue=self.tools.catalogue(),
            provider="offline" if not self.uses_llm else self.config.provider,
        )
        messages: list[Message] = [Message(role="system", content=system), *self.history,
                                   Message(role="user", content=question)]
        specs = self.tools.specs()
        iterations = 0
        repairs = 0
        answer = ""

        while iterations < self.config.max_iterations:
            iterations += 1
            turn = self.client().chat(messages, tools=specs)
            if turn.usage:
                self._emit(on_event, AgentEvent("thinking", {"usage": turn.usage, "iteration": iterations}))

            if turn.wants_tools:
                messages.append(
                    Message(role="assistant", content=turn.content, tool_calls=turn.tool_calls)
                )
                for call in turn.tool_calls:
                    result_text = self._execute_call(call, executions, on_event)
                    messages.append(
                        Message(role="tool", content=result_text, tool_call_id=call.id, name=call.name)
                    )
                continue

            answer = turn.content or ""
            messages.append(Message(role="assistant", content=answer))

            unsupported = find_unsupported_numbers(answer, executions)
            if not unsupported:
                self._emit(on_event, AgentEvent("verification", {"grounded": True, "unsupported": []}))
                return iterations, answer, []

            self._emit(
                on_event,
                AgentEvent("verification", {"grounded": False, "unsupported": unsupported}),
            )
            if repairs >= MAX_REPAIRS:
                return iterations, answer, unsupported
            repairs += 1
            messages.append(
                Message(
                    role="user",
                    content=(
                        "Verification failed. These numbers in your answer were not produced by any "
                        f"tool call: {', '.join(unsupported[:12])}. "
                        "Either call the tool that actually computes them, or remove them from the "
                        "answer. Do not estimate or recall figures from training data."
                    ),
                )
            )

        if not answer:
            answer = (
                "I hit the tool-call limit for this question without reaching an answer. "
                "Try asking something narrower, or raise BYTEANALYST_MAX_ITERATIONS."
            )
        return iterations, answer, find_unsupported_numbers(answer, executions)

    # -- the offline loop -------------------------------------------------
    def _run_offline(
        self,
        question: str,
        executions: list[ToolExecution],
        memories: Sequence[Any],
        on_event: Callable | None,
    ) -> tuple[int, str]:
        planner = self._planner
        assert planner is not None
        iterations = 0
        while iterations < self.config.max_iterations:
            calls = planner.plan(question, executions, memories)
            if not calls:
                break
            iterations += 1
            for call in calls:
                self._execute_call(call, executions, on_event)
        answer = planner.compose(question, executions, memories)
        return iterations, answer

    def _execute_call(
        self, call: ToolCall, executions: list[ToolExecution], on_event: Callable | None
    ) -> str:
        self._emit(on_event, AgentEvent("tool_call", {"name": call.name, "arguments": call.arguments}))
        execution = self.tools.execute(call.name, call.arguments, self.ctx)
        executions.append(execution)
        self._emit(
            on_event,
            AgentEvent(
                "tool_result",
                {
                    "name": call.name,
                    "ok": execution.result.ok,
                    "duration_ms": execution.duration_ms,
                    "summary": execution.result.summary,
                    "error": execution.result.error,
                    "table": execution.result.table.to_dict() if execution.result.table else None,
                    "artifacts": execution.result.artifacts,
                },
            ),
        )
        return execution.result.to_llm_text(self.config.max_tool_output_chars)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _emit(on_event: Callable[[AgentEvent], None] | None, event: AgentEvent) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:  # a broken listener must not break the run
            pass

    def close(self) -> None:
        if self.memory is not None and not self._closed:
            self.memory.close()
            self._closed = True

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def reset(self) -> None:
        self.history.clear()
        self.ctx.artifacts.clear()


# ---------------------------------------------------------------------------
# Offline memory extraction (regex) — used when no model is available
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("filter", re.compile(r"([\w][\w ]{1,38}?)\s+(?:excludes?|does not include|doesn't include)\s+(.{2,60})", re.I)),
    ("definition", re.compile(r"([\w][\w ]{1,38}?)\s+(?:means|is defined as|refers to|counts as)\s+(.{2,80})", re.I)),
    ("preference", re.compile(r"\b(?:always|from now on|going forward|in future|please always)\s+(.{4,120})", re.I)),
    ("correction", re.compile(r"\b(?:actually|that's wrong|incorrect|that is wrong)\b[,: ]+(.{4,120})", re.I)),
]

# Higher priority wins when two patterns match the same span.
_KIND_PRIORITY = {"correction": 4, "filter": 3, "definition": 3, "preference": 2, "fact": 1}

_CUE_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:remember|note|keep in mind|fyi|just so you know)\b[:,]?\s*(?:that\s+)?",
    re.IGNORECASE,
)


def _regex_memory_candidates(
    question: str, registry: DatasetRegistry, primary: str | None
) -> list[dict[str, Any]]:
    """Conservative extraction so the keyless path can still build memory.

    Deliberately narrow: it only fires on explicit definitions, exclusions,
    standing preferences and corrections, strips conversational cue words, and
    keeps at most one candidate per overlapping span so a single sentence never
    becomes two near-duplicate memories.
    """
    text = _CUE_PREFIX.sub("", (question or "").strip()).strip()
    if len(text) < 10:
        return []
    columns = {c.lower() for dataset in registry.datasets() for c in dataset.columns}
    raw: list[tuple[int, str, str, dict, tuple[int, int]]] = []

    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            payload: dict[str, Any] = {}
            if kind == "filter":
                subject, excluded = match.group(1).strip(), match.group(2).strip()
                subject = _CUE_PREFIX.sub("", subject).strip()
                sentence = f"{subject} excludes {excluded}"
                token = excluded.split()[0].strip(".,;:'\"")
                if token and token.lower() not in columns:
                    payload = {"exclude": token, "sql": f"status <> '{token}'"}
            elif kind == "definition":
                sentence = f"{match.group(1).strip()} means {match.group(2).strip()}"
            else:
                sentence = match.group(1).strip()
            sentence = re.sub(r"\s+", " ", sentence).rstrip(".").strip()
            if not (6 <= len(sentence) <= 200):
                continue
            raw.append((_KIND_PRIORITY.get(kind, 1), kind, sentence, payload, match.span()))

    raw.sort(key=lambda item: (-item[0], item[4][0]))
    chosen: list[tuple] = []
    for candidate in raw:
        span_lo, span_hi = candidate[4]
        overlaps = any(not (span_hi <= other[4][0] or span_lo >= other[4][1]) for other in chosen)
        contains = any(
            candidate[2].lower() in other[2].lower() or other[2].lower() in candidate[2].lower()
            for other in chosen
        )
        if overlaps or contains:
            continue
        chosen.append(candidate)

    out: list[dict[str, Any]] = []
    for _, kind, sentence, payload, _ in chosen[:2]:
        keys = [c for c in sorted(columns) if c in sentence.lower()]
        out.append(
            {
                "text": sentence,
                "kind": kind,
                "keys": keys,
                "payload": payload,
                "scope": "global",
                "confidence": 0.8 if kind in {"filter", "definition"} else 0.7,
            }
        )
    return out
