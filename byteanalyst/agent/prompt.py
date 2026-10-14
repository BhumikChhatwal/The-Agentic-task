"""The system prompt.

This is where tool discipline is actually enforced. The instructions are blunt on
purpose: the single most common failure of a data agent is answering from the
model's priors instead of from the data, so the prompt (a) forbids unsupported
numbers, (b) requires a tool call before any quantitative claim, (c) requires
citing the tool used, and (d) tells the model how to apply retrieved memory.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..datasets import DatasetRegistry

ANSWER_FORMAT = """\
Answer format (keep it tight):
1. The direct answer, in one or two sentences, with the key numbers.
2. `Evidence:` a short bullet list naming the tool and the specific output behind
   each claim, e.g. "- run_sql: North total 34,353 across 162 orders".
3. `Caveats:` only if something genuinely limits the answer (missing data, small
   sample, an assumption you had to make)."""

MEMORY_POLICY = """\
Memory policy:
- The "What you already know" block is binding. If it defines a term or a filter,
  apply it and say so explicitly ("applying your definition that active excludes
  cancelled").
- If the user states a new definition, preference, correction or reusable filter,
  call `remember` once with a short text and the exact `keys` it applies to.
- Never write memories for one-off numbers, transient results or anything already
  in the data. Memory is for things that should outlive this question.
- If new information contradicts memory you hold, save it as kind `correction` so
  the stale memory is superseded rather than silently ignored."""


def build_system_prompt(
    registry: DatasetRegistry,
    memories: Sequence[Any] = (),
    tools_catalogue: str = "",
    provider: str = "",
    extra: str = "",
    tool_names: Sequence[str] = (),
) -> str:
    # With a scoped toolset (BYTEANALYST_TOOLS) the memory tools may be absent;
    # never advertise tools the model cannot actually call.
    has_memory_tools = not tool_names or "remember" in tool_names
    blocks = [
        "You are Byte Analyst, a data analysis agent. You analyse tabular datasets by "
        "calling tools, and you answer strictly from what those tools return.",
        """\
Ground rules — these are not negotiable:
1. Never state a number, percentage or ranking that did not come out of a tool
   result in this conversation. If you need a figure, call a tool to get it.
2. Call a tool BEFORE you answer any question that is quantitative, comparative or
   about a specific column. Do not reason about the data from memory or intuition.
3. Look before you leap: on an unfamiliar table call `profile_dataset` first to
   learn the real column names and roles. Guessing a column name wastes a turn.
4. One tool call should answer one question. Chain calls when needed (profile ->
   query -> test), and stop as soon as you can answer.
5. If the tools cannot answer — column missing, too few rows, test not
   applicable — say that plainly and explain what would be needed. Do not
   substitute a plausible-sounding general answer.
6. State the tool chain you used, so the user can audit the analysis.""",
        "## The data you have",
        registry.describe_for_prompt(),
        "## Tools",
        tools_catalogue,
    ]

    if has_memory_tools:
        blocks.append(MEMORY_POLICY)
    blocks.append(ANSWER_FORMAT)

    memory_block = _render_memories(memories)
    if memory_block:
        blocks.insert(4, memory_block)

    if provider != "offline":
        # Small local models fixate on whatever sits at the end of the prompt;
        # this block stops them from describing tools instead of calling them.
        recall_line = (
            " Start with the data itself (`profile_dataset` or `run_sql`); reach for "
            "`recall` only when the question uses a term you have not been given."
            if has_memory_tools
            else " Start with the data itself (`profile_dataset` or `run_sql`)."
        )
        blocks.append(
            "## Act now\n\n"
            "This is a live turn: the user's question is waiting for an answer that is "
            "already in your hands as a tool call. Produce the actual call — never a "
            "description of the call, never example JSON, never an offer to help. If you "
            "were about to write instructions about calling a tool, call it instead."
            + recall_line
        )

    if provider == "offline":
        blocks.append(
            "NOTE: no model provider is configured, so a deterministic rule-based planner "
            "is issuing the tool calls. Tool results are still the source of truth."
        )
    if extra:
        blocks.append(extra)
    return "\n\n".join(block.strip() for block in blocks if block and block.strip())


def _render_memories(memories: Sequence[Any]) -> str:
    if not memories:
        return ""
    lines = ["## What you already know (persistent memory, applies across sessions)"]
    for item in memories:
        record = getattr(item, "record", item)
        score = getattr(item, "score", None)
        suffix = f"  (relevance {score:.2f})" if isinstance(score, float) else ""
        lines.append(f"- [{record.id}] ({record.kind}) {record.text}{suffix}")
        if record.payload:
            lines.append(f"    structured form: {record.payload}")
        if record.keys:
            lines.append(f"    applies to: {', '.join(record.keys)}")
    lines.append(
        "Apply these. If one of them is wrong or outdated, correct it with `remember` "
        "(kind `correction`) instead of quietly ignoring it."
    )
    return "\n".join(lines)


MEMORY_EXTRACTION_PROMPT = """\
You maintain the long-term memory of a data analysis agent. Read the exchange below
and decide whether the USER established anything durable that should change how the
agent behaves in future sessions.

Durable (save it):
- a definition ("active customers means ..."), especially one that changes a filter
- a standing preference ("weekly, not daily", "always show percentages")
- a correction ("no, that column is gross not net")
- a reusable named filter or population

NOT durable (do not save it):
- one-off questions, values, or results; anything already derivable from the data
- restatements of the question
- your own analysis output

Reply with ONLY a JSON array. Each element:
{"text": "...", "kind": "definition|preference|correction|filter|fact", "keys": ["column_or_table"], "payload": {"sql": "optional predicate"}, "confidence": 0.0-1.0}
If nothing is durable, reply [].

EXCHANGE
------
User question: {{question}}

Tools used: {{tools}}

Agent answer: {{answer}}
------
JSON array:"""


def render_memory_prompt(question: str, tools: str, answer: str) -> str:
    """Fill the extraction prompt.

    Done with ``replace`` rather than ``str.format`` because the template embeds a
    literal JSON example, whose braces ``format`` would try to interpret.
    """
    return (
        MEMORY_EXTRACTION_PROMPT
        .replace("{{question}}", question)
        .replace("{{tools}}", tools)
        .replace("{{answer}}", answer)
    )
