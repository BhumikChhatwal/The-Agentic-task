"""A client that replays a fixed script.

Lets the test suite exercise the *real* agent loop — tool dispatch, result
rendering, grounding verification, memory write-back — without a network call or a
paid endpoint. It is a test double, not a fallback provider: the shipping fallback
for a missing API key is the rule-based planner in ``offline.py``.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import AssistantTurn, Message, ToolCall


class ScriptedClient:
    def __init__(self, turns: Sequence[AssistantTurn | dict[str, Any]], model: str = "scripted") -> None:
        self.model = model
        self._turns: list[AssistantTurn] = [
            turn if isinstance(turn, AssistantTurn) else _coerce(turn) for turn in turns
        ]
        self.calls: list[list[Message]] = []

    def chat(self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None) -> AssistantTurn:
        self.calls.append(list(messages))
        if not self._turns:
            raise AssertionError(
                "ScriptedClient ran out of scripted turns (the agent asked for more than expected)"
            )
        return self._turns.pop(0)

    @staticmethod
    def tool(name: str, **arguments: Any) -> AssistantTurn:
        """Convenience: an assistant turn that calls ``name`` once."""
        return AssistantTurn(content="", tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=arguments)])

    @staticmethod
    def say(content: str) -> AssistantTurn:
        return AssistantTurn(content=content)


def _coerce(spec: dict[str, Any]) -> AssistantTurn:
    if "tool" in spec:
        return ScriptedClient.tool(spec["tool"], **(spec.get("arguments") or {}))
    return ScriptedClient.say(spec.get("content", ""))
