"""Native Anthropic Messages API adapter.

Kept separate from the OpenAI-compatible adapter because the tool-calling wire
format differs: tools carry ``input_schema``, results come back as ``user``
messages containing ``tool_result`` blocks, and ``system`` is a top-level field.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import httpx

from ..errors import LLMError
from .base import AssistantTurn, Message, ToolCall

MAX_ATTEMPTS = 3
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class AnthropicClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "https://api.anthropic.com/v1").rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.last_usage: dict[str, Any] = {}

    def chat(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> AssistantTurn:
        system_parts = [m.content for m in messages if m.role == "system"]
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._encode(messages),
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = [
                {
                    "name": spec["function"]["name"],
                    "description": spec["function"]["description"],
                    "input_schema": spec["function"]["parameters"],
                }
                for spec in tools
            ]

        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        last_error = "no attempt made"
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    response = client.post(url, json=payload, headers=headers)
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(1.5 * attempt)
                        continue
                    raise LLMError(f"could not reach Anthropic at {url}: {last_error}") from exc
                if response.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    time.sleep(1.5 * attempt)
                    continue
                if response.status_code >= 400:
                    raise LLMError(
                        f"Anthropic returned HTTP {response.status_code}: {response.text[:500]}"
                    )
                data = response.json()
                return self._decode(data)
        raise LLMError(f"Anthropic failed after {MAX_ATTEMPTS} attempts: {last_error}")

    # -- translation -------------------------------------------------------
    @staticmethod
    def _encode(messages: Sequence[Message]) -> list[dict[str, Any]]:
        encoded: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "call",
                    "content": message.content,
                }
                if encoded and encoded[-1]["role"] == "user" and isinstance(encoded[-1]["content"], list):
                    encoded[-1]["content"].append(block)
                else:
                    encoded.append({"role": "user", "content": [block]})
            elif message.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    blocks.append(
                        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                    )
                encoded.append({"role": "assistant", "content": blocks or message.content})
            else:
                encoded.append({"role": "user", "content": message.content})
        return encoded

    def _decode(self, data: dict[str, Any]) -> AssistantTurn:
        if data.get("type") == "error":
            raise LLMError(f"Anthropic error: {data.get('error')}")
        self.last_usage = data.get("usage") or {}
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=str(block.get("id") or f"call_{len(calls)}"),
                        name=str(block.get("name") or ""),
                        arguments=dict(block.get("input") or {}),
                    )
                )
        return AssistantTurn(
            content="\n".join(texts).strip(),
            tool_calls=calls,
            usage=self.last_usage,
            model=data.get("model", self.model),
            finish_reason=str(data.get("stop_reason") or ""),
        )
