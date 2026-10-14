"""Adapter for every OpenAI-compatible /chat/completions endpoint.

That single adapter covers OpenAI, Azure-style gateways, OpenRouter, Groq,
Together, Fireworks, vLLM, LM Studio, llama.cpp's server and Ollama — so the user
is never locked to one vendor. Only ``base_url``, ``model`` and ``api_key`` change.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import httpx

from ..errors import LLMError
from .base import AssistantTurn, Message, ToolCall, parse_tool_arguments

RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3


def _is_connection_refused(exc: BaseException) -> bool:
    """True when the failure is a refused TCP connection (server not listening).

    A refusal is a definitive answer — retrying only multiplies latency (the OS
    itself takes ~2s per refused loopback connect on Windows) — so it is
    distinguished from timeouts and resets, which are transient. Windows reports
    it as WinError 10061, POSIX as errno 111.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ConnectionRefusedError):
            return True
        text = str(current).lower()
        if "10061" in text or "errno 111" in text or "actively refused" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def _connection_hint(url: str) -> str:
    """Turn the most common connection failures into an actionable next step.

    A refused connection to localhost:11434 means the local model server
    (Ollama, LM Studio, …) simply is not running — a different fix from a wrong
    base URL or a flaky network.
    """
    local = ":11434" in url or "localhost" in url or "127.0.0.1" in url
    if local:
        return (
            "The local model server is not running. Start it (for Ollama: `ollama serve`, "
            "or launch the Ollama/LM Studio app) and confirm the model is pulled, "
            "e.g. `ollama pull qwen2.5:7b`. If you meant to run without a model, "
            "add `--provider offline`."
        )
    return "Check BYTEANALYST_BASE_URL and your network."


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}
        self.last_usage: dict[str, Any] = {}

    # -- wire helpers ------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter likes these for attribution; harmless elsewhere.
        headers.setdefault("HTTP-Referer", "https://github.com/byte-analyst")
        headers.setdefault("X-Title", "Byte Analyst")
        return headers

    @staticmethod
    def _encode_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        encoded: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                encoded.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "call",
                        "content": message.content,
                    }
                )
            elif message.role == "assistant" and message.tool_calls:
                encoded.append(
                    {
                        "role": "assistant",
                        "content": message.content or None,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.raw_arguments
                                    or _dumps(call.arguments),
                                },
                            }
                            for call in message.tool_calls
                        ],
                    }
                )
            else:
                encoded.append({"role": message.role, "content": message.content})
        return encoded

    # -- main call ---------------------------------------------------------
    def chat(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._encode_messages(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"
        last_error = "no attempt made"
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    response = client.post(url, json=payload, headers=self._headers())
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if _is_connection_refused(exc):
                        # Nothing is listening: retrying a refused connection
                        # only multiplies the OS refusal latency without changing
                        # the outcome. Fail fast with an actionable message.
                        raise LLMError(
                            f"could not reach the model endpoint at {url}: {last_error}. "
                            + _connection_hint(url)
                        ) from exc
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(1.5 * attempt)
                        continue
                    raise LLMError(
                        f"could not reach the model endpoint at {url}: {last_error}. "
                        + _connection_hint(url)
                    ) from exc

                if response.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    time.sleep(1.5 * attempt)
                    continue
                if response.status_code >= 400:
                    raise LLMError(
                        f"model endpoint returned HTTP {response.status_code}: {response.text[:500]}"
                    )
                try:
                    data = response.json()
                except ValueError as exc:
                    raise LLMError(f"model endpoint returned non-JSON: {response.text[:300]}") from exc
                return self._decode(data)

        raise LLMError(f"model endpoint failed after {MAX_ATTEMPTS} attempts: {last_error}")

    # -- decoding ----------------------------------------------------------
    def _decode(self, data: dict[str, Any]) -> AssistantTurn:
        if "error" in data and data.get("error"):
            raise LLMError(f"provider error: {data['error']}")
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"provider returned no choices: {str(data)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        self.last_usage = data.get("usage") or {}

        calls: list[ToolCall] = []
        for index, raw_call in enumerate(message.get("tool_calls") or []):
            function = raw_call.get("function") or {}
            arguments, raw_text = parse_tool_arguments(function.get("arguments"))
            calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or f"call_{index}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                    raw_arguments=raw_text,
                )
            )
        content = message.get("content") or ""
        if isinstance(content, list):  # some gateways return content blocks
            content = "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            ).strip()
        return AssistantTurn(
            content=content,
            tool_calls=[c for c in calls if c.name],
            usage=self.last_usage,
            model=data.get("model", self.model),
            finish_reason=str(choice.get("finish_reason") or ""),
        )


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value or {})
