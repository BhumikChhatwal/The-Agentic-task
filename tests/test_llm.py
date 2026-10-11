"""Provider adapters and configuration.

No network is used: the OpenAI-compatible client is exercised against a fake
``httpx.Client``, and the Anthropic adapter's translation is tested directly.
"""

from __future__ import annotations

import json

import pytest

from byteanalyst.config import Config
from byteanalyst.errors import ConfigError, LLMError
from byteanalyst.llm import build_client, parse_tool_arguments
from byteanalyst.llm.anthropic import AnthropicClient
from byteanalyst.llm.base import Message, ToolCall, parse_json_array, parse_memory_json
from byteanalyst.llm.openai_compat import OpenAICompatibleClient
from byteanalyst.llm.scripted import ScriptedClient


# ---------------------------------------------------------------------------
# fake transport
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload)


class FakeHttpxClient:
    def __init__(self, responses, recorder) -> None:
        self._responses = list(responses)
        self._recorder = recorder

    def __enter__(self) -> "FakeHttpxClient":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def post(self, url, json=None, headers=None):
        self._recorder.append({"url": url, "json": json, "headers": headers})
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


@pytest.fixture
def fake_http(monkeypatch):
    """Patch httpx.Client inside the openai_compat module."""
    import byteanalyst.llm.openai_compat as module

    recorder: list[dict] = []
    state: dict = {"responses": []}

    def factory(*args, **kwargs):
        return FakeHttpxClient(state["responses"], recorder)

    monkeypatch.setattr(module.httpx, "Client", factory)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    state["recorder"] = recorder
    return state


def _completion(content=None, tool_calls=None, usage=None):
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _tool_call(name, arguments: str, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


def test_openai_decodes_text_and_tool_calls(fake_http) -> None:
    fake_http["responses"] = [
        FakeResponse(
            200,
            _completion(
                content="Working on it.",
                tool_calls=[_tool_call("run_sql", '{"query": "select 1"}')],
            ),
        )
    ]
    client = OpenAICompatibleClient(model="test-model", api_key="sk-test")
    turn = client.chat([Message(role="user", content="hi")], tools=[{"type": "function", "function": {"name": "run_sql"}}])
    assert turn.wants_tools
    assert turn.tool_calls[0].name == "run_sql"
    assert turn.tool_calls[0].arguments == {"query": "select 1"}
    assert turn.content == "Working on it."
    assert turn.usage["prompt_tokens"] == 10

    sent = fake_http["recorder"][0]
    assert sent["url"].endswith("/chat/completions")
    assert sent["headers"]["Authorization"] == "Bearer sk-test"
    assert sent["json"]["tools"][0]["function"]["name"] == "run_sql"


def test_openai_encodes_assistant_tool_calls_and_results(fake_http) -> None:
    fake_http["responses"] = [FakeResponse(200, _completion(content="done"))]
    client = OpenAICompatibleClient(model="m", api_key="k")
    messages = [
        Message(role="system", content="sys"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="run_sql", arguments={"query": "select 1"})],
        ),
        Message(role="tool", content="42 rows", tool_call_id="c1", name="run_sql"),
    ]
    client.chat(messages)
    encoded = fake_http["recorder"][0]["json"]["messages"]
    assert encoded[1]["tool_calls"][0]["function"]["arguments"] == '{"query": "select 1"}'
    assert encoded[2] == {"role": "tool", "tool_call_id": "c1", "content": "42 rows"}


def test_openai_handles_malformed_tool_arguments(fake_http) -> None:
    fake_http["responses"] = [
        FakeResponse(200, _completion(tool_calls=[_tool_call("run_sql", "{query: 'x'}")]))
    ]
    client = OpenAICompatibleClient(model="m", api_key="k")
    turn = client.chat([Message(role="user", content="hi")])
    # Repairable near-JSON is fixed; the raw text is preserved either way.
    assert turn.tool_calls[0].arguments == {"query": "x"}
    assert turn.tool_calls[0].raw_arguments

    fake_http["responses"] = [FakeResponse(200, _completion(tool_calls=[_tool_call("run_sql", "not json at all")]))]
    client2 = OpenAICompatibleClient(model="m", api_key="k")
    turn2 = client2.chat([Message(role="user", content="hi")])
    assert turn2.tool_calls[0].arguments == {}
    assert turn2.tool_calls[0].raw_arguments == "not json at all"


def test_openai_retries_transient_status(fake_http) -> None:
    fake_http["responses"] = [
        FakeResponse(429, {"error": "rate limited"}),
        FakeResponse(200, _completion(content="recovered")),
    ]
    client = OpenAICompatibleClient(model="m", api_key="k")
    turn = client.chat([Message(role="user", content="hi")])
    assert turn.content == "recovered"
    assert len(fake_http["recorder"]) == 2


def test_openai_raises_on_client_error(fake_http) -> None:
    fake_http["responses"] = [FakeResponse(400, {"error": {"message": "bad model"}})]
    client = OpenAICompatibleClient(model="nope", api_key="k")
    with pytest.raises(LLMError, match="400"):
        client.chat([Message(role="user", content="hi")])


def test_openai_raises_on_provider_error_payload(fake_http) -> None:
    fake_http["responses"] = [FakeResponse(200, {"error": {"message": "context too long"}})]
    client = OpenAICompatibleClient(model="m", api_key="k")
    with pytest.raises(LLMError, match="context too long"):
        client.chat([Message(role="user", content="hi")])


def test_openai_local_endpoint_needs_no_key() -> None:
    client = OpenAICompatibleClient(model="llama3", base_url="http://localhost:11434/v1")
    assert "Authorization" not in client._headers()
    assert client.base_url == "http://localhost:11434/v1"


def test_parse_tool_arguments_variants() -> None:
    assert parse_tool_arguments({"a": 1})[0] == {"a": 1}
    assert parse_tool_arguments("")[0] == {}
    assert parse_tool_arguments('{"a": 1}')[0] == {"a": 1}
    assert parse_tool_arguments("[1, 2]")[0] == {"value": [1, 2]}


# ---------------------------------------------------------------------------
# Anthropic translation
# ---------------------------------------------------------------------------


def test_anthropic_encodes_system_tools_and_results() -> None:
    messages = [
        Message(role="system", content="be careful"),
        Message(role="user", content="question"),
        Message(role="assistant", content="thinking", tool_calls=[ToolCall(id="t1", name="run_sql", arguments={"query": "q"})]),
        Message(role="tool", content="result text", tool_call_id="t1", name="run_sql"),
        Message(role="tool", content="second result", tool_call_id="t2", name="profile_dataset"),
    ]
    encoded = AnthropicClient._encode(messages)
    assert [m["role"] for m in encoded] == ["user", "assistant", "user"]
    assistant_blocks = encoded[1]["content"]
    assert {"type": "text", "text": "thinking"} in assistant_blocks
    assert {"type": "tool_use", "id": "t1", "name": "run_sql", "input": {"query": "q"}} in assistant_blocks
    # Consecutive tool results merge into one user turn.
    results = encoded[2]["content"]
    assert [block["type"] for block in results] == ["tool_result", "tool_result"]
    assert results[1]["tool_use_id"] == "t2"


def test_anthropic_decodes_text_and_tool_use() -> None:
    client = AnthropicClient(model="claude", api_key="k")
    turn = client._decode(
        {
            "model": "claude-3-5-sonnet",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 3},
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tu1", "name": "run_sql", "input": {"query": "select 1"}},
            ],
        }
    )
    assert turn.content == "Let me check."
    assert turn.tool_calls[0].arguments == {"query": "select 1"}
    assert turn.finish_reason == "tool_use"
    assert turn.usage["input_tokens"] == 3


def test_anthropic_error_payload_raises() -> None:
    client = AnthropicClient(model="claude", api_key="k")
    with pytest.raises(LLMError):
        client._decode({"type": "error", "error": {"message": "overloaded"}})


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def test_parse_json_array_handles_fences_and_prose() -> None:
    assert parse_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_json_array('Sure! Here you go: [1, 2, 3] hope that helps') == [1, 2, 3]
    assert parse_json_array('{"memories": [{"a": 1}]}') == [{"a": 1}]
    assert parse_json_array("") == []
    assert parse_json_array("nonsense") == []


def test_parse_memory_json_validates_each_candidate() -> None:
    payload = json.dumps(
        [
            {"text": "Active excludes cancelled accounts", "kind": "filter",
             "keys": "status", "payload": {"sql": "status <> 'cancelled'"}, "confidence": 0.9},
            {"text": "short", "kind": "made-up"},          # too short -> dropped
            {"text": "A perfectly good durable fact", "kind": "weird-kind"},  # kind coerced
            "not a dict",                                    # dropped
        ]
    )
    parsed = parse_memory_json(payload)
    assert len(parsed) == 2
    assert parsed[0]["keys"] == ["status"]
    assert parsed[0]["payload"]["sql"] == "status <> 'cancelled'"
    assert parsed[1]["kind"] == "fact"


def test_parse_memory_json_bounds_the_result() -> None:
    many = [{"text": f"durable fact number {i}", "kind": "fact"} for i in range(20)]
    assert len(parse_memory_json(json.dumps(many))) == 5


# ---------------------------------------------------------------------------
# config + factory
# ---------------------------------------------------------------------------


def test_config_autodetects_provider(monkeypatch, tmp_path) -> None:
    for name in ("BYTEANALYST_PROVIDER", "BYTEANALYST_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "BYTEANALYST_BASE_URL", "BYTEANALYST_MODEL"):
        monkeypatch.delenv(name, raising=False)

    offline = Config.from_env(root=tmp_path)
    assert offline.provider == "offline"
    assert offline.needs_api_key() is False

    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    openai_cfg = Config.from_env(root=tmp_path)
    assert openai_cfg.provider == "openai_compatible"
    assert openai_cfg.api_key == "sk-x"
    assert openai_cfg.base_url.endswith("api.openai.com/v1")

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    anthropic_cfg = Config.from_env(root=tmp_path)
    assert anthropic_cfg.provider == "anthropic"
    assert anthropic_cfg.model.startswith("claude")


def test_config_overrides_and_derived_paths(tmp_path) -> None:
    config = Config.from_env(root=tmp_path, provider="openai_compatible", model="custom", max_iterations=3)
    assert config.provider == "openai_compatible"
    assert config.model == "custom"
    assert config.max_iterations == 3
    assert config.memory_path == config.state_dir / "memory.db"
    assert config.artifact_dir.name == "artifacts"

    with pytest.raises(ConfigError):
        Config.from_env(root=tmp_path, nonsense=1)


def test_config_rejects_unknown_provider(tmp_path) -> None:
    with pytest.raises(ConfigError):
        Config.from_env(root=tmp_path, provider="telepathy")


def test_build_client_rules(tmp_path) -> None:
    with pytest.raises(ConfigError, match="offline"):
        build_client(Config.from_env(root=tmp_path))

    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        build_client(Config(provider="anthropic", api_key=None))

    with pytest.raises(ConfigError, match="no API key"):
        build_client(Config(provider="openai_compatible", api_key=None, base_url="https://api.openai.com/v1"))

    local = build_client(Config(provider="openai_compatible", api_key=None, base_url="http://localhost:1234/v1"))
    assert local.base_url == "http://localhost:1234/v1"

    hosted = build_client(Config(provider="openai_compatible", api_key="sk-1", model="m"))
    assert hosted.model == "m"

    anthropic = build_client(Config(provider="anthropic", api_key="sk-ant", model="claude"))
    assert isinstance(anthropic, AnthropicClient)


# ---------------------------------------------------------------------------
# scripted test double
# ---------------------------------------------------------------------------


def test_scripted_client_replays_and_records() -> None:
    client = ScriptedClient([{"tool": "run_sql", "arguments": {"query": "q"}}, {"content": "done"}])
    first = client.chat([Message(role="user", content="hi")])
    assert first.tool_calls[0].name == "run_sql"
    assert client.chat([]).content == "done"
    assert len(client.calls) == 2
    with pytest.raises(AssertionError):
        client.chat([])
