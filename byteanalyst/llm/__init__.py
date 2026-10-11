"""Provider adapters.

``build_client`` turns a :class:`~byteanalyst.config.Config` into a concrete
client. The ``offline`` provider is not a client at all — it is a deterministic
rule-based planner (see :mod:`byteanalyst.llm.offline`) that the agent uses when
no API key is configured, so the repository is runnable and testable out of the
box without any third-party service.
"""

from __future__ import annotations

from ..config import Config
from ..errors import ConfigError
from .base import AssistantTurn, LLMClient, Message, ToolCall, parse_tool_arguments

__all__ = [
    "AssistantTurn",
    "LLMClient",
    "Message",
    "ToolCall",
    "parse_tool_arguments",
    "build_client",
]


def build_client(config: Config):
    if config.provider == "offline":
        raise ConfigError(
            "provider 'offline' has no HTTP client; the agent uses the rule-based planner instead"
        )
    if config.provider == "anthropic":
        from .anthropic import AnthropicClient

        if not config.api_key:
            raise ConfigError(
                "anthropic provider requires ANTHROPIC_API_KEY (or BYTEANALYST_API_KEY)"
            )
        return AnthropicClient(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            timeout=config.timeout,
        )
    if config.provider == "openai_compatible":
        from .openai_compat import OpenAICompatibleClient

        # A key is optional: local servers (Ollama, LM Studio, vLLM) need none.
        if not config.api_key and "localhost" not in config.base_url and "127.0.0.1" not in config.base_url:
            raise ConfigError(
                "no API key found. Set BYTEANALYST_API_KEY (or OPENAI_API_KEY) for a hosted "
                "endpoint, point BYTEANALYST_BASE_URL at a local model server, or run with "
                "--provider offline to use the built-in rule-based planner."
            )
        return OpenAICompatibleClient(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            timeout=config.timeout,
        )
    raise ConfigError(f"unknown provider {config.provider!r}")
