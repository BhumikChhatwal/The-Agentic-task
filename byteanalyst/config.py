"""Runtime configuration, resolved from environment variables (and ``.env``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

try:  # python-dotenv is listed in requirements but degrade gracefully
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


PROVIDERS = ("openai_compatible", "anthropic", "offline")

DEFAULT_MODELS = {
    "openai_compatible": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "offline": "offline-rule-based-planner",
}

DEFAULT_BASE_URLS = {
    "openai_compatible": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def _env_flag(*names: str, default: bool = False) -> bool:
    raw = _env(*names)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - only on malformed env
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass
class Config:
    """Everything the agent needs that is not a dataset."""

    provider: str = "offline"
    api_key: str | None = None
    base_url: str = ""
    model: str = DEFAULT_MODELS["offline"]
    max_iterations: int = 12
    temperature: float = 0.0
    timeout: float = 120.0
    max_tool_output_chars: int = 6000
    max_preview_rows: int = 50
    memory_enabled: bool = True
    auto_memory: bool = True
    root: Path = field(default_factory=lambda: Path.cwd())
    state_dir: Path = field(default_factory=lambda: Path.cwd() / ".byteanalyst")

    # Derived paths -------------------------------------------------------
    @property
    def memory_path(self) -> Path:
        return self.state_dir / "memory.db"

    @property
    def artifact_dir(self) -> Path:
        return self.state_dir / "artifacts"

    @property
    def session_dir(self) -> Path:
        return self.state_dir / "sessions"

    def ensure_dirs(self) -> None:
        for path in (self.state_dir, self.artifact_dir, self.session_dir):
            path.mkdir(parents=True, exist_ok=True)

    # Construction --------------------------------------------------------
    @classmethod
    def from_env(cls, root: Path | None = None, **overrides) -> "Config":
        """Build a config from env vars, then apply explicit overrides.

        ``overrides`` are applied last so CLI flags win over the environment.
        """
        if load_dotenv is not None:
            root_dir = Path(root) if root else Path.cwd()
            load_dotenv(root_dir / ".env", override=False)

        provider = _env("BYTEANALYST_PROVIDER")
        api_key = _env("BYTEANALYST_API_KEY", "OPENAI_API_KEY")
        anthropic_key = _env("ANTHROPIC_API_KEY")

        if provider is None:
            if api_key:
                provider = "openai_compatible"
            elif anthropic_key:
                provider = "anthropic"
            else:
                provider = "offline"

        if provider not in PROVIDERS:
            raise ConfigError(
                f"unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}"
            )

        if provider == "anthropic":
            api_key = anthropic_key or api_key

        base_url = _env("BYTEANALYST_BASE_URL")
        if not base_url:
            base_url = DEFAULT_BASE_URLS.get(provider, "")

        model = _env("BYTEANALYST_MODEL") or DEFAULT_MODELS[provider]

        state_dir_raw = _env("BYTEANALYST_STATE_DIR")
        root_dir = Path(root) if root else Path.cwd()
        state_dir = Path(state_dir_raw) if state_dir_raw else root_dir / ".byteanalyst"

        config = cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_iterations=_env_int("BYTEANALYST_MAX_ITERATIONS", 12),
            temperature=_env_float("BYTEANALYST_TEMPERATURE", 0.0),
            timeout=_env_float("BYTEANALYST_TIMEOUT", 120.0),
            max_tool_output_chars=_env_int("BYTEANALYST_MAX_TOOL_OUTPUT_CHARS", 6000),
            max_preview_rows=_env_int("BYTEANALYST_MAX_PREVIEW_ROWS", 50),
            memory_enabled=_env_flag("BYTEANALYST_MEMORY", default=True),
            auto_memory=_env_flag("BYTEANALYST_AUTO_MEMORY", default=True),
            root=root_dir,
            state_dir=state_dir,
        )

        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(config, key):
                raise ConfigError(f"unknown config option {key!r}")
            setattr(config, key, value)

        # Re-validate after overrides: a caller-supplied provider must be real too.
        if config.provider not in PROVIDERS:
            raise ConfigError(
                f"unknown provider {config.provider!r}; expected one of {', '.join(PROVIDERS)}"
            )
        return config

    def needs_api_key(self) -> bool:
        return self.provider in {"openai_compatible", "anthropic"} and not self.api_key

    def describe(self) -> str:
        key_state = "set" if self.api_key else "not set"
        return (
            f"provider={self.provider} model={self.model} "
            f"base_url={self.base_url or '-'} api_key={key_state}"
        )
