"""Shared fixtures.

Everything is seeded and self-contained: no network, no writes outside ``tmp_path``
and no dependence on the developer's environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from byteanalyst.config import Config
from byteanalyst.datasets import DatasetRegistry
from byteanalyst.memory import MemoryStore
from byteanalyst.tools import ToolContext, build_default_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(autouse=True)
def _scrub_ambient_llm_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's ``.env`` / shell exports out of the tests.

    ``Config.from_env`` loads ``.env`` via python-dotenv, which puts values into
    ``os.environ``; without this, a local ``.env`` (e.g. one pointing at Ollama)
    would silently change what later tests see.
    """
    for name in list(os.environ):
        if name.startswith("BYTEANALYST_") or name in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config.from_env(root=tmp_path)
    cfg.provider = "offline"
    cfg.api_key = None
    cfg.state_dir = tmp_path / ".byteanalyst"
    cfg.max_preview_rows = 25
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def orders() -> pd.DataFrame:
    """Daily orders with a trend, a weekly cycle, a planted outlier and nulls."""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2023-01-01", periods=240, freq="D")
    region = rng.choice(["North", "South", "East"], 240, p=[0.45, 0.35, 0.20])
    weekday = np.array([0.9, 0.95, 1.0, 1.05, 1.2, 1.35, 1.05])[dates.dayofweek]
    revenue = (80 + np.arange(240) * 0.5) * weekday + (region == "North") * 60 + rng.normal(0, 15, 240)
    revenue = revenue.round(2)
    revenue[100] = round(float(revenue[100] + 900), 2)  # planted outlier
    satisfaction = np.where(rng.random(240) < 0.25, np.nan, rng.choice([1, 2, 3, 4, 5], 240))
    return pd.DataFrame(
        {
            "order_id": [f"O{i:04d}" for i in range(240)],
            "order_date": dates,
            "customer_id": [f"C{i % 40:03d}" for i in rng.permutation(240)],
            "region": region,
            "channel": rng.choice(["Web", "Store"], 240, p=[0.7, 0.3]),
            "status": rng.choice(["completed", "cancelled"], 240, p=[0.85, 0.15]),
            "quantity": rng.integers(1, 8, 240),
            "revenue": revenue,
            "satisfaction": satisfaction,
        }
    )


@pytest.fixture
def customers() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    segment = rng.choice(["Consumer", "SMB", "Enterprise"], 40, p=[0.6, 0.3, 0.1])
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(40)],
            "segment": segment,
            "plan": rng.choice(["Basic", "Plus", "Pro"], 40),
            "country": rng.choice(["US", "DE", "IN"], 40),
        }
    )


@pytest.fixture
def registry(orders: pd.DataFrame, customers: pd.DataFrame) -> DatasetRegistry:
    reg = DatasetRegistry()
    reg.add("orders", orders, "fixture")
    reg.add("customers", customers, "fixture")
    return reg


@pytest.fixture
def memory() -> MemoryStore:
    store = MemoryStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def ctx(registry: DatasetRegistry, config: Config, memory: MemoryStore) -> ToolContext:
    return ToolContext(registry=registry, config=config, memory=memory)


@pytest.fixture
def tools():
    return build_default_registry()


def run_tool(tools, ctx, name: str, **arguments):
    """Execute a tool and return its ToolResult (asserting nothing about success)."""
    return tools.execute(name, arguments, ctx).result
