"""The analysis tools the agent can call.

Seven genuinely different kinds of analysis (profiling, relational querying,
general computation, inferential statistics, anomaly detection, visualisation and
time-series analysis) plus three tools that read and write persistent memory.
"""

from __future__ import annotations

from .base import Tool, ToolContext
from .registry import ToolRegistry, build_default_registry

__all__ = ["Tool", "ToolContext", "ToolRegistry", "build_default_registry"]
