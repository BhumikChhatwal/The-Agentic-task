"""Dispatch layer: specs for the model, safe execution for us."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..errors import ByteAnalystError
from ..formatting import ToolResult
from .base import Tool, ToolContext


@dataclass
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    result: ToolResult
    duration_ms: int
    exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result.to_dict(),
            "duration_ms": self.duration_ms,
            "exception": self.exception,
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or ():
            self.register(tool)

    # -- registration -----------------------------------------------------
    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("tool must declare a name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def catalogue(self) -> str:
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name} [{tool.category}]: {tool.description.splitlines()[0]}")
        return "\n".join(lines)

    # -- execution --------------------------------------------------------
    def execute(self, name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolExecution:
        started = time.perf_counter()
        tool = self._tools.get(name)
        arguments = dict(arguments or {})
        if tool is None:
            result = ToolResult.failure(
                f"unknown tool {name!r}; available tools: {', '.join(self.names())}", tool=name
            )
            return ToolExecution(name, arguments, result, _ms(started))
        try:
            result = tool.run(ctx, **arguments)
            if not isinstance(result, ToolResult):
                result = ToolResult.success(str(result), tool=name)
            result.tool = name
        except ByteAnalystError as exc:
            result = ToolResult.failure(str(exc), tool=name)
        except TypeError as exc:
            # Usually a bad/unknown keyword argument from the model.
            result = ToolResult.failure(
                f"bad arguments for {name}: {exc}. Tool schema: {tool.parameters}", tool=name
            )
        except Exception as exc:  # unexpected: keep the agent alive, surface the trace
            result = ToolResult.failure(
                f"{type(exc).__name__}: {exc}", tool=name
            )
            result.detail["traceback"] = traceback.format_exc(limit=6)
        for path in result.artifacts:
            if path not in ctx.artifacts:
                ctx.artifacts.append(path)
        return ToolExecution(name, arguments, result, _ms(started))


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def build_default_registry(include_memory: bool = True) -> ToolRegistry:
    """Every tool the agent ships with. Import order = order shown to the model."""
    from .anomalies import AnomalyTool
    from .memory_tools import ForgetMemoryTool, RecallMemoryTool, RememberTool
    from .profiling import ProfileTool
    from .python_exec import PythonExecTool
    from .sql import SqlTool
    from .stats_tests import StatisticalTestTool
    from .timeseries import TimeSeriesTool
    from .visualize import VisualizeTool

    tools: list[Tool] = [
        ProfileTool(),
        SqlTool(),
        PythonExecTool(),
        StatisticalTestTool(),
        AnomalyTool(),
        TimeSeriesTool(),
        VisualizeTool(),
    ]
    if include_memory:
        tools.extend([RememberTool(), RecallMemoryTool(), ForgetMemoryTool()])
    return ToolRegistry(tools)


__all__ = ["ToolRegistry", "ToolExecution", "build_default_registry", "ToolContext", "Tool"]
