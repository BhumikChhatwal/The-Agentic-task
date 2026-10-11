"""Exception hierarchy.

Tools never raise into the agent loop: the executor converts any exception into a
``ToolResult`` with ``ok=False`` so the model can see the failure and recover.
These types exist so that *expected* failures carry a readable message.
"""

from __future__ import annotations


class ByteAnalystError(Exception):
    """Base class for all Byte Analyst errors."""


class ConfigError(ByteAnalystError):
    """Bad or missing configuration (provider, model, paths)."""


class LLMError(ByteAnalystError):
    """The model provider returned an error or an unusable response."""


class DatasetError(ByteAnalystError):
    """A dataset could not be loaded, found, or interpreted.

    The message is written to be actionable for the model, e.g.
    ``unknown dataset 'sales'; loaded datasets: orders, customers``.
    """


class ToolError(ByteAnalystError):
    """A tool was called with arguments it cannot satisfy."""


class MemoryError_(ByteAnalystError):
    """Memory store failure (kept distinct from the builtin ``MemoryError``)."""
