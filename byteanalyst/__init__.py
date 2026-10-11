"""Byte Analyst — an LLM agent that analyses arbitrary tabular datasets.

The package is organised as four layers:

* ``byteanalyst.datasets``  — loading arbitrary tabular files and describing them.
* ``byteanalyst.tools``     — the analysis tools the agent is allowed to call.
* ``byteanalyst.memory``    — persistent, retrievable session-to-session memory.
* ``byteanalyst.agent``     — the tool-calling loop that ties the above together.

``byteanalyst.llm`` contains the provider adapters (OpenAI-compatible, Anthropic,
and an offline deterministic planner used for tests and keyless demos).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
