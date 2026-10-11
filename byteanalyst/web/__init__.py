"""A small local web UI for demoing the agent.

Deliberately thin: one page, a handful of JSON endpoints, no build step. The
server holds one dataset registry and one agent, so the browser session behaves
like a single CLI session with a nicer transcript.
"""

from .app import create_app

__all__ = ["create_app"]
