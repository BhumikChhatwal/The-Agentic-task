"""FastAPI backend for the demo UI.

Run with ``byteanalyst serve --data data/ecommerce_orders.csv`` and open
http://127.0.0.1:8000.

Design notes:
* One :class:`~byteanalyst.agent.Agent` instance behind a lock. The agent keeps a
  conversation history and a mutable tool context, so serialising requests is the
  honest choice for a single-user local demo.
* Charts rendered by the ``visualize`` tool are returned inline as base64 data
  URIs, so the page never has to guess a filesystem path.
* Uploaded files land in the state directory (``.byteanalyst/uploads``), which is
  git-ignored; nothing is written anywhere else.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import threading
import uuid
from pathlib import Path
from typing import Any, Sequence

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ..agent import Agent, AgentEvent
from ..config import Config
from ..datasets import DatasetRegistry
from ..errors import ByteAnalystError
from ..memory import KINDS, MemoryStore
from ..tools import build_default_registry

STATIC = Path(__file__).parent / "static"
SAMPLES = Path(__file__).resolve().parents[2] / "data"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_INLINE_ARTIFACTS = 4


class AskRequest(BaseModel):
    question: str
    use_memory: bool = True


class MemoryRequest(BaseModel):
    text: str
    kind: str = "fact"
    keys: list[str] = []


class IdRequest(BaseModel):
    id: str


class Session:
    """Mutable server state for one workspace."""

    def __init__(self, config: Config, datasets: Sequence[str | Path] = ()) -> None:
        self.config = config
        self.config.ensure_dirs()
        self.lock = threading.Lock()
        self.registry = DatasetRegistry()
        self.memory = MemoryStore(config.memory_path)
        self.agent: Agent | None = None
        self.uploads_dir = config.state_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        for path in datasets:
            self.load_path(Path(path))

    # -- datasets ---------------------------------------------------------
    def load_path(self, path: Path) -> str:
        if not path.exists():
            raise ByteAnalystError(f"file not found: {path}")
        dataset = self.registry.load_file(path)
        self.agent = None  # rebuild so the prompt picks up the new schema
        return dataset.name

    def load_sample(self, name: str) -> str:
        candidate = SAMPLES / name
        if not candidate.exists():
            raise ByteAnalystError(f"unknown sample dataset {name!r}")
        return self.load_path(candidate)

    def save_upload(self, filename: str, payload: bytes) -> str:
        safe = Path(filename).name
        if not safe:
            raise ByteAnalystError("uploaded file needs a name")
        target = self.uploads_dir / f"{uuid.uuid4().hex[:6]}-{safe}"
        target.write_bytes(payload)
        return self.load_path(target)

    def remove(self, name: str) -> bool:
        removed = self.registry.remove(name)
        if removed:
            self.agent = None
        return removed

    def describe(self) -> list[dict[str, Any]]:
        out = []
        for dataset in self.registry.datasets():
            roles: dict[str, int] = {}
            for role in dataset.roles().values():
                roles[role] = roles.get(role, 0) + 1
            out.append(
                {
                    "name": dataset.name,
                    "rows": dataset.row_count,
                    "columns": dataset.columns,
                    "source": dataset.source,
                    "roles": roles,
                }
            )
        return out

    # -- agent ------------------------------------------------------------
    def get_agent(self) -> Agent:
        if self.agent is None:
            self.agent = Agent(self.config, self.registry, memory=self.memory)
        return self.agent

    def reset_agent(self, drop_memory: bool = False) -> None:
        if self.agent is not None:
            self.agent.reset()
        if drop_memory:
            self.memory.clear(hard=False)

    def ask(self, question: str, use_memory: bool = True) -> dict[str, Any]:
        if not self.registry.names():
            raise ByteAnalystError("no datasets loaded — add one first")
        with self.lock:
            agent = self.get_agent()
            events: list[dict[str, Any]] = []
            answer = agent.ask(
                question,
                on_event=lambda event: events.append(event.to_dict()),
                use_memory=use_memory,
            )
            payload = answer.to_dict()
            payload["trace"] = [event for event in events if event["type"] != "answer"]
            payload["artifacts_inline"] = _inline_artifacts(answer.artifacts)
            payload["memory_stats"] = self.memory.stats()
            return payload

    def close(self) -> None:
        self.memory.close()


def _inline_artifacts(paths: Sequence[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for path in paths[:MAX_INLINE_ARTIFACTS]:
        target = Path(path)
        try:
            payload = target.read_bytes()
        except OSError:
            continue
        mime = mimetypes.guess_type(target.name)[0] or "image/png"
        out.append(
            {
                "name": target.name,
                "data_uri": f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}",
            }
        )
    return out


def create_app(config: Config | None = None, datasets: Sequence[str | Path] = ()) -> FastAPI:
    config = config or Config.from_env()
    session = Session(config, datasets)
    app = FastAPI(title="Byte Analyst", version="0.1.0")
    app.state.session = session

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return {
            "provider": config.provider,
            "model": session.agent.model_name if session.agent else config.model,
            "datasets": session.describe(),
            "samples": sorted(p.name for p in SAMPLES.glob("*.csv")),
            "tools": [
                {"name": tool.name, "category": tool.category,
                 "description": tool.description.splitlines()[0]}
                for tool in build_default_registry().tools()
            ],
            "memory": session.memory.stats(),
            "memory_enabled": config.memory_enabled,
        }

    @app.post("/api/sample")
    def add_sample(body: dict[str, str]) -> dict[str, Any]:
        try:
            name = session.load_sample(body.get("name", ""))
        except ByteAnalystError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"loaded": name, "datasets": session.describe()}

    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        loaded: list[str] = []
        for upload_file in files:
            payload = await upload_file.read()
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"{upload_file.filename} is too large")
            try:
                loaded.append(session.save_upload(upload_file.filename or "upload.csv", payload))
            except ByteAnalystError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"loaded": loaded, "datasets": session.describe()}

    @app.post("/api/remove")
    def remove(body: dict[str, str]) -> dict[str, Any]:
        removed = session.remove(body.get("name", ""))
        return {"removed": removed, "datasets": session.describe()}

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")
        try:
            return session.ask(request.question, request.use_memory)
        except ByteAnalystError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/reset")
    def reset(body: dict[str, Any] | None = None) -> dict[str, Any]:
        session.reset_agent(drop_memory=bool((body or {}).get("memory")))
        return {"ok": True, "memory": session.memory.stats()}

    @app.get("/api/memory")
    def memory_list() -> dict[str, Any]:
        return {
            "stats": session.memory.stats(),
            "records": [record.to_dict() for record in session.memory.list(include_inactive=False)],
            "kinds": list(KINDS),
        }

    @app.post("/api/memory/add")
    def memory_add(request: MemoryRequest) -> dict[str, Any]:
        try:
            outcome = session.memory.save(
                text=request.text, kind=request.kind, keys=request.keys, source="user"
            )
        except ByteAnalystError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "action": outcome.action,
            "record": outcome.record.to_dict(),
            "superseded": [item.to_dict() for item in outcome.superseded],
        }

    @app.post("/api/memory/forget")
    def memory_forget(request: IdRequest) -> dict[str, Any]:
        return {"forgotten": session.memory.forget(request.id)}

    @app.exception_handler(ByteAnalystError)
    def handle_error(request, exc: ByteAnalystError):  # pragma: no cover - trivial
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


def main() -> None:  # pragma: no cover - convenience entry point
    import uvicorn

    config = Config.from_env()
    app = create_app(config)
    uvicorn.run(app, host="127.0.0.1", port=8000)


__all__ = ["create_app", "Session"]
