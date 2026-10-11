"""The demo web UI, exercised through FastAPI's test client (no server needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from byteanalyst.web.app import Session, create_app  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def client(config):
    app = create_app(config)
    with TestClient(app) as test_client:
        yield test_client
    app.state.session.close()


@pytest.fixture
def client_with_data(config, tmp_path):
    csv = tmp_path / "sales.csv"
    csv.write_text(
        "day,region,revenue\n"
        "2024-01-01,North,100\n2024-01-02,South,200\n2024-01-03,North,150\n",
        encoding="utf-8",
    )
    app = create_app(config, [csv])
    with TestClient(app) as test_client:
        yield test_client, csv
    app.state.session.close()


def test_index_renders_the_page(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Byte Analyst" in response.text
    assert "/api/ask" in response.text


def test_state_reports_tools_and_samples(client) -> None:
    payload = client.get("/api/state").json()
    assert payload["provider"] == "offline"
    names = {tool["name"] for tool in payload["tools"]}
    assert "run_sql" in names and "visualize" in names
    assert "ecommerce_orders.csv" in payload["samples"]
    assert payload["datasets"] == []


def test_sample_dataset_can_be_loaded_and_described(client) -> None:
    response = client.post("/api/sample", json={"name": "employees.csv"})
    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert datasets[0]["name"] == "employees"
    assert datasets[0]["rows"] == 320
    assert "salary" in datasets[0]["columns"]


def test_unknown_sample_is_a_400(client) -> None:
    response = client.post("/api/sample", json={"name": "nope.csv"})
    assert response.status_code == 400
    assert "unknown sample" in response.json()["detail"]


def test_ask_returns_answer_trace_and_grounding(client_with_data) -> None:
    client, _ = client_with_data
    response = client.post("/api/ask", json={"question": "total revenue by region"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["tools_used"] == ["run_sql"]
    assert payload["grounded"] is True
    assert payload["memory_stats"]["active"] == 0

    kinds = [event["type"] for event in payload["trace"]]
    assert kinds[0] == "start"
    assert "tool_call" in kinds and "tool_result" in kinds and "verification" in kinds
    call = next(event for event in payload["trace"] if event["type"] == "tool_call")
    assert call["name"] == "run_sql"
    assert "select" in call["arguments"]["query"].lower()


def test_ask_without_datasets_is_a_400(client) -> None:
    response = client.post("/api/ask", json={"question": "anything"})
    assert response.status_code == 400
    assert "no datasets" in response.json()["detail"]


def test_ask_rejects_an_empty_question(client_with_data) -> None:
    client, _ = client_with_data
    response = client.post("/api/ask", json={"question": "   "})
    assert response.status_code == 400


def test_charts_come_back_inline(client_with_data) -> None:
    client, _ = client_with_data
    payload = client.post(
        "/api/ask", json={"question": "plot the distribution of revenue"}
    ).json()
    assert payload["tools_used"] == ["visualize"]
    assert payload["artifacts_inline"]
    assert payload["artifacts_inline"][0]["data_uri"].startswith("data:image/png;base64,")


def test_upload_csv_then_analyse(client, tmp_path) -> None:
    source = tmp_path / "blob.csv"
    source.write_text("k,v\na,1\nb,2\nc,3\n", encoding="utf-8")
    with source.open("rb") as handle:
        response = client.post("/api/upload", files={"files": ("blob.csv", handle, "text/csv")})
    assert response.status_code == 200
    assert response.json()["loaded"]

    answer = client.post("/api/ask", json={"question": "how many rows are there"}).json()
    assert answer["ok"] and answer["tools_used"] == ["run_sql"]


def test_memory_endpoints_round_trip(client_with_data) -> None:
    client, _ = client_with_data
    created = client.post(
        "/api/memory/add",
        json={"text": "Active customers exclude cancelled accounts", "kind": "filter", "keys": ["status"]},
    ).json()
    assert created["action"] == "created"
    memory_id = created["record"]["id"]

    listing = client.get("/api/memory").json()
    assert listing["stats"]["active"] == 1
    assert listing["records"][0]["kind"] == "filter"
    assert "filter" in listing["kinds"]

    # A memory written in the browser changes the next answer.
    answer = client.post("/api/ask", json={"question": "how many active customers are there by region"}).json()
    assert any(item["id"] == memory_id for item in answer["memories_used"]) or answer["tools_used"]

    forgotten = client.post("/api/memory/forget", json={"id": memory_id}).json()
    assert forgotten["forgotten"] is True
    assert client.get("/api/memory").json()["stats"]["active"] == 0


def test_memory_can_be_disabled_per_request(client_with_data) -> None:
    client, _ = client_with_data
    client.post("/api/memory/add", json={"text": "Some durable rule about revenue", "kind": "definition"})
    payload = client.post(
        "/api/ask", json={"question": "total revenue by region", "use_memory": False}
    ).json()
    assert payload["memories_used"] == []


def test_dataset_removal(client_with_data) -> None:
    client, _ = client_with_data
    name = client.get("/api/state").json()["datasets"][0]["name"]
    response = client.post("/api/remove", json={"name": name})
    assert response.json()["removed"] is True
    assert client.get("/api/state").json()["datasets"] == []


def test_reset_clears_history_but_keeps_memory_by_default(client_with_data) -> None:
    client, _ = client_with_data
    client.post("/api/memory/add", json={"text": "Revenue is net of refunds", "kind": "definition"})
    client.post("/api/ask", json={"question": "total revenue by region"})
    client.post("/api/reset", json={})
    assert client.get("/api/memory").json()["stats"]["active"] == 1

    client.post("/api/reset", json={"memory": True})
    assert client.get("/api/memory").json()["stats"]["active"] == 0


def test_session_loads_bundled_sample_by_path(config) -> None:
    session = Session(config, [SAMPLES / "ecommerce_orders.csv"])
    try:
        assert "ecommerce_orders" in session.registry.names()
        assert session.describe()[0]["rows"] == 2600
    finally:
        session.close()


def test_session_rejects_a_missing_file(config) -> None:
    from byteanalyst.errors import ByteAnalystError

    with pytest.raises(ByteAnalystError):
        Session(config, [Path("does-not-exist.csv")])
