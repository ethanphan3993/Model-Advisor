"""API smoke tests."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_use_cases(client):
    r = client.get("/api/use-cases")
    assert r.status_code == 200
    body = r.json()
    assert any(u["id"] == "coding" for u in body)
    assert any(u["id"] == "agentic" for u in body)


def test_harnesses(client):
    r = client.get("/api/harnesses")
    assert r.status_code == 200
    body = r.json()
    ids = {h["id"] for h in body}
    assert "cline" in ids
    assert "claude-code" in ids
    assert "hermes-agent" in ids
    assert "openclaw" in ids
    assert "ollama" in ids


def test_model_not_found_returns_404(client):
    r = client.get("/api/models/nonexistent:99b:instruct")
    assert r.status_code == 404
