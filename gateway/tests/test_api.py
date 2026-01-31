import os
import importlib
import json 

import json
from fastapi.testclient import TestClient

# IMPORTANT: conftest already set env before this import happens
from src.app import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_kb_status_requires_admin():
    r = client.get("/kb_status", headers={"x-api-key": "dev-user-key"})
    assert r.status_code == 403
    assert "detail" in r.json()


def test_kb_status_admin_ok(monkeypatch):
    import src.app as mod

    mod.KB_CHUNKS = [
        {"source": "kb/faq.md", "text": "hello", "start": 0, "section": "# FAQ", "doc_type": "FAQ"}
    ]
    mod.KB_EMB = None

    r = client.get("/kb_status", headers={"x-api-key": "dev-admin-key"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["files"] >= 1
    assert data["chunks"] >= 1


def test_ask_returns_enterprise_ui(monkeypatch):
    import src.app as mod

    # Prevent embeddings path
    mod.KB_CHUNKS = []
    mod.KB_EMB = None

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            content = json.dumps({
                "summary": "Stubbed answer for CI.",
                "steps": ["step 1"],
                "notes": ["note 1"],
                "confidence": 0.9
            })
            return {"choices": [{"message": {"content": content}}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None):
            return FakeResp()

    monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)

    r = client.post(
        "/ask",
        headers={"x-api-key": "dev-user-key"},
        json={"question": "Why do we need embeddings?", "k": 3},
    )
    assert r.status_code == 200
    data = r.json()

    assert data["status"] == "ok"
    assert "request_id" in data
    assert "answer" in data
    assert isinstance(data["answer"]["steps"], list)
    assert "meta" in data
    assert "timings_ms" in data
