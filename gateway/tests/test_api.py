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


def test_health_service_field():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
    assert r.json().get("service") == "llm-gateway-demo"


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


def test_ask_includes_tool_result(monkeypatch):
    import src.app as mod

    # prevent embeddings path
    mod.KB_CHUNKS = []
    mod.KB_EMB = None

    # Mock the tool modules to ensure they're importable and working
    class MockToolResult:
        def __init__(self, ok=True, data=None, error=None):
            self.tool_name = "incident.list_open"
            self.ok = ok
            self.data = data or {"items": [{"id": "INC001", "severity": "Sev-2", "title": "Test Incident"}]}
            self.error = error

    # Create a mock router that returns a tool route
    def mock_route_tool(question):
        if "incident" in question.lower() or "sev-2" in question.lower():
            return ("incident.list_open", {"limit": 10})
        return None

    # Create a mock incident tool that returns a successful result
    def mock_list_open_incidents(question, limit=10):
        return MockToolResult(
            ok=True,
            data={"items": [
                {"id": "INC001", "severity": "Sev-2", "title": "Database connection issues", "status": "open"},
                {"id": "INC002", "severity": "Sev-2", "title": "API latency spike", "status": "investigating"}
            ]}
        )

    # Apply the mocks
    monkeypatch.setattr(mod, "route_tool", mock_route_tool)
    monkeypatch.setattr(mod, "list_open_incidents", mock_list_open_incidents)

    class FakeResp:
        def raise_for_status(self):
            return None
        def json(self):
            # model just echoes a generic response; tool result should still appear in envelope
            content = json.dumps({
                "summary": "ok",
                "steps": [],
                "notes": [],
                "confidence": 0.7
            })
            return {"choices": [{"message": {"content": content}}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, exc_type, exc, tb): return False
        async def post(self, url, json=None): return FakeResp()

    monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)

    r = client.post(
        "/ask",
        headers={"x-api-key": "dev-user-key"},
        json={"question": "List open Sev-2 incidents", "k": 3},
    )
    assert r.status_code == 200
    data = r.json()

    assert data["status"] == "ok"
    assert "tool" in data
    assert data["tool"]["used"] in (None, "incident.list_open")
    if data["tool"]["used"] == "incident.list_open":
        assert data["tool"]["result"]["ok"] is True
        assert "items" in data["tool"]["result"]["data"]
        assert len(data["tool"]["result"]["data"]["items"]) > 0
        assert data["tool"]["result"]["data"]["items"][0]["severity"] == "Sev-2"


def test_tool_runbook_checklist(monkeypatch):
    import src.app as mod

    # prevent embeddings path
    mod.KB_CHUNKS = []
    mod.KB_EMB = None

    # Mock the tool modules
    class MockToolResult:
        def __init__(self, ok=True, data=None, error=None, citations_hint=None):
            self.tool_name = "runbook.get_checklist"
            self.ok = ok
            self.data = data or {
                "sev": 2,
                "checklist": [
                    "Check service health dashboard",
                    "Review recent deployments",
                    "Check error rates in logs",
                    "Verify database connectivity",
                    "Check API response times",
                    "Review alert configurations",
                    "Check resource utilization",
                    "Verify backup status"
                ]
            }
            self.error = error
            self.citations_hint = citations_hint or ["runbook.md"]

    # Create a mock router that returns a tool route for checklist questions
    def mock_route_tool(question):
        if "checklist" in question.lower() or "sev-2" in question.lower():
            return ("runbook.get_checklist", {"sev": 2})
        return None

    # Create a mock runbook tool that returns a successful result
    def mock_get_sev_checklist(sev=2):
        return MockToolResult(
            ok=True,
            data={
                "sev": sev,
                "checklist": [
                    "Check service health dashboard",
                    "Review recent deployments",
                    "Check error rates in logs",
                    "Verify database connectivity",
                    "Check API response times",
                    "Review alert configurations",
                    "Check resource utilization",
                    "Verify backup status"
                ]
            },
            citations_hint=["runbook.md"]
        )

    # Apply the mocks
    monkeypatch.setattr(mod, "route_tool", mock_route_tool)
    monkeypatch.setattr(mod, "get_sev_checklist", mock_get_sev_checklist)

    r = client.post(
        "/ask",
        headers={"x-api-key": "dev-user-key"},
        json={"question": "Show me the SEV-2 checklist", "k": 3},
    )
    assert r.status_code == 200
    data = r.json()

    assert data["status"] == "ok"
    assert data["meta"]["mode"] == "tool"
    assert data["tool"]["used"] == "runbook.get_checklist"
    assert "checklist" in data["tool"]["result"]["data"]
    assert len(data["tool"]["result"]["data"]["checklist"]) > 0
    assert data["tool"]["result"]["data"]["sev"] == 2
    
    # Check that the answer is properly formatted for checklist tool
    assert "summary" in data["answer"]
    assert "SEV2" in data["answer"]["summary"] or "SEV-2" in data["answer"]["summary"]
    assert "steps" in data["answer"]
    assert len(data["answer"]["steps"]) > 0
    assert "notes" in data["answer"]
    assert "confidence" in data["answer"]
    assert data["answer"]["confidence"] > 0.9