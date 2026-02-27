# Internal LLM Gateway  
### Enterprise-Style Tool-First RAG System with Secure Access, Audit Logging & Deterministic Business Tools

An internal-facing LLM Gateway that wraps a raw LLM backend (vLLM) with:

- authentication and role-based access control
- deterministic business tools
- retrieval-augmented grounding
- structured citation outputs
- audit logging and metrics
- CI-safe testability

> This project demonstrates how real organizations deploy internal AI assistants — not as raw model endpoints, but as secure, observable gateway systems.

---

# Why This Project Matters

Most LLM demos look like this:

User → Model → Text Output

Real production systems look like this:

User  
↓  
Gateway (Auth + Policy + Tools + RAG + Audit)  
↓  
LLM Backend  

This repository implements the gateway layer.

---

# Core Capabilities

## 1) Tool-First Architecture (Deterministic Layer)

Operational queries are routed to deterministic tools before calling the LLM.

Example:

- "List open Sev-2 incidents" → incident.list_open tool
- No LLM call
- No token usage
- Fully deterministic output
- Still backed by internal documentation citations

If the LLM backend is offline:

```
docker compose stop vllm
```

Tool-based queries continue to function.

This mirrors real production priorities:
- reliability
- cost control
- deterministic workflows

Execution modes (auto-selected via routing rules):

- tool        → deterministic business logic only  
- tool+llm    → tool output enriched with model reasoning  
- rag+llm     → retrieval grounding + model generation  
- llm         → direct model invocation  

---

## 2) Retrieval-Augmented Generation (RAG)

For knowledge questions, the gateway:

- indexes Markdown documents under `gateway/kb/`
- chunks with overlap
- builds embeddings (SentenceTransformer)
- performs cosine top-k retrieval
- injects context into the LLM prompt

Answers are grounded in internal documentation, reducing hallucination risk.

---

## 3) Enterprise Citation Cards

Responses return structured citation metadata:

```
{
"sources": [
{
"title": "Incident Response",
"doc_type": "Runbook",
"section": "# Severity Levels",
"score": 1.0,
"preview": "...",
"rank": 1
}
]
}
```


Tool-mode citations are intelligently distributed across multiple documents to maximize policy coverage.

---

## 4) Secure Access (Auth + RBAC)

All endpoints require API keys.

Roles:

- user → /ask
- admin → /reload_kb, /kb_status

Unauthorized access returns structured 403 responses.

---

## 5) Audit Logging (Privacy-Preserving)

Each request logs:

- request_id
- user_id
- role
- latency
- tool usage
- prompt hash (not raw prompt)

No sensitive text is stored.

Audit logs:

```
gateway/artifacts/audit/audit.jsonl
```


This mirrors internal AI governance requirements.

---

## 6) Observability

Prometheus metrics included:

- request counts
- error types
- latency histogram

Endpoint:

```
GET /metrics
```


---

# Architecture Overview

User  
↓  
POST /ask  
↓  
FastAPI Gateway  
  - Auth + RBAC  
  - Rate limiting  
  - Tool routing  
  - RAG retrieval  
  - Citation builder  
  - Structured JSON envelope  
↓  
vLLM (OpenAI-compatible backend)

---

# Repository Structure

```
llm-internal-assistant/
│
├── gateway/
│ ├── src/app.py
│ ├── src/tools/
│ ├── kb/
│ ├── tests/
│ ├── requirements.txt
│ └── Dockerfile
│
├── configs/
│ └── server.example.yaml
│
├── docker-compose.yml
├── pytest.ini
└── .github/workflows/ci.yml
```

---

# Running the System

Start services:
```
docker compose up -d --build
```

Health check:

```
curl http://127.0.0.1:8000/health
```

---

# Demo Scenarios

## 1) Deterministic Tool Query (LLM Not Required)

Stop the LLM backend (vLLM):

```
docker compose stop vllm
```
Send a tool-routed request (gateway will NOT call the LLM):

```
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method POST `
  -Headers @{"x-api-key"="dev-user-key"} `
  -ContentType "application/json" `
  -Body '{"question":"List open Sev-2 incidents","k":3}' |
  ConvertTo-Json -Depth 8
```
Expected:

meta.engine = tool

timings_ms.llm = 0

sources contains evidence cards even with vLLM offline

---

## 2) Tool + Policy Evidence

Query:

"Show me the SEV-2 checklist"

Returns:

- deterministic checklist steps
- distributed citation cards across multiple documents

---

## 3) RAG + LLM Knowledge Query

```
docker compose start vllm
```

Query:

"Why do we use embeddings?"

Returns:

- grounded citations
- RAG context injection
- structured JSON answer

---

# CI-Safe Testing

- No GPU required
- vLLM calls stubbed
- Embeddings disabled in CI
- Deterministic vector fallback

Run locally:

```
export PYTHONPATH=gateway
export DISABLE_EMBEDDINGS=1
pytest -q
```

---

# What This Demonstrates

This project demonstrates:

- production-style LLM wrapping
- secure gateway design
- deterministic tool routing
- grounded RAG systems
- structured response contracts
- observability and audit logging
- CI-safe AI engineering patterns

The system remains functional even if the LLM backend is unavailable.


# Design Tradeoffs

This system intentionally separates deterministic tools from LLM reasoning.

Key decisions:

- Operational workflows must remain functional even if the LLM backend is offline
- Business-critical queries should not depend on token generation
- Deterministic tools reduce latency and cost
- RAG is only invoked when semantic reasoning is required

Embeddings are lazily loaded and can be disabled in CI:

- avoids external downloads
- ensures deterministic test behavior
- stabilizes automated pipelines

These constraints reflect real production AI platform engineering tradeoffs.