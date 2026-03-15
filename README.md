# Internal LLM Gateway
### Enterprise-Style Tool-First RAG System with Secure Access, Audit Logging & Deterministic Business Tools

An internal-facing LLM Gateway that wraps a raw LLM backend (vLLM) with:

- hybrid keyword-semantic tool routing
- authentication and role-based access control
- deterministic business tools
- retrieval-augmented grounding
- retrieval quality evaluation harness
- structured citation outputs
- audit logging and metrics
- CI-safe testability

> This project demonstrates how real organizations deploy internal AI assistants — not as raw model endpoints, but as secure, observable gateway systems.

---

# Live Demo (Render Deployment)

You can try the live instance hosted on Render:

https://llm-internal-assistant.onrender.com/

---

## Quick Start (60 seconds)

1) Verify service health:

```bash
curl https://llm-internal-assistant.onrender.com/health
```

Expected:

```json
{"status":"ok","service":"llm-gateway-demo"}
```

2) Try a deterministic tool query:

```bash
curl -X POST https://llm-internal-assistant.onrender.com/ask \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-user-key" \
  -d '{"question":"List open Sev-2 incidents","k":3}'
```

3) Open the Web UI in browser:

https://llm-internal-assistant.onrender.com/

---

## Demo API Keys

| Role  | API Key        |
|-------|----------------|
| User  | dev-user-key   |
| Admin | dev-admin-key  |

Include API key in requests:

```bash
-H "x-api-key: dev-user-key"
```

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

This repository implements the **gateway layer** — the part real companies actually build.

---

# Core Capabilities

## 1) Tool-First Architecture (Deterministic Layer)

Operational queries are routed to deterministic tools before calling the LLM.

Tool routing works in two layers:
- **Keyword routing** — exact phrase matching; fast, deterministic, no embedding cost
- **Semantic fallback** — when keyword routing finds no match, an embedding-based classifier compares the query against precomputed intent vectors. Definitional questions (`"what is ..."`, `"explain ..."`) are excluded before embedding to prevent false positives from shared domain vocabulary.

Example:

- "List open Sev-2 incidents" → `incident.list_open`
- No LLM call
- No token usage
- Fully deterministic output
- Still backed by internal documentation citations

If the LLM backend is offline:

```bash
docker compose stop vllm
```

Tool-based queries continue to function.

This mirrors real production priorities:

- reliability  
- cost control  
- deterministic workflows  

---

### Execution Modes

The gateway dynamically selects execution mode **before token generation**:

- `tool` → deterministic business logic only
- `tool+llm` → tool output enriched with model reasoning
- `rag+llm` → retrieval grounding + model generation
- `llm` → direct model invocation

Mode selection is handled via routing rules.

---

## 2) Retrieval-Augmented Generation (RAG)

For knowledge questions, the gateway:

- indexes Markdown documents under `gateway/kb/`
- chunks with overlap
- builds embeddings (SentenceTransformer)
- performs cosine top-k retrieval
- filters out low-relevance chunks below a cosine threshold before building LLM context
- injects context into the LLM prompt

If no chunks clear the relevance threshold, the response includes a structured warning (`warnings` field) rather than silently sending ungrounded context to the LLM.

Answers are grounded in internal documentation, reducing hallucination risk.

Embeddings can be disabled in CI:

```bash
DISABLE_EMBEDDINGS=1
```

---

## 3) Retrieval Evaluation Harness

Retrieval quality is measured offline against a labeled eval set:

- 15 ground-truth examples across incident, escalation, runbook, audit, and policy categories
- tagged by difficulty (`easy` / `medium` / `hard`)
- reports hit@1, hit@3, per-category and per-difficulty breakdown
- identifies failed retrievals with predicted top-1 and top-3 source files

Run without a live app or GPU:

```bash
PYTHONPATH=gateway python gateway/eval/retrieval_eval.py
```

Baseline (keyword-only routing): hit@3 = 13/15 (86%). The two persistent hard failures motivate the semantic routing layer.

---

## 5) Enterprise Citation Cards

Responses return structured citation metadata:

```json
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

## 6) Secure Access (Auth + RBAC)

All endpoints require API keys.

Roles:

- `user` → `/ask`
- `admin` → `/reload_kb`, `/kb_status`

Unauthorized access returns structured 403 responses.

---

## 7) Audit Logging (Privacy-Preserving)

Each request logs:

- request_id
- user_id
- role
- latency
- tool usage
- prompt hash (not raw prompt text)

No sensitive text is stored.

Audit logs:

```
gateway/artifacts/audit/audit.jsonl
```

This mirrors internal AI governance requirements.

---

## 8) Observability

Prometheus metrics included:

- request counts
- error types
- latency histogram

Endpoint:

```
GET /metrics
```

---

# Response Contract (Structured JSON Envelope)

All responses follow a structured contract:

```json
{
  "status": "ok",
  "request_id": "uuid",
  "answer": {
    "summary": "...",
    "steps": [],
    "notes": [],
    "confidence": 0.95
  },
  "sources": [],
  "tool": {
    "used": "incident.list_open",
    "result": {}
  },
  "meta": {
    "mode": "tool",
    "engine": "gateway"
  },
  "timings_ms": {
    "llm": 0,
    "total": 5
  },
  "warnings": []
}
```

This ensures:

- deterministic structure
- API contract stability
- UI compatibility
- auditability

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

# Running Locally

Start services:

```bash
docker compose up -d --build
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

---

# CI-Safe Testing

No GPU required.

Run locally:

```bash
export PYTHONPATH=gateway
export DISABLE_EMBEDDINGS=1
pytest -q
```

---

# Design Tradeoffs

This system intentionally separates deterministic tools from LLM reasoning.

Key decisions:

- Operational workflows must remain functional even if the LLM backend is offline
- Business-critical queries should not depend on token generation
- Deterministic tools reduce latency and cost
- RAG is invoked only when semantic reasoning is required

Embeddings are lazily loaded and can be disabled in CI to ensure:

- no external downloads
- deterministic test behavior
- stable automated pipelines