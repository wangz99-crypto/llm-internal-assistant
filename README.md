# Internal LLM Gateway with Enterprise-Style RAG + Citations

**An internal-facing LLM Gateway service with secure access control, lightweight RAG retrieval, and enterprise-style citation outputs.**

> **Goal:** demonstrate how a raw local LLM endpoint (vLLM) can be engineered into a secure, observable, CI-tested, enterprise-ready gateway system.

---

## Project Overview

This repository implements a production-inspired **internal LLM Gateway architecture**:

- **vLLM** serves the core OpenAI-compatible chat model  
- A **FastAPI gateway** wraps the model with authentication, audit logging, retrieval, and citations  
- Internal Markdown documentation forms a lightweight Knowledge Base (KB)  
- Responses are grounded with structured enterprise-style citation cards  

Rather than exposing a raw model endpoint directly, this project demonstrates how real organizations deploy internal assistants with:

- security controls  
- grounded answers from internal policy/runbook docs  
- operational endpoints for KB management  
- reproducible CI-safe testing without GPUs  

---
## Key Features

### Secure API Access (Auth + Roles)

All endpoints require an API key with role-based access control:

- **User role:** `/ask`  
- **Admin role:** `/reload_kb`, `/kb_status`  

Unauthorized access returns a clean enterprise-style `403` response.

---

### Retrieval-Augmented Generation (RAG)

The gateway builds a semantic index over internal documentation:

- Markdown files under `gateway/kb/`  
- Chunking with overlap  
- SentenceTransformer embeddings (`bge-small-en-v1.5`)  
- Cosine similarity **top-k retrieval**  

This ensures answers are grounded in internal sources, not hallucinated.

---
### Enterprise UI Response Format

The `/ask` endpoint returns a production-style JSON envelope:

- `request_id`
- `answer` (summary + steps + confidence)
- `sources`
- `meta`
- `timings_ms`
- warnings/debug hooks

Example:

```json
{
  "request_id": "...",
  "status": "ok",
  "answer": {
    "summary": "...",
    "steps": ["..."],
    "confidence": 0.95
  },
  "sources": [...],
  "timings_ms": {...}
}
### Operational KB Management

Admins can hot-reload the Knowledge Base (KB) without restarting the service:

```bash
curl -X POST http://127.0.0.1:8000/reload_kb \
  -H "x-api-key: dev-admin-key" \
  -d "{}"
```
Status endpoint:
```
curl http://127.0.0.1:8000/kb_status \
  -H "x-api-key: dev-admin-key"
```
Returns:
```
{
  "files": 11,
  "chunks": 44,
  "emb_shape": [44, 384]
}

```
This supports operational workflows where documentation can be updated and re-indexed dynamically.

### Audit Logging (Privacy-Preserving)

The gateway logs all requests with enterprise-style compliance in mind:

- `request_id`
- user identity (`user_id`, role)
- latency + timing breakdown
- prompt hash only (no raw text stored)

This mirrors real internal AI governance requirements, where systems must be observable without retaining sensitive user content.

Audit logs are written under:

- `gateway/artifacts/audit/audit.jsonl`

---
### CI-Safe Testing (Pytest + GitHub Actions)

This repository includes full CI testing via GitHub Actions:

- runs on every push / pull request  
- no GPU required  
- no external model downloads  
- vLLM calls are stubbed in tests  

To avoid embedding downloads in CI:

- embeddings are lazily loaded  
- `DISABLE_EMBEDDINGS=1` forces deterministic zero vectors  

Run tests locally:

```
export PYTHONPATH=gateway
export DISABLE_EMBEDDINGS=1
pytest -q
```
These checks ensure that refactoring gateway logic or upgrading dependencies cannot silently break API behavior.

## Architecture Overview

```
User
 │
 │ POST /ask   (API key required)
 ▼
Gateway (FastAPI)
 │  ├── Auth + RBAC
 │  ├── Rate limiting
 │  ├── Audit logging (hash only)
 │  ├── KB retrieval (embeddings + cosine top-k)
 │  └── Enterprise JSON response + citations
 ▼
vLLM Backend (OpenAI-compatible API)
```
The gateway retrieves relevant KB chunks, injects them into the prompt context, and returns answers grounded in citations.

## Repository Structure

```
llm-internal-assistant/
│
├── gateway/
│   ├── src/app.py              # Main FastAPI gateway
│   ├── kb/                     # Internal KB documents (Markdown)
│   ├── tests/                  # Pytest suite (CI-safe)
│   ├── requirements.txt        # Core runtime dependencies
│   └── Dockerfile              # Containerized gateway
│
├── configs/
│   └── server.example.yaml     # Config template (no secrets)
│
├── docker-compose.yml          # vLLM + Gateway stack
├── pytest.ini                  # Pytest configuration
└── .github/workflows/ci.yml    # GitHub Actions CI pipeline
```
## Running the System (Docker Compose)

Start both services (Gateway + vLLM backend):

```
docker compose up -d --build
```
Check gateway health:
```
curl http://127.0.0.1:8000/health
```

Expected:
```
{"status":"ok"}
```

## Example Query with Citations

Send a user query to the gateway:

```
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-user-key" \
  -d "{\"question\":\"Why do we need embeddings?\",\"k\":3}"
```
The response includes grounded citations retrieved from internal KB documents.

## Why This Project Matters (Interview Summary)

This project demonstrates applied LLM engineering beyond prompt demos:

- wrapping models with real security + observability  
- building grounded RAG retrieval pipelines  
- producing enterprise-ready JSON outputs  
- supporting operational workflows (reload/status)  
- implementing CI-safe testing patterns  

It reflects how internal AI assistants are built in real organizations.
---

## Contact

This repository is part of a personal ML engineering portfolio.

Feedback and discussion are welcome via GitHub Issues.
