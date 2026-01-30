# API Reference

## Overview
This gateway provides:
- Secure access via API key
- Policy enforcement (input length, rate limiting, max tokens)
- Audit logging (no raw prompts)
- Optional RAG citations from internal KB

## Endpoints

### GET /health
Returns basic readiness.

**Response**
- status: "ok"

### GET /metrics
Prometheus metrics endpoint.

### POST /v1/chat/completions
OpenAI-compatible chat completions proxy to vLLM.

**Headers**
- x-api-key: required when auth.enabled=true

**Body**
- model: string
- messages: [{role, content}]
- max_tokens: optional (capped by role policy)
- temperature: optional

**Notes**
- Gateway enforces max_input_chars by policy
- Gateway stores prompt_hash instead of raw prompt

### POST /v1/chat/completions/stream
Streaming SSE proxy to vLLM.

### POST /ask
RAG endpoint: retrieve top-k KB chunks and generate answer grounded in citations.

**Headers**
- x-api-key: required

**Body**
- question: string (required)
- k: integer (default 3)

**Response**
- answer: string
- citations: list of objects
  - source: "kb/<file>.md"
  - doc_type: "Runbook" | "FAQ" | "Security" | "Architecture" | "Document"
  - section: heading nearest to chunk (if available)
  - score: cosine similarity
  - chunk_id: int
  - chunk_preview: string

### POST /reload_kb (admin only)
Rebuild KB index and embeddings.

### GET /kb_status (admin only)
Shows KB status: files, chunks, embedding shape.
