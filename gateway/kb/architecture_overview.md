# Architecture Overview

## Components
### 1) Gateway (FastAPI)
Responsibilities:
- Auth (API keys)
- Policy enforcement (input size, max tokens)
- Audit logging (prompt_hash)
- Optional RAG retrieval + citations
- Proxy requests to vLLM

### 2) vLLM (OpenAI-compatible server)
Responsibilities:
- Host LLM model
- Provide /v1/chat/completions
- Provide streaming SSE

### 3) Knowledge Base (kb/*.md)
Responsibilities:
- Provide internal, curated docs
- Used for RAG grounding

## Request flow
1) Client calls gateway endpoint with x-api-key
2) Gateway validates auth + rate limit
3) If /ask:
   - Embed question (CPU embeddings)
   - Retrieve top-k KB chunks
   - Inject context into prompt
4) Gateway forwards request to vLLM
5) Gateway returns answer + citations
6) Gateway writes audit event (JSONL)

## Why this design is “enterprise-like”
- Control plane at gateway (policies, auth, audit)
- Inference backend isolated behind internal network
- Citations provide transparency and traceability
- Metrics for SRE/ops

## Future extensions
- Redis rate limiter
- Persistent vector store (FAISS, Qdrant)
- Multi-tenant auth & RBAC
- CI checks for KB quality
