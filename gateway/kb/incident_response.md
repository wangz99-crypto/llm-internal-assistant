# Incident Response Runbook

## Purpose
This document describes how to respond to common incidents for:
- Gateway (FastAPI)
- vLLM backend
- RAG indexing

## Severity levels
- SEV1: service down, all requests failing
- SEV2: partial degradation, high latency, some failures
- SEV3: minor issues, workaround exists

## SEV1: Gateway not reachable
### Symptoms
- GET /health fails from host
- Connection refused on 127.0.0.1:8000

### Checklist
1) docker compose ps
2) docker compose logs --tail 200 gateway
3) netstat -ano | findstr :8000

### Fix
- Restart gateway:
  docker compose restart gateway
- If code changed, rebuild:
  docker compose up -d --build

## SEV1: Gateway cannot reach vLLM
### Symptoms
- httpx.ConnectError in gateway logs
- /v1/chat/completions returns 502

### Checklist
1) docker compose ps (both Up?)
2) From inside gateway container:
   curl http://vllm:8000/health
3) Check vLLM logs:
   docker compose logs --tail 200 vllm

### Common causes
- vLLM still warming up / loading model
- HF download failure
- vLLM crashed and restarted

### Fix
- Wait for vLLM to finish model load
- Restart vLLM:
  docker compose restart vllm
- If HF 502, retry later or pre-download

## SEV2: RAG citations missing
### Symptoms
- /ask returns answer but citations empty
- kb_status shows files=0 chunks=0

### Checklist
1) GET /kb_status (admin)
2) Verify /app/kb exists in gateway container
3) Confirm KB files are .md or .txt and not empty
4) POST /reload_kb as admin

### Fix
- Put KB files under gateway/kb/
- Run POST /reload_kb
- Ensure KB_DIR points to /app/kb

## SEV3: Quality issues
### Symptoms
- Answer ignores citations or hallucination

### Checklist
- Verify context injected into prompt
- Increase k (e.g., 5)
- Reduce max_chars for chunking to improve precision
- Expand KB coverage

### Fix
- Improve KB docs (add examples and commands)
- Return section + score for transparency
