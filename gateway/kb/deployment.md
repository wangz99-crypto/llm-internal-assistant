# Deployment Guide

## Supported environments
- Local dev (Docker Compose)
- Single-node VM (Docker Compose)
- Later: Kubernetes (future)

## Local: Docker Compose quickstart
1) Build and start:
- docker compose up -d --build

2) Validate gateway:
- GET http://127.0.0.1:8000/health  -> {"status":"ok"}

3) Validate vLLM:
- GET http://127.0.0.1:8001/health  -> 200

4) Validate RAG:
- POST http://127.0.0.1:8000/ask with API key
- Expect citations from kb/*.md

## Environment variables
### Gateway container
- CONFIG_PATH=/app/configs/server.yaml
- VLLM_BASE_URL=http://vllm:8000

## Common deployment pitfalls
## 1) Model download fails (HF 502 / network)
Symptoms:
- vLLM logs show Hugging Face 502 or tokenizer download errors

Fix:
- Retry later
- Pre-download model into HF cache
- Ensure outbound internet is available

## 2) Port mismatch
Symptoms:
- Host calls to gateway fail

Check:
- docker compose ps
- gateway should map 8000->8000
- vLLM should map 8001->8000

## 3) GPU not detected
Symptoms:
- vLLM starts in CPU or fails to load CUDA

Check:
- Docker Desktop / WSL GPU support
- nvidia-smi on host
- compose device reservation includes capabilities: [gpu]
