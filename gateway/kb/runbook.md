# Runbook

## Service startup checklist
1) docker compose up -d --build
2) GET http://127.0.0.1:8000/health should return {"status":"ok"}
3) Check vLLM health: http://127.0.0.1:8001/health
4) After editing KB files, call POST /reload_kb (admin)
5) Verify /ask returns citations

## Troubleshooting: gateway cannot reach vLLM
- Confirm both containers are Up: docker compose ps
- From gateway container: curl http://vllm:8000/health
- If model download fails (HF 502), retry later or pre-download model
