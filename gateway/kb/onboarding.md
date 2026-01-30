# Onboarding Guide

## What is this system?
This is an internal LLM Gateway that provides:
- secure access via API keys
- audit logging (prompt hash only)
- rate limiting
- RAG from internal knowledge base documents

## Quickstart
1) docker compose up -d --build
2) call GET /health
3) call POST /ask (with x-api-key)
4) edit KB docs then call POST /reload_kb (admin)
