# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run tests (CI-safe, no GPU required)
```bash
export PYTHONPATH=gateway
export DISABLE_EMBEDDINGS=1
pytest -q
```
`conftest.py` automatically sets `CONFIG_PATH` to `configs/server.example.yaml` and `DISABLE_EMBEDDINGS=1`, so tests never download embeddings or call vLLM.

### Run a single test
```bash
export PYTHONPATH=gateway
export DISABLE_EMBEDDINGS=1
pytest gateway/tests/test_api.py::test_health_ok -q
```

### Run locally (full stack with vLLM)
```bash
docker compose up -d --build
```
Gateway listens on `http://localhost:8000`. vLLM listens on `http://localhost:8001`.

### Run gateway only (no GPU/LLM)
```bash
cd gateway
DISABLE_EMBEDDINGS=1 DEMO_MODE=1 CONFIG_PATH=configs/server.example.yaml \
  uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run gateway with embeddings, free-form /ask (no LLM, no GPU)
```bash
cd gateway
DEMO_SCRIPT=0 DEMO_MODE=1 CONFIG_PATH=configs/server.example.yaml \
  uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
`DEMO_SCRIPT` defaults to `"1"` (on) even when unset. Without `DEMO_SCRIPT=0`, every `/ask` without a `scenario_id` returns HTTP 400. Use this command for free-form local testing with embeddings active.

## Architecture

All meaningful server logic lives in a single file: `gateway/src/app.py`. Everything else supports it.

### Request flow through `POST /ask`

1. **Auth** — `auth_user()` (line ~547) checks `x-api-key` header against `server.yaml` API keys. Returns `ApiKeyUser(user_id, role)`.
2. **Rate limiting** — `_check_rate_limit()` (line ~569) uses an in-memory per-user counter.
3. **Scenario override** — if `scenario_id` is present, the request is forced into deterministic mode (no LLM) and returns early from a pre-configured demo scenario.
4. **Tool routing** — `route_tool(question)` in `gateway/src/tools/router.py` does keyword matching first. If it returns `None`, `_semantic_route_tool()` in `app.py` runs an embedding-based fallback against precomputed intent vectors (`TOOL_INTENT_EMB`). Semantic routing applies two guards before embedding: (a) returns `None` if embeddings are unavailable or disabled; (b) returns `None` if the question matches `_DEFINITIONAL_RE` — definitional openers (`what is`, `what's`, `define`, `explain`) that should never trigger a tool regardless of cosine score.
5. **Tool execution** — if a tool matched: `get_sev_checklist()` or `list_open_incidents()` is called. On success, the request **returns immediately** — no embed, no LLM, no token usage.
6. **RAG embed** — `embed_texts([question])` (line ~630) via SentenceTransformer. Skipped if `DISABLE_EMBEDDINGS=1` or KB is empty.
7. **RAG retrieve** — `cosine_topk_with_scores()` (line ~295) against the in-memory KB embedding matrix. Builds `sources` citation cards via `citation_card()`.
8. **LLM call** — `httpx.AsyncClient.post()` to `{vllm_base_url}/v1/chat/completions`. System prompt varies by `mode` (`kb` / `hybrid` / `chat`). LLM must return strict JSON.
9. **Parse** — `safe_parse_ui_json()` (line ~988) handles malformed LLM output; strips code fences, falls back to regex.
10. **Fallback** — if vLLM is unreachable (`httpx.RequestError`), falls back to `keyword_topk()` + `demo_template_answer()` with `engine: "kb_fallback"`.
11. **Audit** — `write_audit()` (line ~816) appends a JSONL line to `gateway/artifacts/audit/audit.jsonl`. Stores prompt hash, not raw text.
12. **Response** — `make_ok_envelope()` (line ~1032) wraps everything into the structured contract: `{status, request_id, answer, sources, tool, meta, timings_ms, warnings}`.

### Key modules

| Path | Role |
|---|---|
| `gateway/src/app.py` | Entire FastAPI app — all routes, KB indexing, RAG, LLM orchestration |
| `gateway/src/tools/router.py` | Keyword-based tool dispatcher |
| `gateway/src/tools/runbook_tool.py` | Returns hardcoded SEV-2 checklist |
| `gateway/src/tools/incident_tool.py` | Returns demo incidents from JSON |
| `gateway/src/ui/demo_page.py` | HTML demo UI served at `GET /` |
| `gateway/configs/server.yaml` | Auth keys, rate limits, model names, vLLM URL |
| `gateway/kb/` | Markdown knowledge base; chunked and embedded at startup |
| `gateway/tests/conftest.py` | Sets env vars before app import — order matters |

### Configuration

`server.yaml` is loaded at import time. Fields that matter most:
- `auth.api_keys` — list of `{key, user_id, role}` objects
- `backend.model` / `backend.embedding_model` — model IDs passed to vLLM / SentenceTransformer
- `backend.vllm_base_url` — overridden by `VLLM_BASE_URL` env var
- `policy.rate_limit_rpm_user` / `policy.rate_limit_rpm_admin`

Use `configs/server.example.yaml` as the base for local development and tests. `configs/server.yaml` is the live config (may differ).

### Environment variables

| Variable | Effect |
|---|---|
| `DISABLE_EMBEDDINGS=1` | Skip SentenceTransformer load; `keyword_topk()` used instead |
| `DEMO_MODE=1` | Disable all LLM calls; use deterministic KB/tool responses only |
| `DEMO_SCRIPT=1` | Require `scenario_id` on every `/ask` request. **Default is `"1"` (on)** — must pass `DEMO_SCRIPT=0` explicitly to allow free-form queries |
| `CONFIG_PATH` | Path to YAML config file |
| `VLLM_BASE_URL` | Override vLLM endpoint URL |
| `DEMO_UI=1` | Include `debug.routing`, `debug.tool_used`, `debug.llm_used` in every `/ask` response; renders a grey debug panel in the UI |

### Adding a new tool

1. Implement the tool function in `gateway/src/tools/` returning a `ToolResult(ok, data, citations_hint)`.
2. Add a keyword routing rule to `gateway/src/tools/router.py` → `route_tool()`.
3. Add 2–3 natural-language intent descriptions to `INTENT_DESCRIPTIONS` in `router.py` so the semantic fallback can also match the tool.
4. Add a dispatch branch in `app.py` around line 1414 where `tool_name` is matched.
5. Add an answer-formatting branch at the same location.
6. Add a return branch in `_semantic_route_tool()` in `app.py` for the new `tool_name`.

### KB documents

Markdown files in `gateway/kb/` are chunked (480 chars, 120 overlap) and embedded at startup into `KB_CHUNKS` (list of dicts) and `KB_EMB` (numpy matrix). Both are protected by `KB_LOCK` (asyncio.Lock). To add knowledge, drop a `.md` file into `gateway/kb/` and call `POST /reload_kb` (admin key required) or restart the server.

### Retrieval evaluation

`gateway/eval/` contains an offline eval harness:
- `eval_set.json` — 15 ground-truth (question → expected source file) examples, tagged by `category` and `difficulty` (easy / medium / hard)
- `retrieval_eval.py` — standalone script; no running app, no new dependencies

```bash
PYTHONPATH=gateway python gateway/eval/retrieval_eval.py
```

The script replicates `keyword_topk()` and `chunk_text()` from `app.py` exactly (alignment comments mark each copy). Reports hit@1, hit@3, per-category, per-difficulty, and failed examples with top-1 predicted file. Baseline: hit@3 = 13/15 (86%) on keyword path. The two persistent failures (`inc-03`, `aud-02`) are hard semantic cases that motivate the semantic router.

### Known design pitfalls

**`make_ok_envelope()` has no `warnings` parameter.**
Warnings are attached after the call: `resp["warnings"] = warnings`. Every early-return path must do this explicitly before `return JSONResponse(...)`. Missing it silently drops all accumulated warnings. The `kb_fallback` early-return path was a prior example of this.

**DEMO_MODE uses keyword retrieval, not cosine.**
The `if DEMO_MODE and tool_result is None:` branch (~line 1641) calls `keyword_topk()` and returns before the LLM normal path. `cosine_topk_with_scores()` and `RETRIEVAL_SCORE_THRESHOLD` only apply in the LLM normal path (~line 1935+). Do not expect the retrieval threshold warning to fire in `DEMO_MODE=1`.

**`warnings = [...]` vs `warnings.append(...)` — assignment silently drops earlier warnings.**
Some branches historically used assignment (`warnings = ["..."]`) instead of append. This drops any warning accumulated earlier in the same request (e.g. the retrieval threshold warning). Always use `.append()` unless intentionally resetting the list.

**`debug` fields from `if DEMO_UI:` only reach the main LLM return path.**
There are three early-return paths (DEMO_MODE tool-first ~line 1675, DEMO_MODE KB ~line 1983, `kb_fallback` ~line 2152) that exit before the final `if DEMO_UI:` block. Each must explicitly set `resp["debug"]` and apply the same `if DEMO_UI:` injection before returning. Missing one means the debug panel silently shows nothing for that execution path.

**Raising `SEMANTIC_TOOL_THRESHOLD` alone cannot fix definitional false positives.**
Queries like "what is SEV-1?" score misleadingly high (~0.50–0.60) against procedural intent phrases because they share domain vocabulary. The gap between a genuine procedural match and a definitional false positive may be only 0.05–0.10, making a safe threshold hard to find. The correct fix is `_DEFINITIONAL_RE`: a structural guard that blocks `what is / what's / define / explain` openers before any embedding work. Current calibration: threshold = 0.60; `what are ...` is intentionally NOT in the regex so procedural "what are the steps for..." queries still route correctly.

## Working Style Rules (Important)

When modifying code in this repo:

- Always explain the plan before making larger changes.
- Prefer minimal, safe edits over refactors.
- Do not refactor unless explicitly asked.
- Do not change unrelated files.

For backend changes:

- Follow the existing structure in `gateway/src/app.py`.
- Avoid restructuring the `/ask` flow.
- Preserve the response contract from `make_ok_envelope()`.

For debugging:

- Analyze first, fix second.
- Explain root cause clearly before implementing fixes.

For tests:

- Follow existing pytest style.
- Keep tests minimal and focused.
- Avoid unnecessary mocks or refactors.

Never:

- introduce large architectural changes without approval
- rename core functions unnecessarily
- change response schemas without explanation