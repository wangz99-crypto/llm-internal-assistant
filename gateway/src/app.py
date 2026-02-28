import os
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import asyncio
import logging
from typing import Any, Dict, Optional,List, Tuple

import re
import hashlib
import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# --- tools (business layer) ---
from src.tools.router import route_tool
from src.tools.incident_tool import list_open_incidents
from src.tools.runbook_tool import get_sev_checklist

# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class ApiKeyUser:
    user_id: str
    role: str


@dataclass(frozen=True)
class Policy:
    max_input_chars: int
    max_tokens_default: int
    max_tokens_admin: int
    rate_limit_rpm_user: int
    rate_limit_rpm_admin: int
    timeout_seconds: int


@dataclass(frozen=True)
class Settings:
    auth_enabled: bool
    api_keys: Dict[str, ApiKeyUser]
    policy: Policy
    vllm_base_url: str
    model: str
    embedding_model: str


# ----------------------------
# Base paths (stable)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # /app


# ----------------------------
# Logger
# ----------------------------
logger = logging.getLogger("gateway")


# ----------------------------
# Global RAG state + lock
# ----------------------------
KB_LOCK = asyncio.Lock()

KB_DIR = BASE_DIR / "kb"
KB_CHUNKS: list[dict] = []
KB_EMB: np.ndarray | None = None


def chunk_text(text: str, max_chars: int = 320, overlap: int = 80) -> list[tuple[str, int]]:
    """
    Return list of (chunk_text, start_offset).
    Smaller chunks -> more realistic enterprise KB retrieval.
    """
    text = text.strip()
    if not text:
        return []
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    out: list[tuple[str, int]] = []
    i = 0
    n = len(text)

    while i < n:
        j = min(n, i + max_chars)
        out.append((text[i:j], i))
        if j >= n:
            break
        i = j - overlap

    return out


def display_source(path_str: str) -> str:
    """Make citations look nicer: /app/kb/x.md -> kb/x.md (relative to BASE_DIR)."""
    p = Path(path_str)
    try:
        return str(p.relative_to(BASE_DIR))
    except Exception:
        return path_str
def pretty_title(doc_type: str, source: str) -> str:
    base = Path(source).stem.replace("_", " ").replace("-", " ").title()
    if doc_type and doc_type != "Document":
        return f"{doc_type}: {base}"
    return base

def citation_card(chunk: dict, score: float, chunk_id: int) -> dict:
    src_rel = display_source(chunk["source"])  # e.g. kb/runbook.md
    src_short = str(Path(src_rel).name)        # e.g. runbook.md

    return {
        "id": f"{src_rel}#{chunk_id}",
        "title": pretty_title(chunk.get("doc_type", "Document"), src_rel),
        "doc_type": chunk.get("doc_type", "Document"),
        "section": chunk.get("section"),
        "score": round(score, 4),
        "source": src_short,
        "preview": (chunk.get("text", "")[:240]).strip(),
        "chunk_id": chunk_id,
    }

def safe_parse_json_answer(raw: str) -> dict:
    """
    Expect model to output JSON: {summary:str, steps:[...], notes:[...]}
    Fallback gracefully if model outputs plain text.
    """
    raw = (raw or "").strip()
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("answer is not a dict")
        obj.setdefault("summary", raw)
        obj.setdefault("steps", [])
        obj.setdefault("notes", [])
        if not isinstance(obj["steps"], list):
            obj["steps"] = []
        if not isinstance(obj["notes"], list):
            obj["notes"] = []
        return obj
    except Exception:
        return {"summary": raw, "steps": [], "notes": []}

# ----------------------------
# KB metadata helpers
# ----------------------------
DOC_TYPE_MAP = {
    "policies.md": "Policy",
    "runbook.md": "Runbook",
    "onboarding.md": "Onboarding",
    "faq.md": "FAQ",
}

_SECTION_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")

def infer_doc_type(path_str: str) -> str:
    name = Path(path_str).name.lower()
    return DOC_TYPE_MAP.get(name, "Document")

def extract_section_for_chunk(full_text: str, chunk_start: int) -> str | None:
    """
    Given full doc text and a chunk start offset,
    return the nearest preceding markdown heading as section title.
    """
    # find all headings with their positions
    last = None
    for m in _SECTION_RE.finditer(full_text):
        if m.start() <= chunk_start:
            # build something like: "## Troubleshooting"
            hashes = m.group(1)
            title = m.group(2).strip()
            last = f"{hashes} {title}"
        else:
            break
    return last

def clean_preview(text: str, limit: int = 240) -> str:
    text = _ws.sub(" ", text).strip()
    return text[:limit]

def cosine_topk_with_scores(query_vec: np.ndarray, mat: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Return top-k (chunk_index, cosine_similarity)."""
    if mat is None or mat.size == 0:
        return []
    n = mat.shape[0]
    if k <= 0:
        return []
    k = min(k, n)

    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = m @ q  # shape: (n,)
    idxs = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idxs]


def load_settings() -> Settings:
    """
    Load Settings from YAML config.
    - Uses CONFIG_PATH env if set, else defaults to configs/server.yaml
    - CI-friendly fallback to gateway/configs/server.example.yaml when missing
    - Allows ENV override for VLLM_BASE_URL
    """
    # 1) Resolve config path
    config_path = os.environ.get("CONFIG_PATH", "configs/server.yaml")
    path = Path(config_path)

    if not path.exists():
        # Fallback: <repo>/gateway/configs/server.example.yaml
        # This file lives at: gateway/src/app.py -> parents[1] == gateway/
        fallback = Path(__file__).resolve().parents[1] / "configs" / "server.example.yaml"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                f"Config not found: {path} (CONFIG_PATH={config_path}). "
                f"Fallback also missing: {fallback}"
            )

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # 2) Backend base URL (ENV override)
    vllm_base_url = os.environ.get("VLLM_BASE_URL", str(cfg.get("backend", {}).get("vllm_base_url", "http://localhost:8001")))
    vllm_base_url = vllm_base_url.rstrip("/")

    # 3) Auth
    auth_cfg = cfg.get("auth", {}) or {}
    auth_enabled = bool(auth_cfg.get("enabled", True))
    api_keys_cfg = auth_cfg.get("api_keys", []) or []

    api_keys: Dict[str, ApiKeyUser] = {}
    for item in api_keys_cfg:
        # defensive reads
        key = str(item.get("key", "")).strip()
        user_id = str(item.get("user_id", "")).strip()
        role = str(item.get("role", "user")).strip()
        if key:
            api_keys[key] = ApiKeyUser(user_id=user_id, role=role)

    # 4) Policy
    pol = cfg.get("policy", {}) or {}
    policy = Policy(
        max_input_chars=int(pol.get("max_input_chars", 8000)),
        max_tokens_default=int(pol.get("max_tokens_default", 256)),
        max_tokens_admin=int(pol.get("max_tokens_admin", 512)),
        rate_limit_rpm_user=int(pol.get("rate_limit_rpm_user", 30)),
        rate_limit_rpm_admin=int(pol.get("rate_limit_rpm_admin", 120)),
        timeout_seconds=int(pol.get("timeout_seconds", 60)),
    )

    # 5) Model / Embeddings model
    backend = cfg.get("backend", {}) or {}
    model = str(backend.get("model", "Qwen/Qwen2.5-1.5B-Instruct")).strip()

    # Important: embeddings model should default to a real embedding model, not the chat model
    embedding_model = str(backend.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")).strip()

    return Settings(
        auth_enabled=auth_enabled,
        api_keys=api_keys,
        policy=policy,
        vllm_base_url=vllm_base_url,
        model=model,
        embedding_model=embedding_model,
    )


DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

def auth_user(x_api_key: Optional[str]) -> ApiKeyUser:
    if not SETTINGS.auth_enabled:
        return ApiKeyUser(user_id="anonymous", role="user")

    if not x_api_key or x_api_key not in SETTINGS.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    return SETTINGS.api_keys[x_api_key]


# ----------------------------
# Simple rate limiter (in-memory)
# ----------------------------
_rate_state: Dict[str, Dict[str, Any]] = {}


def _check_rate_limit(user: ApiKeyUser, policy: Policy) -> None:
    now = time.time()
    key = f"{user.user_id}:{user.role}"
    window = 60.0
    limit = policy.rate_limit_rpm_admin if user.role == "admin" else policy.rate_limit_rpm_user

    st = _rate_state.get(key)
    if st is None or now - st["start"] >= window:
        _rate_state[key] = {"start": now, "count": 1}
        return

    st["count"] += 1
    if st["count"] > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (requests per minute).")


# ----------------------------
# Initialize settings + embedder
# ----------------------------
SETTINGS = load_settings()
# ----------------------------
# Observability (Prometheus)
# ----------------------------
REQ_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["status", "role", "model"],
    )
ERR_TOTAL = Counter(
    "llm_errors_total",
    "Total LLM errors by type",
    ["type"],
    )
LAT_MS = Histogram(
    "llm_request_latency_ms",
    "LLM request latency in ms",
    ["role", "model"],
    buckets=(50, 100, 200, 400, 800, 1500, 3000, 6000, 12000),
    )
# CPU embedder for RAG (no vLLM embeddings)
# NOTE: lazy init so importing src.app won't trigger HF downloads in CI
_EMBEDDER = None

# ----------------------------
# Embeddings (CPU) for RAG
# ----------------------------
_EMBEDDER = None

def get_embedder():
    """
    Lazy import + lazy init.
    IMPORTANT: do not import sentence_transformers at module import time,
    otherwise CI will require torch.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer  # local import
        _EMBEDDER = SentenceTransformer(SETTINGS.embedding_model)
    return _EMBEDDER


async def embed_texts(texts: list[str]) -> np.ndarray:
    """
    CPU embeddings for RAG using SentenceTransformers.

    CI-friendly: if DISABLE_EMBEDDINGS=1, return zeros (deterministic) and
    avoids importing sentence_transformers/torch.
    """
    if os.getenv("DISABLE_EMBEDDINGS", "0") == "1":
        return np.zeros((len(texts), 384), dtype=np.float32)

    embedder = get_embedder()

    def _encode():
        return embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

    vecs = await asyncio.to_thread(_encode)
    return np.array(vecs, dtype=np.float32)




async def build_kb_index() -> None:
    global KB_CHUNKS, KB_EMB

    async with KB_LOCK:
        if not KB_DIR.exists():
            KB_CHUNKS = []
            KB_EMB = None
            logger.info("KB directory not found: %s", KB_DIR)
            return

        chunks: list[dict] = []

        for p in KB_DIR.glob("**/*"):
            if p.suffix.lower() not in [".md", ".txt"]:
                continue

            text = p.read_text(encoding="utf-8-sig", errors="ignore")
            if not text:
                continue

            for c_text, c_start in chunk_text(text, max_chars=320, overlap=80):
                section = extract_section_for_chunk(text, c_start)
                chunks.append({
                    "source": str(p),
                    "text": c_text,
                    "start": int(c_start),
                    "section": section,
                    "doc_type": infer_doc_type(str(p)),
                })

        KB_CHUNKS = chunks


        if not KB_CHUNKS:
            KB_EMB = None
            logger.info("KB ready: files=0, chunks=0, emb_shape=None")
            return


        KB_EMB = await embed_texts([c["text"] for c in KB_CHUNKS])

        logger.info(
            "KB ready: files=%d, chunks=%d, emb_shape=%s",
            len({c["source"] for c in KB_CHUNKS}),
            len(KB_CHUNKS),
            None if KB_EMB is None else KB_EMB.shape,
        )


# ----------------------------
# privacy-preserving hash
# ----------------------------
_ws = re.compile(r"\s+")


def compute_prompt_hash(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = str(m.get("role", "")).strip()
        content = str(m.get("content", ""))
        content = _ws.sub(" ", content).strip()
        parts.append(f"{role}:{content}")
    normalized = "\n".join(parts)
    full_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return full_hash[:16]


# ----------------------------
# Audit log (JSONL)
# ----------------------------
AUDIT_DIR = BASE_DIR / "artifacts" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_FILE = AUDIT_DIR / "audit.jsonl"


def write_audit(event: Dict[str, Any]) -> None:
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ----------------------------
# FastAPI app and routes
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _build():
        try:
            logger.info("KB indexing started (background).")
            await build_kb_index()
            logger.info("KB indexing finished.")
        except Exception as e:
            logger.exception("KB indexing failed (ignored): %s", e)

    asyncio.create_task(_build())
    logger.info("Startup complete (KB indexing in background).")
    yield


app = FastAPI(title="Internal LLM Gateway with RAG", version="0.2.0", lifespan=lifespan)

# ----------------------------
# CORS (for website playground)
# ----------------------------
allowed = os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
allowed = [x.strip() for x in allowed if x.strip()]

if allowed:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed,
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Content-Type", "x-api-key"],
    )


@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Internal LLM Gateway – Demo</title>
  <style>
    body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;max-width:900px;margin:40px auto;padding:0 16px;line-height:1.5}
    code,pre{background:#f6f8fa;padding:2px 6px;border-radius:6px}
    pre{padding:12px;overflow:auto}
    .card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:14px 0}
    a{color:#2563eb;text-decoration:none}
    a:hover{text-decoration:underline}
    .muted{color:#6b7280}
    .btn{background:#2563eb;color:white;padding:8px 16px;border-radius:6px;border:none;cursor:pointer;font-size:14px}
    .btn:hover{background:#1d4ed8}
    .demo-box{background:#f9fafb;border-radius:8px;padding:16px;margin:16px 0}
    .demo-response{background:#1e293b;color:#e2e8f0;padding:12px;border-radius:6px;font-family:monospace;white-space:pre-wrap;display:none}
    .loading{display:none;color:#6b7280;margin-left:8px}
    .error{color:#dc2626}
  </style>
</head>
<body>
  <h1>🚀 Internal LLM Gateway</h1>
  <p class="muted">Tool-first architecture + RAG + citations + audit logging. Deployed on Render.</p>

  <div class="card">
    <h3>📋 Quick Links</h3>
    <ul>
      <li><a href="/health">/health</a> (health check)</li>
      <li><a href="/docs">/docs</a> (interactive API docs)</li>
      <li><a href="/kb_status">/kb_status</a> (admin-only KB status)</li>
      <li><a href="/metrics">/metrics</a> (Prometheus metrics)</li>
    </ul>
  </div>

  <div class="card">
    <h3>🎮 Live Demo</h3>
    <p>Try it out instantly:</p>
    
    <div class="demo-box">
      <select id="demo-question" style="width:100%;padding:8px;margin-bottom:8px">
        <option value="Show me the SEV-2 checklist">📋 Show me the SEV-2 checklist</option>
        <option value="List open Sev-2 incidents">🔍 List open Sev-2 incidents</option>
        <option value="Why do we use embeddings?">🤔 Why do we use embeddings?</option>
        <option value="How should I handle a Sev-1 incident?">🚨 How should I handle a Sev-1 incident?</option>
      </select>
      <div style="display:flex;gap:8px;align-items:center">
        <button class="btn" onclick="runDemo()">Run Demo</button>
        <span class="loading" id="loading">Loading...</span>
      </div>
      <div id="response" class="demo-response"></div>
    </div>
  </div>

  <div class="card">
    <h3>💻 Example: PowerShell</h3>
    <pre>Invoke-RestMethod -Uri "https://YOUR-RENDER-URL/ask" -Method POST `
  -Headers @{"x-api-key"="dev-user-key"} -ContentType "application/json" `
  -Body '{"question":"Show me the SEV-2 checklist","k":3}' |
  ConvertTo-Json -Depth 8</pre>
    <p class="muted">Replace <code>YOUR-RENDER-URL</code> with your deployment domain.</p>
  </div>

  <div class="card">
    <h3>💻 Example: cURL</h3>
    <pre>curl -X POST "https://YOUR-RENDER-URL/ask" \
  -H "x-api-key: dev-user-key" \
  -H "Content-Type: application/json" \
  -d '{"question":"Show me the SEV-2 checklist","k":3}'</pre>
  </div>

  <div class="card">
    <h3>🎯 What to try</h3>
    <ul>
      <li><strong>Tool mode</strong>: <code>List open Sev-2 incidents</code> (works even if vLLM is down)</li>
      <li><strong>Tool + evidence</strong>: <code>Show me the SEV-2 checklist</code> (returns structured steps + citations)</li>
      <li><strong>RAG + LLM</strong>: <code>Why do we use embeddings?</code> (when KB and vLLM are enabled)</li>
      <li><strong>Admin only</strong>: Check <code>/kb_status</code> to see indexed documents</li>
    </ul>
  </div>

  <div class="card">
    <h3>🔧 Environment</h3>
    <ul>
      <li>DEMO_MODE: 0 (full functionality)</li>
      <li>Auth: Enabled (use x-api-key: dev-user-key)</li>
      <li>Model: Qwen/Qwen2.5-1.5B-Instruct</li>
      <li>Embeddings: all-MiniLM-L6-v2 (CPU)</li>
      <li>Rate limits: 30 rpm (user), 120 rpm (admin)</li>
    </ul>
  </div>

  <script>
    async function runDemo() {
      const question = document.getElementById('demo-question').value;
      const responseDiv = document.getElementById('response');
      const loadingSpan = document.getElementById('loading');
      
      responseDiv.style.display = 'none';
      loadingSpan.style.display = 'inline';
      
      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': 'dev-user-key'
          },
          body: JSON.stringify({ question, k: 3 })
        });
        
        const data = await res.json();
        
        if (res.ok) {
          // Pretty print the response
          const formatted = {
            request_id: data.request_id,
            answer: data.answer,
            sources: data.sources?.map(s => ({
              source: s.source,
              title: s.title,
              score: s.score,
              preview: s.preview.substring(0, 100) + '...'
            })),
            tool: data.tool?.used ? {
              used: data.tool.used,
              ok: data.tool.result?.ok
            } : null,
            timings_ms: data.timings_ms
          };
          
          responseDiv.textContent = JSON.stringify(formatted, null, 2);
          responseDiv.style.display = 'block';
        } else {
          throw new Error(data.detail || 'Request failed');
        }
      } catch (err) {
        responseDiv.textContent = 'Error: ' + err.message;
        responseDiv.style.display = 'block';
      } finally {
        loadingSpan.style.display = 'none';
      }
    }
  </script>
</body>
</html>
"""


@app.post("/reload_kb")
async def reload_kb(x_api_key: Optional[str] = Header(default=None)):
    user = auth_user(x_api_key)
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")

    logger.info("KB reload requested by user=%s role=%s", user.user_id, user.role)

    await build_kb_index()

    async with KB_LOCK:
        return {
            "status": "ok",
            "files": len({c["source"] for c in KB_CHUNKS}),
            "chunks": len(KB_CHUNKS),
            "emb_shape": None if KB_EMB is None else list(KB_EMB.shape),
        }

@app.get("/kb_status")
async def kb_status(x_api_key: Optional[str] = Header(default=None)):
    """
    Return current KB indexing status for debugging / demo.
    If you prefer it to be internal-only, keep it admin-only.
    """
    user = auth_user(x_api_key)

    
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")

    async with KB_LOCK:
        files = len({c["source"] for c in KB_CHUNKS})
        chunks = len(KB_CHUNKS)
        emb_shape = None if KB_EMB is None else list(KB_EMB.shape)

    return {
        "status": "ok",
        "kb_dir": str(display_source(str(KB_DIR))),
        "files": files,
        "chunks": chunks,
        "emb_shape": emb_shape,
    }

def _now_ms() -> int:
    # monotonic, higher precision than time.time()
    return int(time.perf_counter() * 1000)

def strip_fenced_json(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` wrappers if the model returns them.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        # remove first fence line
        t = t.split("\n", 1)[1] if "\n" in t else ""
        # remove trailing fence
        t = t.rsplit("```", 1)[0].strip()
    return t.strip()

def safe_parse_ui_json(raw_text: str) -> dict:
    """
    Parse the model response that should be strict JSON.
    Falls back to a safe object if parsing fails.
    """
    raw_text = strip_fenced_json(raw_text)
    obj = safe_parse_json_answer(raw_text)  # your existing parser
    # enforce keys for UI stability
    if not isinstance(obj, dict):
        return {"summary": "I'm not sure.", "steps": [], "notes": [], "confidence": 0.0}
    obj.setdefault("summary", "I'm not sure.")
    obj.setdefault("steps", [])
    obj.setdefault("notes", [])
    # optional confidence
    if "confidence" not in obj:
        obj["confidence"] = 0.5
    return obj


@app.post("/ask")
async def ask(request: Request, x_api_key: Optional[str] = Header(default=None)):
    request_id = str(uuid.uuid4())
    t0 = _now_ms()

    user = auth_user(x_api_key)

    status = "ok"
    model_name = SETTINGS.model  
    try:
        _check_rate_limit(user, SETTINGS.policy)

        payload = await request.json()
        question = str(payload.get("question", "")).strip()
        k = int(payload.get("k", 3))

        if not question:
            status = "blocked"
            raise HTTPException(status_code=400, detail="Missing question.")

        warnings: list[str] = []
        sources: list[dict] = []
        context = ""

        # ----------------------------
        # 0) Tool invocation (business layer)
        # ----------------------------
        tool_used = None
        tool_result = None

        routed = route_tool(question)
        if routed is not None:
            tool_name, tool_args = routed
            tool_used = tool_name

            # Execute tool (deterministic, no LLM)
            if tool_name == "incident.list_open":
                tool_result = list_open_incidents(question, limit=int(tool_args.get("limit", 10)))
            elif tool_name == "runbook.get_checklist":
                tool_result = get_sev_checklist(sev=int(tool_args.get("sev", 2)))
            else:
                tool_result = None
                warnings.append(f"Tool routed but not implemented: {tool_name}")

        # ----------------------------
        # DEMO MODE: tool-only, no LLM/RAG dependency
        # ----------------------------
        if DEMO_MODE and tool_result is None:
            # Demo: tool-only, no LLM/RAG dependency
            return {
                "request_id": request_id,
                "status": "blocked",
                "answer": {
                    "summary": "Demo mode is tool-first only. Try: 'Show me the SEV-2 checklist' or 'List open Sev-2 incidents'.",
                    "steps": [],
                    "notes": ["DEMO_MODE=1 keeps this demo reliable and cost-free (no LLM calls)."],
                    "confidence": 0.9
                },
                "sources": [],
                "tool": {"used": None, "result": None},
                "meta": {
                    "k": k,
                    "mode": "demo_tool_only",
                    "kb": {"dir": str(display_source(str(KB_DIR))), "files": 0, "chunks": 0},
                    "model": None,
                    "engine": "tool",
                },
                "timings_ms": {"embed": 0, "retrieve": 0, "llm": 0, "audit": 0, "total": (_now_ms() - t0)},
                "warnings": ["DEMO_MODE=1: LLM/RAG disabled"],
                "debug": {"prompt_hash": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16], "rag_enabled": False},
            }

        # ----------------------------
        # Deterministic tool response (skip LLM)
        # ----------------------------
        if tool_result is not None and tool_result.ok:
            # deterministic structured answer based on tool type
            if tool_used == "incident.list_open":
                answer = {
                    "summary": f"There are {len(tool_result.data.get('items', []))} open incidents.",
                    "steps": [
                        f"{it['id']} - {it['title']} (owner: {it.get('owner', 'unassigned')})"
                        for it in tool_result.data.get("items", [])
                    ],
                    "notes": [],
                    "confidence": 0.98
                }
            elif tool_used == "runbook.get_checklist":
                items = tool_result.data.get("checklist", [])
                answer = {
                    "summary": f"SEV{tool_result.data.get('sev', 2)} checklist retrieved from runbook.",
                    "steps": items[:8],
                    "notes": ["Use this checklist to stabilize service before deeper RCA."],
                    "confidence": 0.98
                }
            else:
                # Fallback for unknown tools
                answer = {
                    "summary": f"Tool {tool_used} executed successfully.",
                    "steps": [],
                    "notes": [],
                    "confidence": 0.95
                }

            # tool mode citations (lightweight, distributed across files)
            sources: list[dict] = []
            if getattr(tool_result, "citations_hint", None):
                # normalize hints to file basenames, keep stable order
                hint_files: list[str] = []
                seen_hint = set()
                for h in tool_result.citations_hint:
                    name = Path(h).name
                    if name and name not in seen_hint:
                        hint_files.append(name)
                        seen_hint.add(name)

                # Index KB chunks by source filename for quick lookup
                chunks_by_file: dict[str, list[tuple[int, dict]]] = {}
                for idx, ch in enumerate(KB_CHUNKS):
                    fname = Path(ch["source"]).name
                    chunks_by_file.setdefault(fname, []).append((idx, ch))

                def _make_card(idx: int, ch: dict) -> dict:
                    return {
                        "id": f"{display_source(ch['source'])}#{idx}",
                        "title": pretty_title(ch.get("doc_type", "Document"), ch["source"]),
                        "doc_type": ch.get("doc_type", "Document"),
                        "section": ch.get("section", ""),
                        "score": 1.0,  # tool-citation is "policy reference", not similarity
                        "source": Path(ch["source"]).name,
                        "preview": ch.get("text", "")[:260],
                        "chunk_id": idx,
                        "rank": 0,  # fill later
                    }

                # 1) First pass: take 1 chunk per hinted file (best coverage)
                used_files = set()
                used_chunk_ids = set()
                for fname in hint_files:
                    if fname not in chunks_by_file:
                        continue
                    # pick the first chunk for that file (stable)
                    idx, ch = chunks_by_file[fname][0]
                    card = _make_card(idx, ch)
                    sources.append(card)
                    used_files.add(fname)
                    used_chunk_ids.add(idx)
                    if len(sources) >= 3:
                        break

                # 2) Second pass: if still < 3, add more chunks from hinted files (next chunks)
                if len(sources) < 3:
                    for fname in hint_files:
                        if fname not in chunks_by_file:
                            continue
                        for idx, ch in chunks_by_file[fname][1:]:
                            if idx in used_chunk_ids:
                                continue
                            sources.append(_make_card(idx, ch))
                            used_chunk_ids.add(idx)
                            if len(sources) >= 3:
                                break
                        if len(sources) >= 3:
                            break

                # 3) Third pass: if still < 3, fallback to any other KB files (broad evidence)
                if len(sources) < 3:
                    for idx, ch in enumerate(KB_CHUNKS):
                        if idx in used_chunk_ids:
                            continue
                        fname = Path(ch["source"]).name
                        if fname in used_files:
                            continue
                        sources.append(_make_card(idx, ch))
                        used_files.add(fname)
                        used_chunk_ids.add(idx)
                        if len(sources) >= 3:
                            break

                # Final: assign ranks
                for r, s in enumerate(sources, start=1):
                    s["rank"] = r

            t_aud0 = _now_ms()
            # Audit
            q_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
            write_audit({
                "request_id": request_id,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "user_id": user.user_id,
                "role": user.role,
                "engine": "tool",
                "model": None,
                "prompt_hash": q_hash,
                "status": "ok",
                "latency_ms": (t_aud0 - t0),
                "rag": bool(sources),
                "rag_k": k,
                "rag_sources": [s["source"] for s in sources],
                "tool_used": tool_used,
            })
            t_aud1 = _now_ms()

            async with KB_LOCK:
                kb_files = len({c["source"] for c in KB_CHUNKS})
                kb_chunks = len(KB_CHUNKS)

            t1 = _now_ms()
            total_ms = max(1, t1 - t0)
            audit_ms = max(0, t_aud1 - t_aud0)

            # IMPORTANT: calculate total timing before return
            timings = {
                "embed": 0,
                "retrieve": 0,
                "llm": 0,
                "audit": audit_ms,
                "total": total_ms,
            }

            return {
                "request_id": request_id,
                "status": "ok",
                "answer": answer,
                "sources": sources,
                "tool": {
                    "used": tool_used,
                    "result": {
                        "tool_name": tool_result.tool_name,
                        "ok": tool_result.ok,
                        "data": tool_result.data,
                        "error": tool_result.error,
                        "citations_hint": getattr(tool_result, "citations_hint", None)
                    }
                },
                "meta": {
                    "k": k,
                    "mode": "tool",
                    "kb": {"dir": str(display_source(str(KB_DIR))), "files": kb_files, "chunks": kb_chunks},
                    "model": None,
                    "engine": "tool",
                },
                "timings_ms": timings,
                "warnings": warnings,
                "debug": {"prompt_hash": q_hash, "rag_enabled": False,"evidence_enabled": bool(sources), "tool_used": tool_used},
            }

        # ----------------------------
        # 1) RAG retrieval (only if no tool or tool failed)
        # ----------------------------
        t_embed0 = _now_ms()
        async with KB_LOCK:
            has_kb = (KB_EMB is not None and bool(KB_CHUNKS))

        if has_kb:
            q_emb = await embed_texts([question])
        t_embed1 = _now_ms()

        t_ret0 = _now_ms()
        if has_kb:
            async with KB_LOCK:
                hits = cosine_topk_with_scores(q_emb[0], KB_EMB, k=k)
                ctx_parts = []
                for rank, (i, score) in enumerate(hits, start=1):
                    chunk = KB_CHUNKS[i]
                    card = citation_card(chunk, score, i)
                    card["rank"] = rank
                    sources.append(card)

                    section = card.get("section") or ""
                    ctx_parts.append(
                        f"[{rank}] {card.get('source')} {section} (score={score:.4f}, chunk={i})\n"
                        f"{chunk.get('text','')}"
                    )
                context = "\n\n".join(ctx_parts)
        else:
            warnings.append("KB index is empty or not loaded. Answer may be ungrounded.")

        # tool context (prepend, so model sees it first) - for failed tools only
        if tool_result is not None and not tool_result.ok and tool_result.error:
            tool_block = (
                f"TOOL_ERROR ({tool_used}):\n"
                f"{tool_result.error}\n"
                "The tool failed. Answer based on other context if available.\n"
            )
            context = tool_block + "\n\n" + context

        t_ret1 = _now_ms()

        # ----------------------------
        # 2) LLM call (force UI JSON)
        # ----------------------------
        system_prompt = (
            "You are a helpful internal assistant.\n"
            "Use the provided Internal Context when relevant.\n"
            "Return STRICT JSON only (no markdown, no code fences) with EXACT keys:\n"
            "- summary: string\n"
            "- steps: array of strings\n"
            "- notes: array of strings\n"
            "- confidence: number between 0 and 1\n"
            "If the answer is not in the context, say you are not sure in summary and keep steps empty.\n"
            "Do not include any extra keys."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question:\n{question}\n\nInternal Context:\n{context}"},
        ]

        url = f"{SETTINGS.vllm_base_url}/v1/chat/completions"
        vllm_payload = {
            "model": SETTINGS.model,
            "messages": messages,
            "max_tokens": SETTINGS.policy.max_tokens_default,
            "temperature": 0.2,
        }

        t_llm0 = _now_ms()
        timeout = httpx.Timeout(SETTINGS.policy.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=vllm_payload)
            r.raise_for_status()
            out = r.json()
        t_llm1 = _now_ms()

        raw_answer = out["choices"][0]["message"]["content"]
        answer_obj = safe_parse_ui_json(raw_answer)

        # ----------------------------
        # 3) Audit (no raw prompt)
        # ----------------------------
        t_aud0 = _now_ms()
        q_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        write_audit({
            "request_id": request_id,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user_id": user.user_id,
            "role": user.role,
            "engine": "vllm",
            "model": SETTINGS.model,
            "prompt_hash": q_hash,
            "status": "ok",
            "latency_ms": (t_aud0 - t0),
            "rag": has_kb,
            "rag_k": k if has_kb else None,
            "rag_sources": list({s.get("source") for s in sources if s.get("source")}),
            "tool_used": tool_used,
        })
        t_aud1 = _now_ms()

        async with KB_LOCK:
            kb_files = len({c["source"] for c in KB_CHUNKS})
            kb_chunks = len(KB_CHUNKS)

        t1 = _now_ms()

        # Determine execution mode
        if tool_used:
            mode = "tool+llm"  # tool was used but failed, so fell back to LLM
        elif has_kb:
            mode = "rag+llm"
        else:
            mode = "llm"

        return {
            "request_id": request_id,
            "status": "ok",
            "answer": answer_obj,
            "sources": sources,
            "tool": {
                "used": tool_used,
                "result": None if tool_result is None else {
                    "tool_name": tool_result.tool_name,
                    "ok": tool_result.ok,
                    "data": tool_result.data,
                    "error": tool_result.error,
                }
            },
            "meta": {
                "k": k,
                "mode": mode,
                "kb": {"dir": str(display_source(str(KB_DIR))), "files": kb_files, "chunks": kb_chunks},
                "model": SETTINGS.model,
                "engine": "vllm",
            },
            "timings_ms": {
                "embed": (t_embed1 - t_embed0) if has_kb else 0,
                "retrieve": (t_ret1 - t_ret0) if has_kb else 0,
                "llm": (t_llm1 - t_llm0),
                "audit": (t_aud1 - t_aud0),
                "total": (t1 - t0),
            },
            "warnings": warnings,
            "debug": {"prompt_hash": q_hash, "rag_enabled": bool(has_kb)},
        }

    except HTTPException as e:
        # 400/401/403/429/… blocked or error
        if e.status_code in (400, 401, 403, 429):
            status = "blocked" if e.status_code != 429 else "blocked"
        else:
            status = "error"
        raise

    except httpx.ReadTimeout:
        status = "timeout"
        raise HTTPException(status_code=504, detail="Inference timeout.")

    except Exception:
        status = "error"
        raise

    finally:
        # Minimum interpretable metric: latency+number of requests
        latency_ms = _now_ms() - t0
        LAT_MS.labels(role=user.role, model=model_name).observe(latency_ms)
        REQ_TOTAL.labels(status=status, role=user.role, model=model_name).inc()





@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, x_api_key: Optional[str] = Header(default=None)):
    req_id = str(uuid.uuid4())
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start = time.time()

    user = auth_user(x_api_key)
    _check_rate_limit(user, SETTINGS.policy)

    payload = await request.json()

    # ---- policy: input length ----
    msgs = payload.get("messages", [])
    full_text = "".join([str(m.get("content", "")) for m in msgs])
    input_chars = len(full_text)
    prompt_hash = compute_prompt_hash(msgs)

    if input_chars > SETTINGS.policy.max_input_chars:
        write_audit({
            "request_id": req_id, "ts": ts,
            "user_id": user.user_id, "role": user.role,
            "engine": "vllm", "model": payload.get("model", ""),
            "prompt_hash": prompt_hash,
            "input_chars": input_chars,
            "status": "blocked", "reason": "max_input_chars",
        })
        REQ_TOTAL.labels(status="blocked", role=user.role, model=payload.get("model", "unknown")).inc()
        raise HTTPException(status_code=400, detail="Input too large for internal policy.")

    # ---- policy: max_tokens by role ----
    max_tokens = payload.get("max_tokens")
    if max_tokens is None:
        payload["max_tokens"] = SETTINGS.policy.max_tokens_admin if user.role == "admin" else SETTINGS.policy.max_tokens_default
    else:
        cap = SETTINGS.policy.max_tokens_admin if user.role == "admin" else SETTINGS.policy.max_tokens_default
        if int(max_tokens) > cap:
            payload["max_tokens"] = cap

    # ---- forward to vLLM OpenAI server ----
    url = f"{SETTINGS.vllm_base_url}/v1/chat/completions"
    model_name = payload.get("model", "unknown")

    status = "ok"
    err_type = ""
    try:
        timeout = httpx.Timeout(SETTINGS.policy.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                status = "error"
                err_type = f"downstream_{resp.status_code}"
                ERR_TOTAL.labels(type=err_type).inc()
                body = resp.text
                raise HTTPException(status_code=502, detail=f"Downstream error: {body[:200]}")
            data = resp.json()
    except httpx.ReadTimeout:
        status = "timeout"
        err_type = "timeout"
        ERR_TOTAL.labels(type=err_type).inc()
        raise HTTPException(status_code=504, detail="Inference timeout.")
    except httpx.ConnectError:
        status = "error"
        err_type = "connect_error"
        ERR_TOTAL.labels(type=err_type).inc()
        raise HTTPException(status_code=502, detail="Cannot reach inference backend.")
    finally:
        latency_ms = int((time.time() - start) * 1000)
        LAT_MS.labels(role=user.role, model=model_name).observe(latency_ms)
        REQ_TOTAL.labels(status=status, role=user.role, model=model_name).inc()

        write_audit({
            "request_id": req_id,
            "ts": ts,
            "user_id": user.user_id,
            "role": user.role,
            "engine": "vllm",
            "model": model_name,
            "prompt_hash": prompt_hash,
            "input_chars": input_chars,
            "max_tokens": payload.get("max_tokens"),
            "latency_ms": latency_ms,
            "status": status,
            "error_type": err_type or None,
        })

    return JSONResponse(content=data)


@app.post("/v1/chat/completions/stream")
async def chat_completions_stream(request: Request, x_api_key: Optional[str] = Header(default=None)):
    req_id = str(uuid.uuid4())
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start = time.time()

    user = auth_user(x_api_key)
    _check_rate_limit(user, SETTINGS.policy)

    payload = await request.json()

    # force stream
    payload["stream"] = True

    msgs = payload.get("messages", [])
    full_text = "".join([str(m.get("content", "")) for m in msgs])
    input_chars = len(full_text)

    prompt_hash = compute_prompt_hash(msgs)

    # policy: input length
    if input_chars > SETTINGS.policy.max_input_chars:
        write_audit({
            "request_id": req_id,
            "ts": ts,
            "user_id": user.user_id,
            "role": user.role,
            "engine": "vllm",
            "model": payload.get("model", "unknown"),
            "prompt_hash": prompt_hash,
            "input_chars": input_chars,
            "status": "blocked",
            "reason": "max_input_chars",
        })
        REQ_TOTAL.labels(status="blocked", role=user.role, model=payload.get("model", "unknown")).inc()
        raise HTTPException(status_code=400, detail="Input too large for internal policy.")

    # policy: max_tokens by role
    max_tokens = payload.get("max_tokens")
    if max_tokens is None:
        payload["max_tokens"] = SETTINGS.policy.max_tokens_admin if user.role == "admin" else SETTINGS.policy.max_tokens_default
    else:
        cap = SETTINGS.policy.max_tokens_admin if user.role == "admin" else SETTINGS.policy.max_tokens_default
        if int(max_tokens) > cap:
            payload["max_tokens"] = cap

    url = f"{SETTINGS.vllm_base_url}/v1/chat/completions"
    model_name = payload.get("model", "unknown")

    async def event_generator():
        nonlocal start
        status = "ok"
        err_type = ""

        try:
            timeout = httpx.Timeout(SETTINGS.policy.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code >= 400:
                        status = "error"
                        err_type = f"downstream_{resp.status_code}"
                        ERR_TOTAL.labels(type=err_type).inc()
                        text = await resp.aread()
                        raise HTTPException(status_code=502, detail=f"Downstream error: {text[:200]}")

                    async for chunk in resp.aiter_bytes():
                        yield chunk

        except httpx.ReadTimeout:
            status = "timeout"
            err_type = "timeout"
            ERR_TOTAL.labels(type=err_type).inc()
        except httpx.ConnectError:
            status = "error"
            err_type = "connect_error"
            ERR_TOTAL.labels(type=err_type).inc()
        finally:
            latency_ms = int((time.time() - start) * 1000)
            LAT_MS.labels(role=user.role, model=model_name).observe(latency_ms)
            REQ_TOTAL.labels(status=status, role=user.role, model=model_name).inc()

            write_audit({
                "request_id": req_id,
                "ts": ts,
                "user_id": user.user_id,
                "role": user.role,
                "engine": "vllm",
                "model": model_name,
                "prompt_hash": prompt_hash,
                "input_chars": input_chars,
                "max_tokens": payload.get("max_tokens"),
                "latency_ms": latency_ms,
                "status": status,
                "error_type": err_type or None,
                "stream": True,
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)