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


# Move _ws definition BEFORE any function that uses it (clean_preview)
_ws = re.compile(r"\s+")


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


# STOPWORDS: added summarize, bullets, policy to reduce noise
STOPWORDS = {
    "the","a","an","to","of","and","or","in","on","for","with","is","are","be","as",
    "i","we","you","me","my","our","your","this","that","it","they","their",
    "what","why","how","should","do","does","first","please","give","explain","simply",
    "summarize","bullets","policy",  # added to prevent faq_ops from being boosted accidentally
}

# query -> preferred kb files (stability for demo)
# Stronger, broader escalation regex and higher boost
QUERY_FILE_BOOST = [
    (re.compile(r"\bescalat|\bsev\b|\bseverity\b|\bincident\b|\bpolicy\b", re.I),
     {"escalation_policy.md": 8.0, "incident_response.md": 3.0}),  # escalation gets huge boost
    (re.compile(r"\baudit\b|\blog\b|\btrace\b", re.I), {"audit_and_access.md": 4.0}),
    (re.compile(r"\bsev[- ]?1\b|\bsev[- ]?2\b|\boutage\b|\bincident\b|\bdisruption\b", re.I), {"incident_response.md": 3.0, "runbook.md": 2.0}),
    (re.compile(r"\bgateway\b|\brequest flow\b|\barchitecture\b", re.I), {"architecture_overview.md": 3.0}),
    (re.compile(r"\bverified\b|\bhybrid\b|\bmode\b", re.I), {"knowledge_usage_guidelines.md": 3.0}),
]

def _tokens(question: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9_-]{2,}", (question or "").lower())
    toks = [t for t in raw if t not in STOPWORDS]
    return toks[:24]

def keyword_topk(question: str, chunks: list[dict], k: int = 3) -> list[tuple[int, float]]:
    terms = _tokens(question)
    if not terms:
        return []

    scores: list[tuple[int, float]] = []
    q = question or ""

    for i, ch in enumerate(chunks):
        text = (ch.get("text", "") or "").lower()
        if not text:
            continue

        # base score: term frequency
        base = 0.0
        for t in terms:
            c = text.count(t)
            if c:
                base += min(3, c)  # cap
        if base <= 0:
            continue

        # section/title bonus (markdown headings are informative)
        section = (ch.get("section") or "").lower()
        for t in terms:
            if t in section:
                base += 1.2

        # file boosts for demo stability
        fname = Path(ch.get("source", "")).name
        for rx, boosts in QUERY_FILE_BOOST:
            if rx.search(q):
                base += boosts.get(fname, 0.0)

        scores.append((i, base))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[: max(1, min(k, len(scores)))]


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
DEMO_ADMIN_KEY = os.getenv("DEMO_ADMIN_KEY", "dev-admin-key")
DEMO_COOKIE_NAME = "demo_key"


# ----------------------------
# Demo Script (100% deterministic)
# ----------------------------
DEMO_SCRIPT_ENABLED = os.getenv("DEMO_SCRIPT", "1") == "1"  # default ON in demo

DEMO_SCENARIOS = {
    # TOOL-FIRST (always stable)
    "sev2_checklist": {
        "label": "Runbook: SEV-2 checklist (tool)",
        "canonical_question": "Show me the SEV-2 checklist",
        "force_engine": "tool",
        "force_tool": ("runbook.get_checklist", {"sev": 2}),
    },

    # KB-LOCKED (always hit the correct doc)
    "escalation_policy": {
        "label": "Policy: Escalation policy (KB)",
        "canonical_question": "Summarize the escalation policy in 5 bullets.",
        "force_engine": "kb_demo",
        "force_file": "escalation_policy.md",
    },
    "audit_logging": {
        "label": "Governance: What gets logged for audit and why? (KB)",
        "canonical_question": "What gets logged for audit and why?",
        "force_engine": "kb_demo",
        "force_file": "audit_and_access.md",
    },
    "sev1_response": {
        "label": "Incident: Handle SEV-1 (KB)",
        "canonical_question": "How should I handle a Sev-1 incident?",
        "force_engine": "kb_demo",
        "force_file": "incident_playbook_sev1.md",
    },
    "customer_outage_first": {
        "label": "Ops: Customer outage—what should I do first? (KB)",
        "canonical_question": "I have a service outage affecting customers. What should I do first?",
        "force_engine": "kb_demo",
        "force_file": "incident_response.md",
    },
    "customer_update_template": {
        "label": "Comms: Draft customer update (KB template)",
        "canonical_question": "Draft a customer update for a service disruption.",
        "force_engine": "kb_demo",
        "force_file": "incident_comms_templates.md",
    },
    "modes_guidance": {
        "label": "Usage: When should I use Verified vs Hybrid? (KB)",
        "canonical_question": "When should I use Verified vs Hybrid?",
        "force_engine": "kb_demo",
        "force_file": "knowledge_usage_guidelines.md",
    },
    "architecture_overview": {
        "label": "System: Architecture overview (KB)",
        "canonical_question": "Explain the gateway architecture and request flow in 6 bullets.",
        "force_engine": "kb_demo",
        "force_file": "architecture_overview.md",
    },
    "security_controls": {
        "label": "Security: Key controls (KB)",
        "canonical_question": "Summarize the security controls in 6 bullets.",
        "force_engine": "kb_demo",
        "force_file": "security_controls.md",
    },
    "product_overview": {
        "label": "Product: What is this system? (KB)",
        "canonical_question": "Give a concise product overview and the main guarantees.",
        "force_engine": "kb_demo",
        "force_file": "product_overview.md",
    },
}


def resolve_api_key(x_api_key: Optional[str], request: Optional[Request]) -> Optional[str]:
    # Prefer explicit header
    if x_api_key:
        return x_api_key

    # Demo cookie (HttpOnly)
    if request is not None:
        ck = request.cookies.get(DEMO_COOKIE_NAME)
        if ck:
            return ck

    return None


def auth_user(x_api_key: Optional[str], request: Optional[Request] = None) -> ApiKeyUser:
    if not SETTINGS.auth_enabled:
        return ApiKeyUser(user_id="anonymous", role="user")

    key = resolve_api_key(x_api_key, request)

    # Demo mode: if no key provided, treat as admin (for public product demo)
    if DEMO_MODE and not key:
        return ApiKeyUser(user_id="demo", role="admin")

    if not key or key not in SETTINGS.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    return SETTINGS.api_keys[key]


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
# Global exception handler (ensures JSON responses)
# ----------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error_type": type(exc).__name__},
    )


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


# 🔧 Patch 1 & 2: /demo endpoint now reads from DEMO_SCENARIOS
@app.get("/demo")
def demo_scenarios():
    return {
        "title": "Enterprise Gateway Demo Scenarios",
        "script_enabled": DEMO_SCRIPT_ENABLED,
        "scenarios": [
            {"id": sid, "label": cfg["label"], "question": cfg["canonical_question"]}
            for sid, cfg in DEMO_SCENARIOS.items()
        ],
    }


# Helper for template-based answers in demo mode (enterprise-style)
def demo_template_answer(question: str, steps: list[str], sources: list[dict]) -> dict:
    q = (question or "").lower()

    # label by intent (enterprise-style)
    if "sev-2" in q or "sev2" in q or "checklist" in q:
        title = "Runbook result (deterministic)."
        notes = ["This is a tool-first workflow: no LLM required."]
    elif "audit" in q or "logged" in q:
        title = "Governance answer (source-backed)."
        notes = ["Audit fields are policy-controlled and privacy-preserving."]
    elif "draft" in q or "customer update" in q:
        title = "Template retrieved (source-backed)."
        notes = ["In private deployments, the LLM can rewrite tone/length while keeping the same citations."]
    elif "escalation" in q or "policy" in q:
        title = "Policy summary (source-backed)."
        notes = ["In public demo, summarization is deterministic; private mode adds LLM reasoning."]
    else:
        title = "Source-backed deterministic answer."
        notes = ["This hosted demo runs with LLM disabled to emphasize reliability and governance."]

    # stable “top evidence” line
    if sources:
        top = sources[0]
        sec = top.get("section") or ""
        notes.insert(0, f"Top evidence: {top.get('source')} {sec}".strip())

    return {
        "summary": title,
        "steps": steps[:8] if steps else ["Open Sources below to review the exact matched internal snippet."],
        "notes": notes[:4],
        "confidence": 0.75 if steps else 0.55,
    }


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Internal Assistant — Governance-First AI System</title>
  <style>
    :root{
      --bg:#0b1220; --panel:#0f172a; --muted:#94a3b8; --text:#e5e7eb;
      --card:#111c33; --border:rgba(148,163,184,0.18);
      --btn:#2563eb; --btn2:#334155; --ok:#22c55e; --warn:#f59e0b;
      --code:#0b1020;
    }
    *{box-sizing:border-box;font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;}
    body{margin:0;background:linear-gradient(180deg,#070b14 0%,#0b1220 100%);color:var(--text);}
    a{color:#93c5fd;text-decoration:none}
    .wrap{max-width:1100px;margin:0 auto;padding:28px 18px 42px;}
    .hero{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:18px;}
    .title{font-size:22px;font-weight:800;letter-spacing:0.2px;}
    .sub{color:var(--muted);margin-top:6px;line-height:1.4}
    .badges{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
    .badge{display:inline-flex;gap:8px;align-items:center;padding:6px 10px;border:1px solid var(--border);
      background:rgba(17,28,51,0.6);border-radius:999px;color:#cbd5e1;font-size:13px}
    .dot{width:8px;height:8px;border-radius:50%}
    .dot.ok{background:var(--ok)} .dot.warn{background:var(--warn)}
    .grid{display:grid;grid-template-columns: 1.15fr 0.85fr;gap:16px;align-items:start;}
    @media (max-width: 980px){.grid{grid-template-columns:1fr}}
    .panel{background:rgba(15,23,42,0.82);border:1px solid var(--border);border-radius:16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);overflow:hidden}
    .panel-h{padding:14px 16px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
    .panel-h .h{font-weight:800}
    .panel-b{padding:16px}
    .muted{color:var(--muted)}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .input, .select, .num{
      background:rgba(2,6,23,0.6);border:1px solid var(--border);color:var(--text);
      border-radius:12px;padding:10px 12px;font-size:14px;outline:none;
    }
    textarea.input{width:100%;min-height:120px;resize:vertical;line-height:1.45}
    .select{padding:10px 12px}
    .num{width:92px}
    .btn{background:var(--btn);border:none;color:white;padding:10px 14px;border-radius:12px;
      font-weight:700;cursor:pointer}
    .btn.secondary{background:var(--btn2)}
    .btn.ghost{background:transparent;border:1px solid var(--border);color:#cbd5e1}
    .btn:disabled{opacity:.6;cursor:not-allowed}
    .try{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
    .try .btn{font-size:13px;padding:8px 10px}
    .pill{font-size:12px;color:#cbd5e1;background:rgba(2,6,23,0.55);border:1px solid var(--border);
      border-radius:999px;padding:6px 10px}
    .card{background:rgba(17,28,51,0.65);border:1px solid var(--border);border-radius:14px;padding:12px 12px;margin-top:12px}
    .card h4{margin:0 0 8px 0;font-size:14px}
    .kvs{display:grid;grid-template-columns: 150px 1fr;gap:6px 10px;font-size:13px}
    .kvs div:nth-child(odd){color:var(--muted)}
    .list{margin:8px 0 0 18px;color:#e2e8f0}
    .list li{margin:5px 0}
    .callout{margin-top:12px;padding:10px 12px;border-radius:12px;border:1px solid var(--border);
      background:rgba(2,6,23,0.45);color:#cbd5e1;font-size:13px;line-height:1.45}
    details{margin-top:12px}
    summary{cursor:pointer;color:#cbd5e1;font-weight:700}
    pre{background:var(--code);border:1px solid var(--border);padding:12px;border-radius:12px;overflow:auto;color:#d1d5db}
    .small{font-size:12px}
    .okline{color:#86efac}
    .warnline{color:#fbbf24}
    .errline{color:#fca5a5}
    .spacer{height:8px}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <!-- Landing area with more product-oriented language -->
        <div class="title">🧩 Internal Assistant — Governance-First AI System</div>
        <div class="sub">
          A system-oriented assistant that delivers <b>source-backed answers</b>,
          <b>action-ready runbooks</b>, and <b>auditable request traces</b>.
          Designed for reliability, compliance, and operational clarity.
        </div>
        <div class="badges">
          <div class="badge"><span class="dot ok"></span><b>Source-backed</b> <span class="muted">verify before acting</span></div>
          <div class="badge"><span class="dot ok"></span><b>Action steps</b> <span class="muted">runbooks & checklists</span></div>
          <div class="badge"><span class="dot ok"></span><b>Audit-ready</b> <span class="muted">request id + timings</span></div>
          <div class="badge"><span class="dot warn"></span><b>Hybrid</b> <span class="muted">best default mode</span></div>
        </div>
      </div>
      <div class="pill">
        Endpoint: <a href="/docs" target="_blank">/docs</a> · Health: <a href="/health" target="_blank">/health</a>
      </div>
    </div>

    <div class="grid">
      <!-- Left: Playground (renamed to Demo Runner) -->
      <div class="panel">
        <div class="panel-h">
          <div class="h">🎬 Demo Runner</div>
          <div class="muted small" id="statusLine">Ready</div>
        </div>
        <div class="panel-b">
          <div class="muted small">Pick a scenario below. This hosted demo runs in deterministic mode (LLM disabled) to showcase reliability, evidence, and auditability.</div>
          <div class="spacer"></div>
          <textarea id="q" class="input" placeholder="Scenario question will appear here" readonly></textarea>

          <!-- Quick preset dropdown (now using scenario_id) -->
          <div class="row" style="margin-top:10px">
            <span class="muted small">Select scenario</span>
            <select id="preset" class="select" style="flex:1;min-width:260px">
              <option value="">— Choose a scenario —</option>
              <option value="sev2_checklist">SEV-2 checklist (runbook)</option>
              <option value="sev1_response">SEV-1 response (incident)</option>
              <option value="escalation_policy">Escalation policy</option>
              <option value="customer_outage_first">Customer outage scenario</option>
              <option value="customer_update_template">Customer update draft</option>
              <option value="audit_logging">Audit logging overview</option>
              <option value="modes_guidance">Usage guidance</option>
              <option value="architecture_overview">Architecture overview</option>
              <option value="security_controls">Security controls</option>
              <option value="product_overview">Product overview</option>
            </select>
            <button class="btn secondary" onclick="applyPreset()">Load</button>
          </div>

          <div class="spacer"></div>
          <div class="row">
            <label class="muted">k</label>
            <input id="k" class="num" type="number" min="0" max="8" value="3" />
            <label class="muted">mode</label>
            <select id="mode" class="select">
              <option value="hybrid" selected>Hybrid (recommended)</option>
              <option value="kb">Verified (KB only)</option>
              <option value="chat">Assist: In private deployment, LLM-backed rewriting is enabled; hosted demo focuses on governance-first behavior.</option>
            </select>

            <button id="askBtn" class="btn" onclick="runAsk()">Run scenario</button>
            <button class="btn ghost" onclick="clearUI()">Clear</button>
          </div>

          <div class="try">
            <button class="btn secondary" onclick="quickAskScenario('escalation_policy')">Escalation policy</button>
            <button class="btn secondary" onclick="quickAskScenario('customer_outage_first')">Customer outage</button>
            <button class="btn secondary" onclick="quickAskScenario('sev2_checklist')">SEV-2 checklist</button>
            <button class="btn secondary" onclick="quickAskScenario('sev1_response')">Handle SEV-1</button>
            <button class="btn secondary" onclick="quickAskScenario('customer_update_template')">Customer update</button>
            <button class="btn secondary" onclick="quickAskScenario('audit_logging')">Audit logging</button>
            <button class="btn secondary" onclick="quickAskScenario('modes_guidance')">Usage guidance</button>
          </div>

          <div id="friendlyHint" class="callout" style="display:none"></div>

          <!-- Answer -->
          <div id="answerCard" class="card" style="display:none">
            <h4>🧠 Answer</h4>
            <div id="answerBody" class="small"></div>
          </div>

          <!-- Sources -->
          <div id="sourcesCard" class="card" style="display:none">
            <h4>📚 Sources</h4>
            <div id="sourcesBody" class="small"></div>
          </div>

          <!-- Request details -->
          <details id="techBox" style="display:none">
            <summary>🧾 Request details (click to expand)</summary>
            <div class="spacer"></div>
            <div id="techMeta" class="small"></div>
            <div class="spacer"></div>
            <button class="btn ghost" onclick="copyRaw()">Copy raw JSON</button>
            <div class="spacer"></div>
            <pre id="rawJson" class="small"></pre>
          </details>

          <!-- Error -->
          <div id="errorBox" class="card" style="display:none">
            <h4>❗ Error</h4>
            <pre id="errorText" class="small errline"></pre>
          </div>
        </div>
      </div>

      <!-- Right: How to use -->
      <div class="panel">
        <div class="panel-h">
          <div class="h">🧭 How to use</div>
          <div class="pill">Quick guidance</div>
        </div>
        <div class="panel-b">
          <div class="callout">
            <b>Usage tips</b>
            <ol class="list">
              <li>Start with <b>SEV-2 checklist</b> to see an actionable, step-by-step response with sources.</li>
              <li>Ask a general question in <b>Hybrid</b> mode to get the best balance of internal knowledge + explanation.</li>
              <li>Open <b>Request details</b> to view an audit-friendly record (request id + timings).</li>
            </ol>
          </div>

          <div class="card">
            <h4>Modes (plain English)</h4>
            <div class="kvs">
              <div>Hybrid</div>
              <div><b>Best for most questions.</b> Uses internal knowledge first; adds a clear explanation when needed.</div>

              <div>Verified</div>
              <div><b>Best for compliance.</b> Answers only when internal sources support the response.</div>

              <div>Assist</div>
              <div><b>Best for drafting.</b> More free-form help. (In this hosted demo, advanced LLM is limited; a GPU-backed version is available in private deployment.)</div>
            </div>
          </div>

          <div class="card">
            <h4>Business value</h4>
            <!-- Enhanced Business value block -->
            <ul class="list">
              <li><b>Operational clarity</b> — structured steps reduce ambiguity during incidents.</li>
              <li><b>Risk reduction</b> — answers are grounded in internal policies before action.</li>
              <li><b>Governance-first design</b> — every request is traceable and reviewable.</li>
            </ul>
          </div>

          <div class="card small muted">
            This demo focuses on reliability and traceability. In a private environment, the same design supports a GPU-backed LLM for richer reasoning while keeping the same safety controls.
          </div>
          <!-- Project highlights card (interview-friendly) -->
          <div class="card">
            <h4>Project highlights</h4>
            <ul class="list">
              <li>Retrieval + verification architecture</li>
              <li>Mode-based response strategy (Verified / Hybrid / Assist)</li>
              <li>Structured runbook outputs</li>
              <li>Audit-friendly logging (JSONL)</li>
              <li>Tool routing for deterministic workflows</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
  let DEMO_MAP = {};      // scenario_id -> {label, question}
  let SELECTED_SCENARIO = "";

  async function loadDemoScenarios(){
    try{
      const r = await fetch("/demo");
      const j = await r.json();
      (j.scenarios || []).forEach(s => { DEMO_MAP[s.id] = s; });
    }catch(e){
      // ignore
    }
  }

  // call on page load
  loadDemoScenarios();

  let LAST_RAW = "";

  function setStatus(msg){ document.getElementById("statusLine").textContent = msg; }

  function clearUI(){
    document.getElementById("friendlyHint").style.display = "none";
    document.getElementById("answerCard").style.display = "none";
    document.getElementById("sourcesCard").style.display = "none";
    document.getElementById("techBox").style.display = "none";
    document.getElementById("errorBox").style.display = "none";
    document.getElementById("answerBody").innerHTML = "";
    document.getElementById("sourcesBody").innerHTML = "";
    document.getElementById("techMeta").innerHTML = "";
    document.getElementById("rawJson").textContent = "";
    document.getElementById("errorText").textContent = "";
    LAST_RAW = "";
    setStatus("Ready");
  }

  function quickAskScenario(sid){
    SELECTED_SCENARIO = sid;
    const item = DEMO_MAP[sid];
    document.getElementById("q").value = (item && item.question) ? item.question : `Scenario: ${sid}`;
    runAsk();
  }

  function applyPreset(){
    const sid = document.getElementById("preset").value;
    if (!sid) return;

    SELECTED_SCENARIO = sid;

    const item = DEMO_MAP[sid];
    if (item && item.question){
      document.getElementById("q").value = item.question;
    } else {
      // fallback: show id if demo endpoint not loaded
      document.getElementById("q").value = `Scenario: ${sid}`;
    }
  }

  function esc(s){
    return (s||"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
  }

  function confLabel(x){
    if (typeof x !== "number") return null;
    if (x >= 0.8) return "High";
    if (x >= 0.55) return "Medium";
    return "Low";
  }

  function renderAnswer(data){
    const ans = data.answer;
    const meta = data.meta || {};
    const engine = (meta.engine || "").toLowerCase();
    const model = meta.model || null;

    // Friendly hint line (product language)
    let hint = "";
    // Enhanced product capability hint
    if (engine === "tool" || engine === "kb") {
      hint = "✅ Source-backed result — grounded in internal policies and runbooks.";
    } else {
      hint = "✨ Best-effort explanation: combines internal context (when available) with a clear summary.";
    }

    const hintBox = document.getElementById("friendlyHint");
    hintBox.innerHTML = esc(hint);
    hintBox.style.display = "block";

    // Build Answer UI (handles your tool-style object + string)
    let html = "";
    if (typeof ans === "string") {
      html += `<div>${esc(ans)}</div>`;
    } else if (ans && typeof ans === "object") {
      if (ans.summary) html += `<div><b>${esc(ans.summary)}</b></div>`;
      if (Array.isArray(ans.steps) && ans.steps.length){
        html += `<div class="spacer"></div><div class="muted">Steps</div><ol class="list">` +
          ans.steps.map(x=>`<li>${esc(x)}</li>`).join("") + `</ol>`;
      }
      if (Array.isArray(ans.notes) && ans.notes.length){
        html += `<div class="spacer"></div><div class="muted">Notes</div><ul class="list">` +
          ans.notes.map(x=>`<li>${esc(x)}</li>`).join("") + `</ul>`;
      }
      const cl = confLabel(ans.confidence);
      if (cl){
        html += `<div class="spacer"></div><div class="muted">Confidence</div><div>${esc(cl)}</div>`;
      }
    } else {
      html += `<div class="muted">No answer payload.</div>`;
    }

    document.getElementById("answerBody").innerHTML = html;
    document.getElementById("answerCard").style.display = "block";
  }

  function renderSources(data){
    const sources = data.sources || [];
    if (!sources.length){
      document.getElementById("sourcesCard").style.display = "none";
      return;
    }
    const html = sources.slice(0,3).map(s=>{
      const title = s.title ? ` — <span class="muted">${esc(s.title)}</span>` : "";
      const preview = s.preview ? esc(s.preview) : "";
      return `<div class="card" style="margin-top:10px">
        <div><b>${esc(s.source || "source")}</b>${title}</div>
        <div class="muted small" style="margin-top:6px;white-space:pre-wrap">${preview}</div>
      </div>`;
    }).join("");
    document.getElementById("sourcesBody").innerHTML = html;
    document.getElementById("sourcesCard").style.display = "block";
  }

  function renderTech(data){
    const meta = data.meta || {};
    const t = data.timings_ms || {};
    const tool = data.tool || {};
    const engine = meta.engine || "";
    const model = meta.model || "—";
    const rid = data.request_id || "—";

    const metaHtml = `
      <div class="kvs">
        <div>request_id</div><div>${esc(rid)}</div>
        <div>engine</div><div>${esc(engine)}</div>
        <div>model</div><div>${esc(String(model))}</div>
        <div>timings</div><div>${esc(JSON.stringify(t))}</div>
        <div>tool</div><div>${esc(tool.used || "—")}</div>
      </div>
    `;
    document.getElementById("techMeta").innerHTML = metaHtml;

    LAST_RAW = JSON.stringify(data, null, 2);
    document.getElementById("rawJson").textContent = LAST_RAW;
    document.getElementById("techBox").style.display = "block";
  }

  async function copyRaw(){
    if (!LAST_RAW) return;
    try{
      await navigator.clipboard.writeText(LAST_RAW);
      setStatus("Copied raw JSON ✅");
      setTimeout(()=>setStatus("Ready"), 1200);
    }catch(e){
      setStatus("Copy failed");
    }
  }

  async function runAsk(){
    const q = document.getElementById("q").value.trim();
    const k = Number(document.getElementById("k").value || 3);
    const mode = document.getElementById("mode").value;

    const scenarioId = SELECTED_SCENARIO || document.getElementById("preset").value || "";

    if (!scenarioId){
      document.getElementById("errorBox").style.display = "block";
      document.getElementById("errorText").textContent = "Please select a demo scenario.";
      return;
    }

    clearUI();

    const askBtn = document.getElementById("askBtn");
    askBtn.disabled = true;
    setStatus("Running scenario…");

    try{
      const headers = {"Content-Type":"application/json"};
      // No API key header - handled by cookie or demo mode

      const res = await fetch("/ask", {
        method:"POST",
        headers,
        body: JSON.stringify({ scenario_id: scenarioId, k:k, mode:mode })
      });

      // Always read text first, then try JSON.
      const text = await res.text();
      let data = null;
      try{ data = JSON.parse(text); }catch(_){}

      if (!res.ok){
        const msg = (data && (data.detail || data.message)) ? (data.detail || data.message) : text;
        throw new Error(msg || `Request failed (${res.status})`);
      }
      if (!data) throw new Error("Server returned non-JSON response.");

      renderAnswer(data);
      renderSources(data);
      renderTech(data);

      // status summary
      const eng = (data.meta && data.meta.engine) ? data.meta.engine : "ok";
      setStatus(`Done · engine=${eng}`);
    }catch(err){
      document.getElementById("errorBox").style.display = "block";
      document.getElementById("errorText").textContent = String(err && err.message ? err.message : err);
      setStatus("Error");
    }finally{
      askBtn.disabled = false;
    }
  }
</script>
</body>
</html>
    """
    resp = HTMLResponse(html)
    
    # Set demo cookie (HttpOnly) so browser carries it automatically
    if DEMO_MODE:
        # Determine if request is HTTPS to set secure flag appropriately
        is_https = True
        try:
            is_https = (request.url.scheme == "https")
        except Exception:
            pass

        resp.set_cookie(
            key=DEMO_COOKIE_NAME,
            value=DEMO_ADMIN_KEY,
            httponly=True,
            samesite="lax",
            secure=is_https,  # Only secure if actual connection is HTTPS
            max_age=60 * 60 * 24 * 7,
        )
    return resp


@app.post("/reload_kb")
async def reload_kb(x_api_key: Optional[str] = Header(default=None), request: Request = None):
    user = auth_user(x_api_key, request=request)
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
async def kb_status(x_api_key: Optional[str] = Header(default=None), request: Request = None):
    """
    Return current KB indexing status for debugging / demo.
    If you prefer it to be internal-only, keep it admin-only.
    """
    user = auth_user(x_api_key, request=request)

    
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

    user = auth_user(x_api_key, request=request)

    status = "ok"
    model_name = SETTINGS.model  
    try:
        _check_rate_limit(user, SETTINGS.policy)

        payload = await request.json()
        question = str(payload.get("question", "")).strip()
        k = int(payload.get("k", 3))
        mode = str(payload.get("mode", "hybrid")).strip().lower()
        if mode not in ("kb", "hybrid", "chat"):
            raise HTTPException(status_code=400, detail="Invalid mode. Use: kb | hybrid | chat.")

        warnings: list[str] = []  # move warnings up here

        # Public demo: "chat" becomes hybrid (LLM may be disabled)
        if DEMO_MODE and mode == "chat":
            warnings.append("Public demo: advanced LLM is not enabled in this hosted demo; switching to Hybrid.")
            mode = "hybrid"

        # 🔧 Patch 3: DEMO_SCRIPT mode forces scenario_id and canonical question
        forced_file = None
        forced_tool = None

        if DEMO_MODE and DEMO_SCRIPT_ENABLED:
            scenario_id = str(payload.get("scenario_id", "")).strip() or None
            if not scenario_id or scenario_id not in DEMO_SCENARIOS:
                raise HTTPException(status_code=400, detail="DEMO_SCRIPT enabled: select a scenario (scenario_id).")

            sc = DEMO_SCENARIOS[scenario_id]
            question = sc["canonical_question"]  # lock question text
            forced_file = sc.get("force_file")
            forced_tool = sc.get("force_tool")

        if not question:
            status = "blocked"
            raise HTTPException(status_code=400, detail="Missing question.")

        sources: list[dict] = []
        context = ""

        # ----------------------------
        # 0) Tool invocation (business layer)
        # ----------------------------
        tool_used = None
        tool_result = None

        # DEMO script tool overrides router (100% stable)
        if DEMO_MODE and DEMO_SCRIPT_ENABLED and forced_tool:
            tool_name, tool_args = forced_tool
            tool_used = tool_name

            if tool_name == "incident.list_open":
                tool_result = list_open_incidents(question, limit=int(tool_args.get("limit", 10)))
            elif tool_name == "runbook.get_checklist":
                tool_result = get_sev_checklist(sev=int(tool_args.get("sev", 2)))
            else:
                tool_result = None
                warnings.append(f"Forced tool not implemented: {tool_name}")

        else:
            routed = route_tool(question)
            if routed is not None:
                tool_name, tool_args = routed
                tool_used = tool_name

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
            # KB keyword retrieval (no embeddings, no LLM)
            async with KB_LOCK:
                kb_files = len({c["source"] for c in KB_CHUNKS})
                kb_chunks = len(KB_CHUNKS)
                # 🔧 Patch 3: first fetch more hits, then filter
                hits = keyword_topk(question, KB_CHUNKS, k=max(8, k))

                # 🔒 Force to a single KB file for demo stability
                if forced_file:
                    filtered = []
                    for idx, sc in hits:
                        if Path(KB_CHUNKS[idx].get("source", "")).name == forced_file:
                            filtered.append((idx, sc))
                    # if we have matches in that file, use them only
                    if filtered:
                        hits = filtered[:k]
                    else:
                        # hard fail to avoid "wrong doc" demos
                        hits = []

            sources = []
            ctx_lines = []
            for rank, (i, score) in enumerate(hits, start=1):
                ch = KB_CHUNKS[i]
                card = citation_card(ch, score, i)
                card["rank"] = rank
                sources.append(card)
                section = card.get("section") or ""
                ctx_lines.append(f"[{rank}] {card['source']} {section}")

            # Audit log for demo mode
            t_aud0 = _now_ms()
            q_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
            write_audit({
                "request_id": request_id,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "user_id": user.user_id,
                "role": user.role,
                "engine": "kb_demo",
                "model": None,
                "prompt_hash": q_hash,
                "status": "ok",
                "latency_ms": (_now_ms() - t0),
                "rag": True,
                "rag_k": k,
                "rag_sources": [s["source"] for s in sources],
                "tool_used": None,
            })
            t_aud1 = _now_ms()

            # Extract bullets from top chunk (product-oriented)
            def extract_bullets(text: str, max_items: int = 6) -> list[str]:
                """
                Turn a chunk into user-facing bullet steps.
                Prefers existing list items; falls back to short sentences.
                Filters out generic headings and favors actionable verbs.
                """
                if not text:
                    return []

                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                bullets = []
                # Skip lines that are just headings or generic descriptors
                BAD_PREFIX = ("#","##","purpose","this document","severity levels")
                # Action verbs to prioritize
                VERBS = ("run","check","notify","assign","declare","restart","validate","document","post","update")

                for ln in lines:
                    # Skip if line starts with bad prefix
                    if ln.lower().startswith(BAD_PREFIX):
                        continue
                    # markdown list / numbered list
                    if ln.startswith(("-", "*")):
                        candidate = ln.lstrip("-* ").strip()
                        bullets.append(candidate)
                    elif re.match(r"^\d+[\)\.] ", ln):
                        candidate = re.sub(r"^\d+[\)\.] ", "", ln).strip()
                        bullets.append(candidate)
                    if len(bullets) >= max_items:
                        return bullets

                # fallback: split sentences roughly, but only keep those with action verbs
                blob = " ".join(lines)
                parts = re.split(r"(?<=[\.\?\!])\s+", blob)
                for p in parts:
                    p = p.strip()
                    if p and len(p) >= 12 and any(v in p.lower() for v in VERBS):
                        bullets.append(p[:180])
                    if len(bullets) >= max_items:
                        break
                return bullets


            def best_source_note(src_cards: list[dict]) -> str:
                if not src_cards:
                    return "No internal sources were matched."
                top = src_cards[0]
                sec = top.get("section") or ""
                return f"Top evidence: {top.get('source')} {sec}".strip()


            # Smart hit selection: prefer specific files for certain queries
            def pick_best_hit(question: str, hits: list[tuple[int, float]], chunks: list[dict]) -> int | None:
                q = (question or "").lower()
                if not hits:
                    return None
                # Sev-1 specific: prefer incident_playbook_sev1.md
                if "sev-1" in q or "sev1" in q:
                    for i, sc in hits:
                        if Path(chunks[i].get("source","")).name == "incident_playbook_sev1.md":
                            return i
                # Customer update / draft: prefer incident_comms_templates.md
                if "customer update" in q or "draft" in q:
                    for i, sc in hits:
                        if Path(chunks[i].get("source","")).name == "incident_comms_templates.md":
                            return i
                # Default to top hit
                return hits[0][0]

            if sources:
                top_idx = pick_best_hit(question, hits, KB_CHUNKS) if hits else None
                top_text = KB_CHUNKS[top_idx]["text"] if top_idx is not None else ""
                steps = extract_bullets(top_text, max_items=6)

                # Use the enterprise-style template answer generator
                answer = demo_template_answer(question, steps, sources)
                warnings.append("DEMO_MODE=1: LLM disabled; deterministic KB retrieval used")
                status_out = "ok"
            else:
                # Optimized no-match prompt
                answer = {
                    "summary": "No internal source match found. Try one of the suggested scenarios below.",
                    "steps": [],
                    "notes": [
                        "Try: 'Show me the SEV-2 checklist'",
                        "Try: 'Handle a SEV-1 incident'",
                        "Try: 'Gateway overview'",
                        "Try: 'Why do we use embeddings?'"
                    ],
                    "confidence": 0.4,
                }
                status_out = "ok"
                warnings = []

            return {
                "request_id": request_id,
                "status": status_out,
                "answer": answer,
                "sources": sources,
                "tool": {"used": None, "result": None},
                "meta": {
                    "k": k,
                    "mode": "verified_kb_only",
                    "kb": {"dir": str(display_source(str(KB_DIR))), "files": kb_files, "chunks": kb_chunks},
                    "model": None,
                    "engine": "kb_demo",
                },
                "timings_ms": {"embed": 0, "retrieve": 0, "llm": 0, "audit": max(0, t_aud1 - t_aud0), "total": (_now_ms() - t0)},
                "warnings": warnings,
                "debug": {"prompt_hash": q_hash, "rag_enabled": True},
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
            
            if kb_chunks == 0:
                return {
                    "request_id": request_id,
                    "status": "ok",
                    "answer": {
                        "summary": "KB is still loading. Please retry in a few seconds.",
                        "steps": [],
                        "notes": ["This demo indexes markdown KB on startup."],
                        "confidence": 0.4,
                    },
                    "sources": [],
                    "tool": {"used": None, "result": None},
                    "meta": {"k": k, "mode": "demo_kb_only", "kb": {"dir": str(display_source(str(KB_DIR))), "files": kb_files, "chunks": kb_chunks}, "model": None, "engine": "kb_demo"},
                    "timings_ms": {"embed": 0, "retrieve": 0, "llm": 0, "audit": 0, "total": (_now_ms() - t0)},
                    "warnings": ["KB indexing in progress"],
                    "debug": {"rag_enabled": True},
                }

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
        # Mode controls how strictly we rely on retrieved context.
        if mode == "kb":
            system_prompt = (
                "You are a helpful internal assistant.\n"
                "You MUST rely only on the provided Internal Context.\n"
                "Return STRICT JSON only (no markdown, no code fences) with EXACT keys:\n"
                "- summary: string\n"
                "- steps: array of strings\n"
                "- notes: array of strings\n"
                "- confidence: number between 0 and 1\n"
                "If the answer is not in the context, say you are not sure in summary and keep steps empty.\n"
                "Do not include any extra keys."
            )
        elif mode == "chat":
            system_prompt = (
                "You are a helpful assistant.\n"
                "Return STRICT JSON only (no markdown, no code fences) with EXACT keys:\n"
                "- summary: string\n"
                "- steps: array of strings\n"
                "- notes: array of strings\n"
                "- confidence: number between 0 and 1\n"
                "Do not include any extra keys."
            )
        else:  # hybrid
            system_prompt = (
                "You are a helpful internal assistant.\n"
                "Use the provided Internal Context when it is relevant. If no context is provided, answer normally.\n"
                "Return STRICT JSON only (no markdown, no code fences) with EXACT keys:\n"
                "- summary: string\n"
                "- steps: array of strings\n"
                "- notes: array of strings\n"
                "- confidence: number between 0 and 1\n"
                "If you did not use any Internal Context, mention that in notes.\n"
                "Do not include any extra keys."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (f"Question:\n{question}\n\nInternal Context:\n{context}" if mode != "chat" else f"Question:\n{question}")},
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

        if mode == "hybrid" and not sources:
            # Make it explicit when we answered without internal citations.
            answer_obj.setdefault("notes", [])
            if isinstance(answer_obj["notes"], list):
                answer_obj["notes"].append("No internal KB sources were retrieved for this question; answer may be general.")


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

    user = auth_user(x_api_key, request=request)
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

    user = auth_user(x_api_key, request=request)
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