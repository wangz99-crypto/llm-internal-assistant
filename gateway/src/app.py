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
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from sentence_transformers import SentenceTransformer


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


# ----------------------------
# Configuration loading
# ----------------------------
def load_settings() -> Settings:
    config_path = os.environ.get("CONFIG_PATH", "configs/server.yaml")
    vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    auth_enabled = bool(cfg.get("auth", {}).get("enabled", True))
    api_keys_cfg = cfg.get("auth", {}).get("api_keys", [])

    api_keys: Dict[str, ApiKeyUser] = {}
    for item in api_keys_cfg:
        api_keys[item["key"]] = ApiKeyUser(user_id=item["user_id"], role=item["role"])

    pol = cfg.get("policy", {})
    policy = Policy(
        max_input_chars=int(pol.get("max_input_chars", 8000)),
        max_tokens_default=int(pol.get("max_tokens_default", 256)),
        max_tokens_admin=int(pol.get("max_tokens_admin", 512)),
        rate_limit_rpm_user=int(pol.get("rate_limit_rpm_user", 30)),
        rate_limit_rpm_admin=int(pol.get("rate_limit_rpm_admin", 120)),
        timeout_seconds=int(pol.get("timeout_seconds", 60)),
    )

    backend = cfg.get("backend", {})
    model = str(backend.get("model", "Qwen/Qwen2.5-1.5B-Instruct"))
    embedding_model = str(backend.get("embedding_model", model))

    return Settings(
        auth_enabled=auth_enabled,
        api_keys=api_keys,
        policy=policy,
        vllm_base_url=vllm_base_url.rstrip("/"),
        model=model,
        embedding_model=embedding_model,
    )


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
EMBEDDER = SentenceTransformer(SETTINGS.embedding_model)


async def embed_texts(texts: list[str]) -> np.ndarray:
    def _encode():
        vecs = EMBEDDER.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    return await asyncio.to_thread(_encode)



async def build_kb_index() -> None:
    global KB_CHUNKS, KB_EMB

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
    return int(time.time() * 1000)

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
    _check_rate_limit(user, SETTINGS.policy)

    payload = await request.json()
    question = str(payload.get("question", "")).strip()
    k = int(payload.get("k", 3))

    if not question:
        raise HTTPException(status_code=400, detail="Missing question.")

    warnings: list[str] = []
    sources: list[dict] = []
    context = ""

    # ----------------------------
    # 1) RAG retrieval
    # ----------------------------
    t_embed0 = _now_ms()
    async with KB_LOCK:
        has_kb = (KB_EMB is not None and bool(KB_CHUNKS))

    if has_kb:
        # embed question (outside lock)
        q_emb = await embed_texts([question])
    t_embed1 = _now_ms()

    t_ret0 = _now_ms()
    if has_kb:
        async with KB_LOCK:
            hits = cosine_topk_with_scores(q_emb[0], KB_EMB, k=k)

            ctx_parts = []
            for rank, (i, score) in enumerate(hits, start=1):
                chunk = KB_CHUNKS[i]
                card = citation_card(chunk, score, i)  # your function
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
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nInternal Context:\n{context}"
        },
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
        "latency_ms": (t_llm1 - t0),
        "rag": True,
        "rag_k": k,
        "rag_sources": list({s.get("source") for s in sources if s.get("source")}),
    })
    t_aud1 = _now_ms()

    # ----------------------------
    # 4) Meta / UI payload
    # ----------------------------
    async with KB_LOCK:
        kb_files = len({c["source"] for c in KB_CHUNKS})
        kb_chunks = len(KB_CHUNKS)

    t1 = _now_ms()

    return {
        "request_id": request_id,
        "status": "ok",
        "answer": answer_obj,
        "sources": sources,     # <- rename from citations to sources (UI friendly)
        "meta": {
            "k": k,
            "kb": {
                "dir": str(display_source(str(KB_DIR))),
                "files": kb_files,
                "chunks": kb_chunks,
            },
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
        "debug": {
            "prompt_hash": q_hash,
            "rag_enabled": bool(has_kb),
        },
    }




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
