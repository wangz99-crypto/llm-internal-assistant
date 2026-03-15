#!/usr/bin/env python3
"""
Retrieval evaluation harness.

Measures hit@1 and hit@3 for the keyword retrieval path against a
ground-truth eval set, then reports overall, per-category, and
per-difficulty scores plus a failed-example list.

Usage (from repo root):
    PYTHONPATH=gateway python gateway/eval/retrieval_eval.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_FILE = SCRIPT_DIR / "eval_set.json"
KB_DIR = SCRIPT_DIR.parent / "kb"  # gateway/kb/

# ---------------------------------------------------------------------------
# Chunking helpers — intentionally aligned with chunk_text() and
# extract_section_for_chunk() in app.py (same params and logic).
# ---------------------------------------------------------------------------

# Intentionally aligned with _SECTION_RE in app.py
_SECTION_RE = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")


def _extract_section_for_chunk(full_text: str, chunk_start: int) -> str | None:
    # Intentionally aligned with extract_section_for_chunk() in app.py
    last = None
    for m in _SECTION_RE.finditer(full_text):
        if m.start() <= chunk_start:
            hashes = m.group(1)
            title = m.group(2).strip()
            last = f"{hashes} {title}"
        else:
            break
    return last


def _chunk_text(text: str, max_chars: int = 480, overlap: int = 120) -> list[tuple[str, int]]:
    # Intentionally aligned with chunk_text() in app.py
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


# ---------------------------------------------------------------------------
# KB loading — intentionally aligned with build_kb_index() in app.py.
# Produces the same chunk dict structure: {source, text, start, section, doc_type}
# ---------------------------------------------------------------------------

# Intentionally aligned with DOC_TYPE_MAP in app.py
_DOC_TYPE_MAP = {
    "policies.md": "Policy",
    "runbook.md": "Runbook",
    "onboarding.md": "Onboarding",
    "faq.md": "FAQ",
}


def load_kb_chunks(kb_dir: Path) -> list[dict]:
    """Glob kb_dir for .md/.txt files, chunk each, and return the chunk list."""
    chunks: list[dict] = []
    for p in sorted(kb_dir.glob("**/*")):
        if p.suffix.lower() not in (".md", ".txt"):
            continue
        text = p.read_text(encoding="utf-8-sig", errors="ignore")
        if not text:
            continue
        for c_text, c_start in _chunk_text(text, max_chars=480, overlap=120):
            section = _extract_section_for_chunk(text, c_start)
            chunks.append({
                "source": str(p),
                "text": c_text,
                "start": int(c_start),
                "section": section,
                "doc_type": _DOC_TYPE_MAP.get(p.name.lower(), "Document"),
            })
    return chunks


# ---------------------------------------------------------------------------
# Keyword retrieval — intentionally aligned with STOPWORDS, QUERY_FILE_BOOST,
# _tokens(), and keyword_topk() in app.py.
# ---------------------------------------------------------------------------

# Intentionally aligned with STOPWORDS in app.py
STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "with",
    "is", "are", "be", "as", "i", "we", "you", "me", "my", "our", "your",
    "this", "that", "it", "they", "their", "what", "why", "how", "should",
    "do", "does", "first", "please", "give", "explain", "simply",
    "summarize", "bullets", "policy",
}

# Intentionally aligned with QUERY_FILE_BOOST in app.py
QUERY_FILE_BOOST = [
    (
        re.compile(r"\bescalat|\bsev\b|\bseverity\b|\bincident\b|\bpolicy\b", re.I),
        {"escalation_policy.md": 8.0, "incident_response.md": 3.0},
    ),
    (
        re.compile(r"\baudit\b|\blog\b|\btrace\b", re.I),
        {"audit_and_access.md": 4.0},
    ),
    (
        re.compile(r"\bsev[- ]?1\b|\bsev[- ]?2\b|\boutage\b|\bincident\b|\bdisruption\b", re.I),
        {"incident_response.md": 3.0, "runbook.md": 2.0},
    ),
    (
        re.compile(r"\bgateway\b|\brequest flow\b|\barchitecture\b", re.I),
        {"architecture_overview.md": 3.0},
    ),
    (
        re.compile(r"\bverified\b|\bhybrid\b|\bmode\b", re.I),
        {"knowledge_usage_guidelines.md": 3.0},
    ),
]


def _tokens(question: str) -> list[str]:
    # Intentionally aligned with _tokens() in app.py
    raw = re.findall(r"[A-Za-z0-9_-]{2,}", (question or "").lower())
    toks = [t for t in raw if t not in STOPWORDS]
    return toks[:24]


def _keyword_topk(question: str, chunks: list[dict], k: int) -> list[tuple[int, float]]:
    # Intentionally aligned with keyword_topk() in app.py
    terms = _tokens(question)
    if not terms:
        return []

    scores: list[tuple[int, float]] = []
    q = question or ""

    for i, ch in enumerate(chunks):
        text = (ch.get("text", "") or "").lower()
        if not text:
            continue

        # base score: term frequency, capped at 3 per term
        base = 0.0
        for t in terms:
            c = text.count(t)
            if c:
                base += min(3, c)
        if base <= 0:
            continue

        # section/title bonus
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


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _retrieve_source_files(question: str, chunks: list[dict], k: int) -> list[str]:
    """Return up to k unique source filenames (basename) in ranked order."""
    # Oversample then deduplicate to ensure we surface k distinct files
    raw = _keyword_topk(question, chunks, k=k * 6)
    seen: list[str] = []
    for idx, _ in raw:
        fname = Path(chunks[idx]["source"]).name
        if fname not in seen:
            seen.append(fname)
        if len(seen) >= k:
            break
    return seen


def evaluate(eval_examples: list[dict], chunks: list[dict]) -> list[dict]:
    results = []
    for ex in eval_examples:
        question = ex["question"]
        expected = ex["expected_source_file"]

        top1 = _retrieve_source_files(question, chunks, k=1)
        top3 = _retrieve_source_files(question, chunks, k=3)

        results.append({
            "id": ex["id"],
            "category": ex["category"],
            "difficulty": ex["difficulty"],
            "question": question,
            "expected": expected,
            "top1": top1[0] if top1 else None,
            "top3": top3,
            "hit@1": expected in top1,
            "hit@3": expected in top3,
        })
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(num: int, den: int) -> str:
    return f"{100 * num // den}%" if den else "n/a"


def report(results: list[dict]) -> None:
    n = len(results)
    h1 = sum(r["hit@1"] for r in results)
    h3 = sum(r["hit@3"] for r in results)

    print("=" * 54)
    print("  RETRIEVAL EVAL RESULTS")
    print("=" * 54)

    print(f"\nOverall ({n} examples)")
    print(f"  hit@1 : {h1}/{n}  ({_pct(h1, n)})")
    print(f"  hit@3 : {h3}/{n}  ({_pct(h3, n)})")

    # Per-category
    cats: dict[str, list] = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r)

    print("\nhit@3 by category:")
    for cat in sorted(cats):
        items = cats[cat]
        c_h3 = sum(r["hit@3"] for r in items)
        c_n = len(items)
        print(f"  {cat:<24} {c_h3}/{c_n}  ({_pct(c_h3, c_n)})")

    # Per-difficulty
    diffs: dict[str, list] = defaultdict(list)
    for r in results:
        diffs[r["difficulty"]].append(r)

    print("\nhit@3 by difficulty:")
    for diff in ["easy", "medium", "hard"]:
        if diff not in diffs:
            continue
        items = diffs[diff]
        d_h3 = sum(r["hit@3"] for r in items)
        d_n = len(items)
        print(f"  {diff:<10} {d_h3}/{d_n}  ({_pct(d_h3, d_n)})")

    # Failed examples
    failed = [r for r in results if not r["hit@3"]]
    if not failed:
        print("\nAll examples passed hit@3.")
    else:
        print(f"\nFailed examples  ({len(failed)}/{n} missed hit@3):")
        for r in failed:
            hit1_marker = " [hit@1 also missed]" if not r["hit@1"] else " [hit@1 ok]"
            print(f"\n  [{r['id']}] ({r['category']} / {r['difficulty']}){hit1_marker}")
            print(f"    question  : {r['question']}")
            print(f"    expected  : {r['expected']}")
            print(f"    top-1     : {r['top1']}")
            print(f"    top-3     : {r['top3']}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"KB dir    : {KB_DIR}")
    print(f"Eval file : {EVAL_FILE}\n")

    chunks = load_kb_chunks(KB_DIR)
    unique_files = len({Path(c["source"]).name for c in chunks})
    print(f"Loaded {len(chunks)} chunks from {unique_files} files.\n")

    eval_examples = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    print(f"Eval set  : {len(eval_examples)} examples\n")

    results = evaluate(eval_examples, chunks)
    report(results)
