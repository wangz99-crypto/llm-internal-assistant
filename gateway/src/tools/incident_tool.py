# gateway/src/tools/incident_tool.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import re
import time

@dataclass
class ToolResult:
    tool_name: str
    ok: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    citations_hint: Optional[List[str]] = None  # optional: suggest KB docs to cite

def _load_incidents() -> List[Dict[str, Any]]:
    """
    Demo data source: gateway/src/tools/data/incidents.json
    Keep it safe and non-sensitive.
    """
    data_path = Path(__file__).resolve().parent / "data" / "incidents.json"
    if not data_path.exists():
        return []
    return json.loads(data_path.read_text(encoding="utf-8"))

def _parse_sev(question: str) -> Optional[int]:
    m = re.search(r"\bsev[\s\-]?(\d)\b", question.lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def list_open_incidents(question: str, limit: int = 10) -> ToolResult:
    t0 = time.time()
    sev = _parse_sev(question)
    items = _load_incidents()

    # filter
    out = []
    for it in items:
        if it.get("status") != "open":
            continue
        if sev is not None and int(it.get("sev", -1)) != sev:
            continue
        out.append(it)

    out = out[: max(1, min(int(limit), 50))]

    return ToolResult(
        tool_name="incident.list_open",
        ok=True,
        data={
            "count": len(out),
            "filters": {"sev": sev, "status": "open"},
            "items": out,
            "latency_ms": int((time.time() - t0) * 1000),
        },
        citations_hint=["kb/incident_response.md", "kb/runbook.md"],
    )