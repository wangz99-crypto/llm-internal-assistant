# gateway/src/tools/router.py
from __future__ import annotations
from typing import Optional, Tuple


def route_tool(question: str) -> Optional[Tuple[str, dict]]:
    q = (question or "").lower().strip()

    # ----------------------------
    # 1) Most specific intents first
    # ----------------------------
    if "checklist" in q and ("sev2" in q or "sev-2" in q or "sev 2" in q):
        return ("runbook.get_checklist", {"sev": 2})

    # Optional: allow "sev2 runbook" / "sev2 steps"
    if ("runbook" in q or "playbook" in q or "steps" in q) and ("sev2" in q or "sev-2" in q or "sev 2" in q):
        return ("runbook.get_checklist", {"sev": 2})

    # ----------------------------
    # 2) Broader / fallback tool intents
    # ----------------------------
    if ("incident" in q or "sev" in q or "oncall" in q):
        # list open incidents
        if ("list" in q or "open" in q) or ("show" in q and "checklist" not in q):
            return ("incident.list_open", {"limit": 10})

    return None