# gateway/src/tools/runbook_tool.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class ToolResult:
    tool_name: str
    ok: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    citations_hint: Optional[List[str]] = None

SEV2_CHECKLIST = [
    "Confirm impact: elevated latency / partial failures (SEV2).",
    "Run: docker compose ps",
    "Check gateway logs: docker compose logs -n 200 gateway",
    "If vLLM is unhealthy: docker compose logs -n 200 vllm",
    "Mitigate: restart unhealthy service (docker compose restart <svc>)",
    "Validate: /health and a test /ask request",
    "Document actions in incident timeline."
]

def get_sev_checklist(sev: int = 2) -> ToolResult:
    if sev != 2:
        return ToolResult(
            tool_name="runbook.get_checklist",
            ok=False,
            data={},
            error=f"Unsupported sev={sev} in demo tool",
            citations_hint=["incident_response.md", "runbook.md"],
        )
    return ToolResult(
        tool_name="runbook.get_checklist",
        ok=True,
        data={"sev": sev, "checklist": SEV2_CHECKLIST},
        citations_hint=["incident_response.md", "runbook.md"],
    )