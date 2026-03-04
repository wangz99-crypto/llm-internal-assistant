import os
import json
import argparse
import requests

DEFAULT_SCENARIOS = [
    # TOOL-first
    "sev2_checklist",

    # KB locked
    "escalation_policy",
    "audit_logging",
    "sev1_response",
    "customer_outage_first",
    "customer_update_template",
    "modes_guidance",
    "architecture_overview",
    "security_controls",
    "product_overview",
]

CANONICAL_QUESTIONS = {
    # TOOL-first
    "sev2_checklist": "Show me the SEV-2 checklist",

    # KB locked
    "escalation_policy": "Summarize the escalation policy in 5 bullets.",
    "audit_logging": "What gets logged for audit and why?",
    "sev1_response": "How should I handle a Sev-1 incident?",
    "customer_outage_first": "I have a service outage affecting customers. What should I do first?",
    "customer_update_template": "Draft a customer update for a service disruption.",
    "modes_guidance": "When should I use Verified vs Hybrid?",
    "architecture_overview": "Explain the gateway architecture and request flow in 6 bullets.",
    "security_controls": "Summarize the security controls in 6 bullets.",
    "product_overview": "Give a concise product overview and the main guarantees.",
}

def pick(obj, path, default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def main():
    parser = argparse.ArgumentParser(description="Run all demo scenarios and print responses.")
    parser.add_argument("--base", default=os.getenv("GATEWAY_BASE", "http://localhost:8000"))
    parser.add_argument("--k", type=int, default=int(os.getenv("DEMO_K", "2")))
    parser.add_argument("--mode", default=os.getenv("DEMO_MODE_PARAM", "hybrid"), choices=["kb", "hybrid", "chat"])
    parser.add_argument("--scenarios", nargs="*", default=None, help="Override scenario list")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print full JSON response")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    scenarios = args.scenarios or DEFAULT_SCENARIOS

    s = requests.Session()

    # Auth strategy:
    # 1) If DEMO_ADMIN_KEY is provided, use x-api-key header
    # 2) Else, visit GET / to receive demo cookie (demo_key) in DEMO_MODE
    demo_admin_key = os.getenv("DEMO_ADMIN_KEY", "").strip()
    headers = {}
    if demo_admin_key:
        headers["x-api-key"] = demo_admin_key
    else:
        try:
            s.get(f"{base}/", timeout=10)  # triggers cookie set in DEMO_MODE
        except Exception:
            pass

    for sid in scenarios:
        print("=" * 92)
        print(f"SCENARIO: {sid}")

        question = CANONICAL_QUESTIONS.get(sid, "run demo")
        print(f"QUESTION: {question}")

        payload = {
            "question": question,   # ✅关键修复
            "scenario_id": sid,
            "k": args.k,
            "mode": args.mode,
        }

        try:
            r = s.post(f"{base}/ask", json=payload, headers=headers, timeout=30)
        except Exception as e:
            print("REQUEST ERROR:", type(e).__name__, str(e))
            continue

        print("HTTP:", r.status_code)

        try:
            data = r.json()
        except Exception:
            print("NON-JSON RESPONSE (first 800 chars):")
            print((r.text or "")[:800])
            continue

        if args.pretty:
            print(json.dumps(data, indent=2, ensure_ascii=False))
            continue

        # Compact human-readable output
        status = data.get("status")
        req_id = data.get("request_id")
        ans = data.get("answer") or {}
        summary = ans.get("summary", "")
        confidence = ans.get("confidence", None)
        steps = ans.get("steps") or []
        notes = ans.get("notes") or []

        meta = data.get("meta") or {}
        engine = meta.get("engine")
        mode = meta.get("mode")

        tool_used = pick(data, ["tool", "used"])
        sources = data.get("sources") or []

        print(f"status={status} request_id={req_id}")
        print(f"engine={engine} mode={mode} tool_used={tool_used} confidence={confidence}")
        print("\nSUMMARY:")
        print(f"- {summary}")

        if steps:
            print("\nSTEPS:")
            for i, x in enumerate(steps, 1):
                print(f"{i}. {x}")

        if notes:
            print("\nNOTES:")
            for i, x in enumerate(notes, 1):
                print(f"- {x}")

        if sources:
            print("\nSOURCES (top):")
            for src in sources[:3]:
                # your server returns cards with fields like: rank, source, section, score, preview
                print(
                    f"- [{src.get('rank')}] {src.get('source')} {src.get('section') or ''} "
                    f"(score={src.get('score')})"
                )

    print("=" * 92)
    print("DONE")

if __name__ == "__main__":
    main()