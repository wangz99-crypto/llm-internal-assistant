import requests
import json

BASE = "http://localhost:8000"

scenarios = [
    "escalation_policy",
    "customer_outage_first",
    "sev2_checklist",
    "sev1_response",
    "customer_update_template",
    "audit_logging",
    "modes_guidance",
]

for sid in scenarios:
    print("=" * 80)
    print("SCENARIO:", sid)

    r = requests.post(
        f"{BASE}/ask",
        json={
            "scenario_id": sid,
            "k": 2,
            "mode": "hybrid"
        }
    )

    data = r.json()
    print(json.dumps(data, indent=2))