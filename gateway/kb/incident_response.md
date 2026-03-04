# Incident Response Runbook

## First actions during a customer-facing outage

1. Confirm the impact and scope of the outage.
2. Classify the incident severity (SEV-1 / SEV-2 / SEV-3).
3. Notify the on-call incident commander.
4. Check system health dashboards and recent deployments.
5. Review service logs for errors or anomalies.
6. Communicate initial status to stakeholders.

---

## System components involved
- Gateway (FastAPI)
- vLLM backend
- RAG indexing

## Severity levels
- SEV1: service down, all requests failing
- SEV2: partial degradation, high latency, some failures
- SEV3: minor issues, workaround exists