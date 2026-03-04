## Quick Answer (Audit Logging & Access)

1. Log request_id, timestamp, user_id/role, mode, status, and latency.
2. Do NOT log raw prompts; store only prompt hash/fingerprint where applicable.
3. Record which KB sources were referenced (file + section) for traceability.
4. Restrict access using role-based permissions; admin-only for sensitive endpoints.
5. Use logs for compliance reviews, incident retrospectives, and governance reporting.
6. Treat audit logs as security data; rotate keys and limit admin access.