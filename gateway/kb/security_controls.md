## Quick Answer (Security Controls)

1. Require API keys for requests when auth is enabled.
2. Enforce role-based access (admin-only endpoints like reload/status).
3. Apply rate limiting per role to prevent abuse.
4. Enforce request size limits (max_input_chars) and token caps.
5. Write audit logs for traceability (avoid raw prompt storage).
6. Expose metrics for monitoring (counts, errors, latency).