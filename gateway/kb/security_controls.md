# Security Controls

## Data handling
- Do not store raw prompts in logs
- Store only prompt hash / fingerprint
- Record request_id, user_id, role, status, latency

## Access control
- All requests require API key (when enabled)
- Admin endpoints require admin role
  - POST /reload_kb
  - GET /kb_status

## Rate limiting
- User and admin have separate RPM limits
- In-memory limiter is acceptable for single-node PoC
- Future: migrate to Redis for multi-node deployments

## Observability
- Prometheus metrics:
  - request counts by role/status
  - error counts by type
  - latency histogram
- Audit log stored as JSONL

## Threat model (practical)
- Prevent unauthorized access via API key checks
- Prevent sensitive data leakage via prompt hashing
- Ensure auditability for compliance (who requested what, when)

## Secure operations
- Rotate API keys regularly
- Limit admin keys to operators only
- Keep KB documents non-sensitive unless encryption/controls are added

## Non-goals (for this PoC)
- Full IAM integration (SSO)
- Multi-tenant isolation
- Encrypted audit storage
