# Security

## Data handling
- Do not store raw prompts in logs
- Store only prompt fingerprints/hashes
- Log request_id, user_id, role, latency, and status

## Access control
- All requests require an API key
- Admin endpoints (reload/status) require admin role
