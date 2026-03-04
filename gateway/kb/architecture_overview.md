## Quick Answer (Architecture in 6 bullets)

1. Client sends request to Gateway with auth header.
2. Gateway enforces auth, rate limits, and input/max-token policy.
3. For /ask, Gateway embeds the question and retrieves top-k KB chunks.
4. Gateway injects retrieved context and forwards to vLLM /chat/completions.
5. Gateway returns answer with citations (file/section/score) for transparency.
6. Gateway writes an audit event (JSONL) for traceability and governance.