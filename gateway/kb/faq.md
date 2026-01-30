# FAQ

## Q: What is RAG?
RAG = Retrieval-Augmented Generation:
1) retrieve relevant chunks from internal docs
2) provide them to the LLM as context
3) generate an answer grounded in those sources

## Q: Why do we need embeddings?
Embeddings convert text into vectors so we can do semantic search (cosine similarity).
This lets us find relevant policy/runbook sections even if wording differs.
