# Ops FAQ

## Q: Why do we need embeddings?
Embeddings convert text into vectors so we can:
1) Retrieve relevant internal KB chunks by semantic similarity
2) Provide those chunks as context to the LLM
3) Produce answers grounded in internal docs (with citations)

## Q: What embedding model do we use?
We use a SentenceTransformers model (CPU) configured by:
backend.embedding_model in server.yaml

Example:
- BAAI/bge-small-en-v1.5

## Q: Why not use vLLM embeddings API?
Some chat/instruct models do not support Embeddings API.
Using SentenceTransformers avoids dependency on backend embeddings support.

## Q: Why do citations include score?
Score is cosine similarity between query embedding and chunk embedding.
Higher means more relevant.

## Q: What is chunk_id?
chunk_id is the index of the chunk in the in-memory KB_CHUNKS list.
It helps correlate citations to stored chunks during debugging.

## Q: Why are there few chunks?
Because chunking depends on document length and chunk parameters:
- max_chars
- overlap

To increase chunks:
- add more KB files
- write richer docs
- lower max_chars slightly

## Q: What does /reload_kb do?
It rebuilds KB_CHUNKS and KB_EMB from the current KB directory.
Use it after editing KB files.

## Q: Who can call /reload_kb and /kb_status?
Admin only.
They require an API key mapped to role=admin in server.yaml.
