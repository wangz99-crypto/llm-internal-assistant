# Architecture

## Overview
This system has two services:
- gateway (FastAPI): auth, policy, RAG retrieval, audit logging
- vLLM (OpenAI-compatible server): runs the LLM for chat completions

## Data flow
1. Client calls /ask with a question
2. Gateway retrieves top-k KB chunks using embeddings + cosine similarity
3. Gateway sends question + retrieved context to vLLM /v1/chat/completions
4. Gateway returns answer + citations to the client

## Why RAG
RAG reduces hallucinations by grounding answers in internal documentation.
