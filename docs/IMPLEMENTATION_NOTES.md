# Implementation Notes

## What Changed

- Expanded the corpus from a tiny demo set to a broader enterprise-style benchmark corpus.
- Added dataset provenance documentation.
- Added configurable LLM-backed generation with prompt templates.
- Added deterministic local generation fallback for demos without API credentials.
- Added configurable embedding and vector backends.
- Added optional cross-encoder reranking with a lightweight fallback reranker.
- Added retrieval diagnostics for BM25, TF-IDF-like semantic score, vector score, metadata routing score, reranker score, and final score.
- Replaced the original five-case smoke test with a reusable benchmark file and metrics.

## Defaults

The project defaults to reliable local execution:

- `RAG_EMBED_PROVIDER=local`
- `RAG_VECTOR_BACKEND=in_memory`
- `RAG_LLM_PROVIDER=local`
- `RAG_ENABLE_CROSS_ENCODER=false`

These defaults keep the demo fast and reproducible. Portfolio reviewers can enable external providers explicitly.

## Optional Provider Configuration

Use Ollama embeddings:

```cmd
set RAG_EMBED_PROVIDER=ollama
set OLLAMA_BASE_URL=http://localhost:11434
set OLLAMA_EMBED_MODEL=nomic-embed-text
```

Use Chroma persistence:

```cmd
set RAG_VECTOR_BACKEND=chroma
```

Use Ollama chat generation:

```cmd
set RAG_LLM_PROVIDER=ollama
set RAG_LLM_BASE_URL=http://localhost:11434
set RAG_LLM_MODEL=llama3.1
```

Use OpenAI-compatible chat generation:

```cmd
set RAG_LLM_PROVIDER=openai
set RAG_LLM_MODEL=gpt-4.1-mini
set OPENAI_API_KEY=your_key
```

Enable optional cross-encoder reranking:

```cmd
set RAG_ENABLE_CROSS_ENCODER=true
```

The cross-encoder path expects `sentence-transformers` to be installed. If loading fails, the system falls back to the deterministic reranker.

