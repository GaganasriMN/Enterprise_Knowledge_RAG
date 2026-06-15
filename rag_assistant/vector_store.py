from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from collections import Counter

from rag_assistant.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    VECTOR_BACKEND,
)
from rag_assistant.models import Chunk
from rag_assistant.text import tokenize


LOCAL_VECTOR_DIMENSIONS = 384


def _hash_bucket(term: str) -> int:
    return int(hashlib.sha256(term.encode("utf-8")).hexdigest(), 16) % LOCAL_VECTOR_DIMENSIONS


def local_embed_text(text: str) -> list[float]:
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * LOCAL_VECTOR_DIMENSIONS

    counts = Counter(tokens)
    vector = [0.0] * LOCAL_VECTOR_DIMENSIONS
    for term, count in counts.items():
        vector[_hash_bucket(term)] += 1.0 + math.log(count)

    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0 else [value / norm for value in vector]


def ollama_embed_text(text: str) -> list[float]:
    payload = json.dumps({"model": OLLAMA_EMBED_MODEL, "prompt": text}).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
    embedding = body.get("embedding")
    if not embedding:
        raise RuntimeError("Ollama returned no embedding")
    return embedding


def resolve_embedder() -> tuple[str, str]:
    if EMBEDDING_PROVIDER == "local":
        return "local_hash_fallback", f"hash-{LOCAL_VECTOR_DIMENSIONS}"
    if EMBEDDING_PROVIDER == "ollama":
        return "ollama", OLLAMA_EMBED_MODEL
    try:
        ollama_embed_text("health check")
        return "ollama", OLLAMA_EMBED_MODEL
    except (RuntimeError, urllib.error.URLError, TimeoutError, OSError):
        return "local_hash_fallback", f"hash-{LOCAL_VECTOR_DIMENSIONS}"


def embed_text(text: str, provider: str) -> list[float]:
    if provider == "ollama":
        return ollama_embed_text(text)
    return local_embed_text(text)


def chunk_document(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.section}\n{chunk.department}\n{chunk.source_type}\n{chunk.text}"


def build_chroma_vector_store(chunks: list[Chunk]) -> dict:
    provider, model = resolve_embedder()
    collection_name = f"{CHROMA_COLLECTION}_{provider}_{model.replace('-', '_').replace(':', '_')}"

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk_document(chunk) for chunk in chunks]
    embeddings = [embed_text(document, provider) for document in documents]
    metadatas = [
        {
            "source": chunk.source,
            "source_type": chunk.source_type,
            "department": chunk.department,
            "sensitivity": chunk.sensitivity,
            "allowed_roles": ",".join(chunk.allowed_roles),
            "section": chunk.section,
        }
        for chunk in chunks
    ]

    chromadb = None
    if VECTOR_BACKEND == "chroma":
        try:
            import chromadb as chromadb_module

            chromadb = chromadb_module
        except Exception:
            chromadb = None

    if VECTOR_BACKEND != "chroma" or chromadb is None:
        return {
            "provider": provider,
            "model": model,
            "collection_name": "in_memory_fallback",
            "client": None,
            "collection": None,
            "chunk_ids": set(ids),
            "embeddings": dict(zip(ids, embeddings)),
            "documents": dict(zip(ids, documents)),
            "metadatas": dict(zip(ids, metadatas)),
            "backend": "in_memory",
        }

    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        if ids:
            collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    except Exception:
        return {
            "provider": provider,
            "model": model,
            "collection_name": "in_memory_fallback",
            "client": None,
            "collection": None,
            "chunk_ids": set(ids),
            "embeddings": dict(zip(ids, embeddings)),
            "documents": dict(zip(ids, documents)),
            "metadatas": dict(zip(ids, metadatas)),
            "backend": "in_memory",
        }

    return {
        "provider": provider,
        "model": model,
        "collection_name": collection_name,
        "client": client,
        "collection": collection,
        "chunk_ids": set(ids),
        "backend": "chroma",
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def vector_scores(query: str, chunks: list[Chunk], vector_store: dict, top_k: int | None = None) -> dict[str, float]:
    if not chunks:
        return {}

    provider = vector_store["provider"]
    collection = vector_store["collection"]
    allowed_ids = {chunk.chunk_id for chunk in chunks}
    query_embedding = embed_text(query, provider)

    if vector_store.get("backend") == "in_memory" or collection is None:
        scored = []
        embeddings = vector_store.get("embeddings", {})
        for chunk in chunks:
            score = cosine_similarity(query_embedding, embeddings.get(chunk.chunk_id, []))
            scored.append((chunk.chunk_id, score))
        return dict(sorted(scored, key=lambda item: item[1], reverse=True)[: top_k or len(scored)])

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k or len(allowed_ids),
        include=["distances"],
    )

    scores = {}
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for chunk_id, distance in zip(ids, distances):
        if chunk_id in allowed_ids:
            scores[chunk_id] = max(0.0, 1.0 - float(distance))
    return scores
