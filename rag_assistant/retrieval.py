from __future__ import annotations

from collections import Counter
import math
import re

from rag_assistant.models import Candidate, Chunk, QueryPlan
from rag_assistant.config import RETRIEVAL_WEIGHTS
from rag_assistant.reranking import cross_encoder_scores
from rag_assistant.text import tokenize
from rag_assistant.vector_store import build_chroma_vector_store, vector_scores


def chunk_text(chunk: Chunk) -> str:
    return f"{chunk.title} {chunk.section} {chunk.department} {chunk.source_type} {chunk.text}"


def build_retrieval_index(chunks: list[Chunk]) -> dict:
    document_terms = [tokenize(chunk_text(chunk)) for chunk in chunks]
    doc_count = len(document_terms)
    document_frequency = Counter(term for terms in document_terms for term in set(terms))
    idf = {
        term: math.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)
        for term, freq in document_frequency.items()
    }
    avg_doc_len = sum(len(terms) for terms in document_terms) / max(doc_count, 1)
    chunk_positions = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    tfidf_vectors = []
    vector_norms = []
    for terms in document_terms:
        frequencies = Counter(terms)
        vector = {term: (count / len(terms)) * idf.get(term, 0.0) for term, count in frequencies.items()}
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        tfidf_vectors.append(vector)
        vector_norms.append(norm)
    return {
        "terms": document_terms,
        "idf": idf,
        "avg_doc_len": avg_doc_len,
        "chunk_positions": chunk_positions,
        "tfidf_vectors": tfidf_vectors,
        "vector_norms": vector_norms,
        "vector_store": build_chroma_vector_store(chunks),
    }


def bm25_score(query_terms: list[str], chunk: Chunk, index: dict) -> float:
    chunk_position = index["chunk_positions"][chunk.chunk_id]
    terms = index["terms"][chunk_position]
    frequencies = Counter(terms)
    doc_len = len(terms)
    avg_doc_len = index["avg_doc_len"]
    k1 = 1.4
    b = 0.75
    score = 0.0
    for term in query_terms:
        tf = frequencies.get(term, 0)
        if tf == 0:
            continue
        idf = index["idf"].get(term, 0.0)
        denominator = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
        score += idf * (tf * (k1 + 1)) / denominator
    return score


def semantic_score(query: str, plan: QueryPlan, chunk: Chunk, index: dict) -> float:
    expanded_terms = tokenize(f"{query} {' '.join(plan.expanded_terms)}")
    if not expanded_terms:
        return 0.0
    query_frequencies = Counter(expanded_terms)
    query_vector = {
        term: (count / len(expanded_terms)) * index["idf"].get(term, 0.0)
        for term, count in query_frequencies.items()
    }
    query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
    chunk_position = index["chunk_positions"][chunk.chunk_id]
    chunk_vector = index["tfidf_vectors"][chunk_position]
    chunk_norm = index["vector_norms"][chunk_position]
    if query_norm == 0 or chunk_norm == 0:
        cosine = 0.0
    else:
        dot = sum(weight * chunk_vector.get(term, 0.0) for term, weight in query_vector.items())
        cosine = dot / (query_norm * chunk_norm)

    phrase_hits = sum(1 for phrase in plan.expanded_terms if " " in phrase and phrase in chunk_text(chunk).lower())
    return cosine + phrase_hits * 0.25


def metadata_score(plan: QueryPlan, chunk: Chunk) -> float:
    score = 0.0
    if chunk.department in plan.target_departments:
        score += 0.35
    if chunk.source_type in plan.target_sources:
        score += 0.2
    if plan.requires_strict_access and chunk.sensitivity in {"confidential", "restricted"}:
        score += 0.15
    return score


def store_route_score(plan: QueryPlan, chunk: Chunk) -> float:
    source_to_store = {
        "pdf_document": "document_store",
        "structured_report": "document_store",
        "compliance_record": "document_store",
        "json_metadata": "document_store",
        "json_log": "log_store",
        "csv": "sql_store",
        "sql_dump": "sql_store",
    }
    routed_store = source_to_store.get(chunk.source_type, "document_store")
    return 0.25 if routed_store in plan.target_stores else 0.0


def rerank_candidate(query: str, plan: QueryPlan, chunk: Chunk, base_score: float) -> tuple[float, str]:
    query_terms = set(tokenize(query))
    title_section_terms = set(tokenize(f"{chunk.title} {chunk.section}"))
    title_overlap = len(query_terms.intersection(title_section_terms))
    exact_entity_hit = bool(re.search(r"\b(INC-\d+|LOG-\d+|PRJ-\d+|Q[1-4])\b", query.upper()))
    reason_parts = []
    if chunk.department in plan.target_departments:
        reason_parts.append(f"department={chunk.department}")
    if chunk.source_type in plan.target_sources:
        reason_parts.append(f"source={chunk.source_type}")
    if title_overlap:
        reason_parts.append("title/section match")
    if exact_entity_hit:
        reason_parts.append("entity-aware match")

    rerank = base_score + title_overlap * 0.12 + (0.18 if exact_entity_hit else 0)
    return rerank, ", ".join(reason_parts) or "content similarity"


def build_candidate(
    query: str,
    chunk: Chunk,
    index: dict,
    plan: QueryPlan,
    chroma_scores: dict[str, float] | None = None,
    reranker_scores: dict[str, float] | None = None,
) -> Candidate:
    query_terms = tokenize(query)
    lexical = bm25_score(query_terms, chunk, index)
    semantic = semantic_score(query, plan, chunk, index)
    vector = (chroma_scores or {}).get(chunk.chunk_id, 0.0)
    reranker = (reranker_scores or {}).get(chunk.chunk_id, 0.0)
    meta = metadata_score(plan, chunk)
    store = store_route_score(plan, chunk)
    metadata_total = meta + store
    base = (
        lexical * RETRIEVAL_WEIGHTS["lexical"]
        + semantic * RETRIEVAL_WEIGHTS["semantic"]
        + vector * RETRIEVAL_WEIGHTS["vector"]
        + metadata_total * RETRIEVAL_WEIGHTS["metadata"]
        + reranker * RETRIEVAL_WEIGHTS["reranker"]
    )
    rerank, reason = rerank_candidate(query, plan, chunk, base)
    return Candidate(
        chunk=chunk,
        lexical_score=lexical,
        semantic_score=vector,
        metadata_score=metadata_total,
        rerank_score=rerank,
        final_score=rerank,
        reason=reason,
        diagnostics={
            "bm25": lexical,
            "tfidf_semantic": semantic,
            "chroma_vector": vector,
            "metadata_route": metadata_total,
            "cross_encoder": reranker,
            "final": rerank,
        },
    )


def retrieve(query: str, role: str, chunks: list[Chunk], index: dict, plan: QueryPlan, top_k: int = 4):
    allowed = [chunk for chunk in chunks if role in chunk.allowed_roles or role == "Admin"]
    denied = [chunk for chunk in chunks if chunk not in allowed]
    chroma_scores = vector_scores(query, allowed, index["vector_store"], top_k=max(len(allowed), top_k))
    reranker_scores = cross_encoder_scores(query, allowed, plan)

    ranked = sorted(
        (build_candidate(query, chunk, index, plan, chroma_scores, reranker_scores) for chunk in allowed),
        key=lambda candidate: candidate.final_score,
        reverse=True,
    )
    results = [candidate for candidate in ranked[:top_k] if candidate.final_score > 0]
    return results, allowed, denied


def restricted_match(query: str, denied: list[Chunk], index: dict, plan: QueryPlan) -> tuple[float, Chunk | None]:
    chroma_scores = vector_scores(query, denied, index["vector_store"], top_k=len(denied))
    reranker_scores = cross_encoder_scores(query, denied, plan)
    denied_scores = sorted(
        ((build_candidate(query, chunk, index, plan, chroma_scores, reranker_scores).final_score, chunk) for chunk in denied),
        key=lambda item: item[0],
        reverse=True,
    )
    return denied_scores[0] if denied_scores else (0.0, None)
