from functools import lru_cache

from rag_assistant.config import ENABLE_CROSS_ENCODER
from rag_assistant.models import Chunk, QueryPlan
from rag_assistant.text import tokenize


@lru_cache(maxsize=1)
def _load_cross_encoder():
    if not ENABLE_CROSS_ENCODER:
        return None
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


def _heuristic_rerank_score(query: str, plan: QueryPlan, chunk: Chunk) -> float:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(f"{chunk.title} {chunk.section} {chunk.text}"))
    if not query_terms:
        return 0.0
    overlap = len(query_terms.intersection(text_terms)) / len(query_terms)
    phrase_hits = sum(1 for phrase in plan.expanded_terms if " " in phrase and phrase in chunk.text.lower())
    department_hit = 0.15 if chunk.department in plan.target_departments else 0.0
    source_hit = 0.10 if chunk.source_type in plan.target_sources else 0.0
    return min(1.0, overlap + phrase_hits * 0.12 + department_hit + source_hit)


def cross_encoder_scores(query: str, chunks: list[Chunk], plan: QueryPlan) -> dict[str, float]:
    model = _load_cross_encoder()
    if model is None:
        return {chunk.chunk_id: _heuristic_rerank_score(query, plan, chunk) for chunk in chunks}

    pairs = [(query, f"{chunk.title}\n{chunk.section}\n{chunk.text}") for chunk in chunks]
    raw_scores = list(model.predict(pairs))
    if not raw_scores:
        return {}
    minimum = min(raw_scores)
    maximum = max(raw_scores)
    spread = maximum - minimum
    if spread == 0:
        return {chunk.chunk_id: 0.5 for chunk in chunks}
    return {
        chunk.chunk_id: (float(score) - float(minimum)) / float(spread)
        for chunk, score in zip(chunks, raw_scores)
    }
