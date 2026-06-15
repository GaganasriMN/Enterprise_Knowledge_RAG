from rag_assistant.models import Candidate


def candidate_to_score_text(candidate: Candidate) -> str:
    diagnostics = candidate.diagnostics or {}
    return (
        f"bm25_lexical={candidate.lexical_score:.2f}, chroma_vector={candidate.semantic_score:.2f}, "
        f"metadata={candidate.metadata_score:.2f}, reranked={candidate.rerank_score:.2f}\n"
        f"diagnostics={diagnostics}\n"
        f"why: {candidate.reason}"
    )
