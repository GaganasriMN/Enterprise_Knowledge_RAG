from __future__ import annotations

from rag_assistant.models import Chunk, QueryPlan
from rag_assistant.retrieval import restricted_match


def should_block_for_rbac(
    query: str,
    denied: list[Chunk],
    allowed_score: float,
    index: dict,
    plan: QueryPlan,
) -> tuple[bool, float, Chunk | None]:
    denied_score, denied_chunk = restricted_match(query, denied, index, plan)
    should_block = denied_chunk is not None and denied_score > 0.9 and denied_score > allowed_score * 1.15
    return should_block, denied_score, denied_chunk


def access_warning(query: str, role: str, denied: list[Chunk], index: dict, plan: QueryPlan) -> str | None:
    denied_score, chunk = restricted_match(query, denied, index, plan)
    if chunk and denied_score > 0.9:
        return (
            f"Some matching content exists in restricted {chunk.department} sources, "
            f"but it was excluded because your role is {role}."
        )
    return None
