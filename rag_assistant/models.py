from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    source_type: str
    title: str
    section: str
    department: str
    sensitivity: str
    allowed_roles: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    route: str
    retrieval_strategy: str
    target_stores: tuple[str, ...]
    target_departments: tuple[str, ...]
    target_sources: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    requires_strict_access: bool


@dataclass(frozen=True)
class IngestionReport:
    source_count: int
    chunk_count: int
    source_types: dict[str, int]
    departments: dict[str, int]
    sensitivities: dict[str, int]
    changed_sources: tuple[str, ...]
    skipped_sources: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    chunk: Chunk
    lexical_score: float
    semantic_score: float
    metadata_score: float
    rerank_score: float
    final_score: float
    reason: str
    diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str
    action: str
