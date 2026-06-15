from rag_assistant.config import INTENT_RULES
from rag_assistant.models import QueryPlan
from rag_assistant.text import tokenize


def create_query_plan(query: str) -> QueryPlan:
    query_terms = set(tokenize(query))
    scored_intents = []
    for intent, config in INTENT_RULES.items():
        matches = query_terms.intersection(config["terms"])
        scored_intents.append((len(matches), intent, config))

    score, intent, config = max(scored_intents, key=lambda item: item[0])
    if score == 0:
        intent = "general_enterprise_search"
        config = {
            "route": "general_search_route",
            "strategy": "all_allowed_sources_hybrid",
            "stores": ("document_store", "log_store", "sql_store", "vector_store"),
            "departments": tuple(),
            "sources": tuple(),
            "expansion": tuple(query_terms),
        }

    sensitive_terms = {"salary", "payroll", "compensation", "pii", "gdpr", "export", "audit"}
    return QueryPlan(
        intent=intent,
        route=config["route"],
        retrieval_strategy=config["strategy"],
        target_stores=tuple(config["stores"]),
        target_departments=tuple(config["departments"]),
        target_sources=tuple(config["sources"]),
        expanded_terms=tuple(dict.fromkeys([*query_terms, *config["expansion"]])),
        requires_strict_access=bool(query_terms.intersection(sensitive_terms)),
    )
