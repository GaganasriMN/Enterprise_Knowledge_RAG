import re

from rag_assistant.models import Candidate, GuardrailDecision


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"bypass\s+(rbac|security|policy)",
    r"show\s+restricted",
    r"act\s+as\s+(admin|root|system)",
]

SENSITIVE_QUERY_PATTERNS = [
    r"\ball\s+salary\b",
    r"\bcompensation\s+records\b",
    r"\bexport\s+pii\b",
    r"\bshow\s+.*payroll\b",
]


def check_query_guardrails(query: str, role: str) -> GuardrailDecision:
    lowered = query.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailDecision(
                allowed=False,
                reason="Prompt-injection attempt detected in user query.",
                action="block_query",
            )

    for pattern in SENSITIVE_QUERY_PATTERNS:
        if re.search(pattern, lowered) and role not in {"HR", "Finance", "Admin"}:
            return GuardrailDecision(
                allowed=False,
                reason=f"Sensitive data request is not allowed for role {role}.",
                action="block_sensitive_query",
            )

    return GuardrailDecision(allowed=True, reason="Query passed guardrails.", action="allow")


def check_context_guardrails(results: list[Candidate]) -> GuardrailDecision:
    for candidate in results:
        lowered = candidate.chunk.text.lower()
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lowered):
                return GuardrailDecision(
                    allowed=False,
                    reason=f"Prompt-injection content detected in source {candidate.chunk.source}.",
                    action="remove_context",
                )
    return GuardrailDecision(allowed=True, reason="Retrieved context passed guardrails.", action="allow")


def filter_context(results: list[Candidate]) -> tuple[list[Candidate], GuardrailDecision]:
    decision = check_context_guardrails(results)
    if decision.allowed:
        return results, decision

    safe_results = []
    for candidate in results:
        lowered = candidate.chunk.text.lower()
        if not any(re.search(pattern, lowered) for pattern in PROMPT_INJECTION_PATTERNS):
            safe_results.append(candidate)
    return safe_results, decision
