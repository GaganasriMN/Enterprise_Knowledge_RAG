from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from rag_assistant.config import LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER, OPENAI_API_KEY
from rag_assistant.models import Candidate, QueryPlan
from rag_assistant.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE
from rag_assistant.text import tokenize


def confidence_label(results: list[Candidate]) -> tuple[str, int]:
    if not results:
        return "Low", 0
    total = sum(candidate.final_score for candidate in results)
    score = min(95, int(35 + total * 18 + len(results) * 6))
    if score >= 75:
        return "High", score
    if score >= 55:
        return "Medium", score
    return "Low", score


def pack_context(results: list[Candidate], budget_chars: int = 1600) -> str:
    packed = []
    used = 0
    for candidate in results:
        chunk = candidate.chunk
        entry = (
            f"[{chunk.chunk_id}] source={chunk.source}; section={chunk.section}; "
            f"sensitivity={chunk.sensitivity}\n{chunk.text}\n"
        )
        if used + len(entry) > budget_chars:
            continue
        packed.append(entry)
        used += len(entry)
    return "\n".join(packed)


def build_prompt(query: str, role: str, plan: QueryPlan, context: str) -> str:
    user_prompt = ANSWER_USER_TEMPLATE.format(
        role=role,
        intent=plan.intent,
        route=plan.route,
        query=query,
        context=context,
    )
    return f"System:\n{ANSWER_SYSTEM_PROMPT}\n\nUser:\n{user_prompt}"


def has_sufficient_evidence(query: str, results: list[Candidate]) -> bool:
    if not results:
        return False
    query_terms = {term for term in tokenize(query) if len(term) > 2}
    if not query_terms:
        return True
    context_terms = set(tokenize(" ".join(candidate.chunk.text for candidate in results[:3])))
    overlap = len(query_terms.intersection(context_terms)) / max(len(query_terms), 1)
    return results[0].final_score >= 0.45 and overlap >= 0.18


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 45) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _ollama_chat(prompt: str) -> str:
    body = _post_json(
        f"{LLM_BASE_URL.rstrip('/')}/api/chat",
        {
            "model": LLM_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt.split("User:\n", 1)[-1]},
            ],
        },
    )
    return body.get("message", {}).get("content", "").strip()


def _openai_chat(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    body = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt.split("User:\n", 1)[-1]},
            ],
            "temperature": 0.1,
        },
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
    )
    return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def local_grounded_answer(query: str, role: str, results: list[Candidate]) -> str:
    query_terms = set(tokenize(query))
    source_lines = []
    for candidate in results[:4]:
        chunk = candidate.chunk
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        best_sentences = sorted(
            sentences,
            key=lambda sentence: (
                len(query_terms.intersection(tokenize(sentence))),
                1 if any(term in tokenize(sentence) for term in {"avoid", "must", "not", "required"}) else 0,
            ),
            reverse=True,
        )[:3]
        evidence = " ".join(sentence.strip() for sentence in best_sentences if sentence.strip())
        if evidence:
            source_lines.append(f"- {evidence} [{chunk.chunk_id}]")
    if not source_lines:
        return (
            "The available permitted evidence is insufficient to answer this question. "
            "No grounded answer was generated."
        )
    return (
        f"Configured LLM provider is local deterministic mode. Based on permitted evidence for {role}:\n\n"
        + "\n".join(source_lines)
    )


def build_answer(query: str, role: str, results: list[Candidate], plan: QueryPlan | None = None) -> str:
    if not has_sufficient_evidence(query, results):
        return (
            "I do not have enough permitted evidence to answer this question reliably. "
            "No grounded answer was generated from the available context."
        )

    context = pack_context(results, budget_chars=2400)
    if plan is None:
        plan = QueryPlan("", "", "", tuple(), tuple(), tuple(), tuple(), False)
    prompt = build_prompt(query, role, plan, context)
    try:
        if LLM_PROVIDER == "ollama":
            answer = _ollama_chat(prompt)
        elif LLM_PROVIDER == "openai":
            answer = _openai_chat(prompt)
        else:
            answer = local_grounded_answer(query, role, results)
    except (RuntimeError, urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError) as error:
        answer = (
            f"LLM provider '{LLM_PROVIDER}' was unavailable ({error}). "
            "Falling back to deterministic grounded answer.\n\n"
            + local_grounded_answer(query, role, results)
        )

    cited_ids = {candidate.chunk.chunk_id for candidate in results}
    if not any(f"[{chunk_id}]" in answer for chunk_id in cited_ids):
        citations = ", ".join(f"[{candidate.chunk.chunk_id}]" for candidate in results[:3])
        answer = f"{answer}\n\nSources: {citations}"
    return answer
