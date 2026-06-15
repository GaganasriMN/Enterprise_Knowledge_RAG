import json
import re

from rag_assistant.config import DATA_DIR
from rag_assistant.generation import build_answer
from rag_assistant.models import Candidate, Chunk
from rag_assistant.planner import create_query_plan
from rag_assistant.retrieval import retrieve
from rag_assistant.security import should_block_for_rbac


BENCHMARK_PATH = DATA_DIR / "benchmark_queries.json"


def load_benchmark_cases(path=BENCHMARK_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieved_relevance(results: list[Candidate], case: dict) -> list[bool]:
    relevant_sources = set(case.get("relevant_sources", []))
    relevant_chunk_ids = set(case.get("relevant_chunk_ids", []))
    relevance = []
    for candidate in results:
        chunk = candidate.chunk
        relevance.append(chunk.chunk_id in relevant_chunk_ids or chunk.source in relevant_sources)
    return relevance


def recall_at_k(results: list[Candidate], case: dict, k: int) -> float:
    expected = set(case.get("relevant_chunk_ids", [])) or set(case.get("relevant_sources", []))
    if not expected:
        return 1.0
    retrieved = {
        candidate.chunk.chunk_id
        for candidate in results[:k]
        if candidate.chunk.chunk_id in expected or candidate.chunk.source in expected
    }
    retrieved.update(
        candidate.chunk.source
        for candidate in results[:k]
        if candidate.chunk.source in expected
    )
    return min(1.0, len(retrieved) / len(expected))


def reciprocal_rank(results: list[Candidate], case: dict) -> float:
    for index, is_relevant in enumerate(_retrieved_relevance(results, case), start=1):
        if is_relevant:
            return 1.0 / index
    return 0.0


def context_precision(results: list[Candidate], case: dict) -> float:
    if not results:
        return 0.0
    relevance = _retrieved_relevance(results, case)
    return sum(1 for item in relevance if item) / len(relevance)


def citation_accuracy(answer: str, results: list[Candidate], case: dict) -> float:
    citations = set(re.findall(r"\[([A-Za-z0-9_.-]+)\]", answer))
    if not case.get("must_cite"):
        return 1.0 if not citations else 0.5
    if not citations:
        return 0.0
    retrieved_ids = {candidate.chunk.chunk_id for candidate in results}
    relevant_ids = set(case.get("relevant_chunk_ids", []))
    valid = citations.intersection(retrieved_ids)
    relevant = citations.intersection(relevant_ids)
    return (0.5 * len(valid) / len(citations)) + (0.5 * len(relevant) / len(citations))


def faithfulness(answer: str, results: list[Candidate], case: dict) -> float:
    lowered_answer = answer.lower()
    if not case.get("expected_allowed", True):
        refusal_terms = {"access denied", "insufficient", "permitted", "not have enough"}
        return 1.0 if any(term in lowered_answer for term in refusal_terms) else 0.0

    context = " ".join(candidate.chunk.text.lower() for candidate in results)
    expected_terms = [term.lower() for term in case.get("answer_terms", [])]
    if not expected_terms:
        return 1.0
    supported = sum(1 for term in expected_terms if term in lowered_answer and term in context)
    return supported / len(expected_terms)


def run_eval_suite(chunks: list[Chunk], index: dict) -> list[dict]:
    rows = []
    for case in load_benchmark_cases():
        role = case["role"]
        query = case["query"]
        plan = create_query_plan(query)
        results, _, denied = retrieve(query, role, chunks, index, plan)
        allowed_score = results[0].final_score if results else 0.0
        blocked, _, _ = should_block_for_rbac(query, denied, allowed_score, index, plan)
        answer = "Access denied." if blocked else build_answer(query, role, results, plan)
        actual_allowed = not blocked and bool(results)
        recall_4 = recall_at_k(results, case, 4)
        mrr = reciprocal_rank(results, case)
        citations = citation_accuracy(answer, results, case)
        grounded = faithfulness(answer, results, case)
        precision = context_precision(results, case)
        pass_case = (
            actual_allowed == case["expected_allowed"]
            and recall_4 >= (0.5 if case["expected_allowed"] else 0.0)
            and citations >= (0.5 if case.get("must_cite") else 0.5)
            and grounded >= 0.5
        )
        rows.append(
            {
                "id": case["id"],
                "role": role,
                "query": query,
                "intent": plan.intent,
                "expected_allowed": case["expected_allowed"],
                "actual_allowed": actual_allowed,
                "top_source": "BLOCKED" if blocked else (results[0].chunk.source if results else "NO_MATCH"),
                "recall@4": round(recall_4, 3),
                "mrr": round(mrr, 3),
                "citation_accuracy": round(citations, 3),
                "faithfulness": round(grounded, 3),
                "context_precision": round(precision, 3),
                "pass": pass_case,
            }
        )
    return rows


def summarize_eval(rows: list[dict]) -> dict:
    if not rows:
        return {}
    metric_names = ["recall@4", "mrr", "citation_accuracy", "faithfulness", "context_precision"]
    summary = {
        "cases": len(rows),
        "pass_rate": round(sum(1 for row in rows if row["pass"]) / len(rows), 3),
    }
    for metric in metric_names:
        summary[metric] = round(sum(float(row[metric]) for row in rows) / len(rows), 3)
    return summary
