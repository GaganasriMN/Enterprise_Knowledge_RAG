from rag_assistant.evaluation import run_eval_suite, summarize_eval
from rag_assistant.generation import build_answer
from rag_assistant.ingestion import load_corpus_with_report
from rag_assistant.planner import create_query_plan
from rag_assistant.retrieval import build_retrieval_index, retrieve


def test_corpus_loads_and_indexes(corpus_and_index):
    _, report, index = corpus_and_index
    assert report.source_count >= 10
    assert report.chunk_count >= 50
    assert index["vector_store"].get("backend")
    assert index["vector_store"].get("provider")


def test_employee_hr_query_returns_cited_answer(corpus_and_index):
    chunks, _, index = corpus_and_index
    query = "How many days per week can employees work remotely?"
    plan = create_query_plan(query)
    results, _, _ = retrieve(query, "Employee", chunks, index, plan)
    answer = build_answer(query, "Employee", results, plan)

    assert results
    assert "hr_policy_handbook.pdf" == results[0].chunk.source
    assert "[" in answer and "]" in answer


def test_benchmark_pass_rate_is_acceptable(corpus_and_index):
    chunks, _, index = corpus_and_index
    rows = run_eval_suite(chunks, index)
    summary = summarize_eval(rows)

    assert summary["pass_rate"] >= 0.8
    assert summary["faithfulness"] >= 0.8
