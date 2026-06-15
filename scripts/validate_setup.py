from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_assistant.evaluation import run_eval_suite, summarize_eval
from rag_assistant.generation import build_answer
from rag_assistant.ingestion import load_corpus_with_report
from rag_assistant.planner import create_query_plan
from rag_assistant.retrieval import build_retrieval_index, retrieve


def main() -> None:
    chunks, report = load_corpus_with_report()
    assert report.source_count >= 10, "Expected at least 10 corpus sources"
    assert report.chunk_count >= 50, "Expected at least 50 indexed chunks"

    index = build_retrieval_index(chunks)
    assert index["vector_store"]["provider"], "Embedding provider was not resolved"
    assert index["vector_store"].get("backend"), "Vector backend was not resolved"

    query = "How many days per week can employees work remotely?"
    plan = create_query_plan(query)
    results, _, _ = retrieve(query, "Employee", chunks, index, plan)
    assert results, "Expected at least one retrieval result"
    answer = build_answer(query, "Employee", results, plan)
    assert "[" in answer and "]" in answer, "Expected citation markers in answer"

    rows = run_eval_suite(chunks, index)
    summary = summarize_eval(rows)
    assert summary["pass_rate"] >= 0.8, "Evaluation pass rate below expected threshold"

    print("Setup validation passed")
    print(f"sources={report.source_count} chunks={report.chunk_count}")
    print(f"backend={index['vector_store'].get('backend')} provider={index['vector_store'].get('provider')}")
    print(summary)


if __name__ == "__main__":
    main()
