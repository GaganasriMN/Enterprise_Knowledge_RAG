from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_assistant.evaluation import run_eval_suite, summarize_eval
from rag_assistant.ingestion import load_corpus_with_report
from rag_assistant.retrieval import build_retrieval_index


def main() -> None:
    chunks, report = load_corpus_with_report()
    index = build_retrieval_index(chunks)
    rows = run_eval_suite(chunks, index)
    print("Ingestion")
    print(f"  sources: {report.source_count}")
    print(f"  chunks: {report.chunk_count}")
    print(f"  vector_backend: {index['vector_store'].get('backend')}")
    print(f"  embedding_provider: {index['vector_store'].get('provider')}")
    print("\nEvaluation summary")
    for key, value in summarize_eval(rows).items():
        print(f"  {key}: {value}")
    print("\nCases")
    for row in rows:
        status = "PASS" if row["pass"] else "FAIL"
        print(f"  {status} {row['id']} top_source={row['top_source']} recall@4={row['recall@4']} mrr={row['mrr']}")


if __name__ == "__main__":
    main()
