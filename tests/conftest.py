from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def corpus_and_index():
    from rag_assistant.ingestion import load_corpus_with_report
    from rag_assistant.retrieval import build_retrieval_index

    chunks, report = load_corpus_with_report()
    index = build_retrieval_index(chunks)
    return chunks, report, index
