from datetime import datetime

import streamlit as st

from rag_assistant.config import ROLES
from rag_assistant.evaluation import run_eval_suite, summarize_eval
from rag_assistant.generation import build_answer, build_prompt, confidence_label, pack_context
from rag_assistant.guardrails import check_query_guardrails, filter_context
from rag_assistant.ingestion import load_corpus_with_report as load_corpus_with_report_uncached
from rag_assistant.planner import create_query_plan
from rag_assistant.retrieval import build_retrieval_index as build_retrieval_index_uncached
from rag_assistant.retrieval import retrieve
from rag_assistant.security import access_warning, should_block_for_rbac
from rag_assistant.ui import candidate_to_score_text


@st.cache_data
def load_corpus():
    return load_corpus_with_report_uncached()


@st.cache_resource
def build_retrieval_index(chunks):
    return build_retrieval_index_uncached(chunks)


def render_source(candidate):
    chunk = candidate.chunk
    with st.expander(f"{chunk.source} - {chunk.section} | score {candidate.final_score:.2f}"):
        st.write(chunk.text)
        st.caption(
            f"Type: {chunk.source_type} | Department: {chunk.department} | "
            f"Sensitivity: {chunk.sensitivity} | Roles: {', '.join(chunk.allowed_roles)}"
        )
        st.code(candidate_to_score_text(candidate))


def initialize_state():
    if "audit" not in st.session_state:
        st.session_state.audit = []


def main():
    st.set_page_config(page_title="Secure Enterprise RAG", page_icon="R", layout="wide")
    st.title("Secure Enterprise Knowledge Assistant")
    st.caption("AI-infra-first RAG: planning, hybrid retrieval, reranking, RBAC, context packing, and evaluation.")

    initialize_state()
    chunks, ingestion_report = load_corpus()
    index = build_retrieval_index(chunks)

    with st.sidebar:
        st.header("Access Context")
        role = st.selectbox("User role", ROLES)
        st.metric("Indexed chunks", len(chunks))
        st.metric("Raw sources", ingestion_report.source_count)
        st.metric("Retrieval engine", "Hybrid + reranking")
        st.caption(
            f"Vector backend: {index['vector_store'].get('backend', 'chroma')} | "
            f"Embeddings: {index['vector_store']['provider']} "
            f"({index['vector_store']['model']})"
        )
        with st.expander("Ingestion stats"):
            st.json(
                {
                    "source_types": ingestion_report.source_types,
                    "departments": ingestion_report.departments,
                    "sensitivities": ingestion_report.sensitivities,
                    "changed_sources": ingestion_report.changed_sources,
                    "skipped_sources": ingestion_report.skipped_sources,
                }
            )
        st.write("Available roles:")
        st.code(", ".join(ROLES))
        st.divider()
        st.write("Try:")
        st.caption("What is the remote work policy?")
        st.caption("What was Q4 revenue?")
        st.caption("Why did the payment API fail?")
        st.caption("What compliance controls exist for PII?")

    query = st.text_input("Ask an enterprise question", placeholder="Example: Why did the payment API fail?")
    submitted = st.button("Ask", type="primary")

    if submitted and query.strip():
        query_guardrail = check_query_guardrails(query, role)
        if not query_guardrail.allowed:
            st.error(query_guardrail.reason)
            st.session_state.audit.insert(
                0,
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "role": role,
                    "query": query,
                    "intent": "blocked_by_guardrail",
                    "sources": [],
                    "confidence": 100,
                    "restricted_matches_filtered": True,
                },
            )
            st.stop()

        plan = create_query_plan(query)
        results, allowed, denied = retrieve(query, role, chunks, index, plan)
        results, context_guardrail = filter_context(results)
        allowed_score = results[0].final_score if results else 0.0
        should_block, _, denied_chunk = should_block_for_rbac(query, denied, allowed_score, index, plan)

        if should_block:
            results = []

        label, confidence = confidence_label(results)
        warning = access_warning(query, role, denied, index, plan)
        context = pack_context(results)
        prompt = build_prompt(query, role, plan, context)

        if should_block:
            answer = (
                f"Access denied. The most relevant source belongs to the "
                f"{denied_chunk.department} domain and is not available to the {role} role. "
                "No restricted content was retrieved or used for generation."
            )
            confidence = 100
            label = "Policy enforced"
            prompt = build_prompt(query, role, plan, "RBAC policy blocked context assembly.")
        else:
            answer = build_answer(query, role, results, plan)

        st.session_state.audit.insert(
            0,
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "role": role,
                "query": query,
                "intent": plan.intent,
                "sources": [candidate.chunk.source for candidate in results],
                "confidence": confidence,
                "restricted_matches_filtered": bool(warning),
            },
        )

        left, right = st.columns([2, 1])
        with left:
            st.subheader("Answer")
            if warning:
                st.warning(warning)
            if not context_guardrail.allowed:
                st.warning(context_guardrail.reason)
            st.markdown(answer)

        with right:
            st.subheader("AI Pipeline Trace")
            st.metric("Confidence", f"{confidence}%", label)
            st.metric("Allowed chunks searched", len(allowed))
            st.metric("Restricted chunks filtered", len(denied))
            st.write("Query plan")
            st.json(
                {
                    "intent": plan.intent,
                    "route": plan.route,
                    "retrieval_strategy": plan.retrieval_strategy,
                    "target_stores": list(plan.target_stores),
                    "target_departments": list(plan.target_departments),
                    "target_sources": list(plan.target_sources),
                    "expanded_terms": list(plan.expanded_terms),
                    "strict_access": plan.requires_strict_access,
                    "query_guardrail": query_guardrail.reason,
                    "context_guardrail": context_guardrail.reason,
                }
            )

        st.subheader("Retrieved Evidence and Reranker Scores")
        if results:
            for candidate in results:
                render_source(candidate)
        else:
            st.info("No citation was shown because no permitted source had enough relevance.")

        with st.expander("Context Pack and Prompt Contract"):
            st.caption("This is the final context/prompt boundary that would be sent to an LLM in production.")
            st.code(prompt)

    st.divider()
    st.subheader("Offline Retrieval Evaluation")
    if st.button("Run evaluation suite"):
        eval_rows = run_eval_suite(chunks, index)
        summary = summarize_eval(eval_rows)
        passed = sum(1 for row in eval_rows if row["pass"])
        st.metric("Eval pass rate", f"{passed}/{len(eval_rows)}")
        st.json(summary)
        st.dataframe(eval_rows, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Audit Log")
    if st.session_state.audit:
        st.dataframe(st.session_state.audit, use_container_width=True, hide_index=True)
    else:
        st.info("Ask a question to create the first audit entry.")


if __name__ == "__main__":
    main()
