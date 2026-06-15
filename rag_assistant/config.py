from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
ARTIFACTS_DIR = APP_DIR / "artifacts"
CHROMA_DIR = ARTIFACTS_DIR / "chroma"
CHROMA_COLLECTION = "enterprise_rag_chunks"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBEDDING_PROVIDER = os.getenv("RAG_EMBED_PROVIDER", "local")
VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "in_memory")
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "local")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama3.1")
LLM_BASE_URL = os.getenv("RAG_LLM_BASE_URL", OLLAMA_BASE_URL)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENABLE_CROSS_ENCODER = os.getenv("RAG_ENABLE_CROSS_ENCODER", "false").lower() == "true"

RETRIEVAL_WEIGHTS = {
    "lexical": float(os.getenv("RAG_WEIGHT_LEXICAL", "0.45")),
    "semantic": float(os.getenv("RAG_WEIGHT_SEMANTIC", "0.85")),
    "vector": float(os.getenv("RAG_WEIGHT_VECTOR", "1.10")),
    "metadata": float(os.getenv("RAG_WEIGHT_METADATA", "1.00")),
    "reranker": float(os.getenv("RAG_WEIGHT_RERANKER", "0.40")),
}

ROLES = ["Employee", "HR", "Finance", "IT", "Auditor", "Admin"]
ALLOWED_EXTENSIONS = {".md", ".pdf", ".json", ".csv", ".sql"}
ALLOWED_SENSITIVITIES = {"public", "internal", "confidential", "restricted"}
MAX_SOURCE_BYTES = 5 * 1024 * 1024

INTENT_RULES = {
    "finance_analysis": {
        "terms": {"revenue", "budget", "finance", "q4", "cost", "payroll", "compensation"},
        "route": "structured_finance_route",
        "strategy": "sql_csv_report_hybrid",
        "stores": ("sql_store", "document_store", "vector_store"),
        "departments": ("Finance",),
        "sources": ("structured_report", "csv", "sql_dump"),
        "expansion": ("revenue", "budget", "cost", "forecast", "payroll"),
    },
    "incident_analysis": {
        "terms": {"incident", "api", "latency", "outage", "failed", "payment", "root", "cause", "password", "reset", "service", "desk", "vpn", "laptop"},
        "route": "it_incident_route",
        "strategy": "logs_reports_hybrid",
        "stores": ("log_store", "document_store", "sql_store", "vector_store"),
        "departments": ("IT",),
        "sources": ("technical_report", "json_log", "csv", "sql_dump"),
        "expansion": ("incident", "latency", "failure", "root cause", "remediation", "log"),
    },
    "compliance_lookup": {
        "terms": {"pii", "gdpr", "compliance", "audit", "control", "sensitive", "approval"},
        "route": "compliance_audit_route",
        "strategy": "policy_log_sql_hybrid",
        "stores": ("document_store", "log_store", "sql_store", "vector_store"),
        "departments": ("Compliance",),
        "sources": ("compliance_record", "json_log", "csv", "sql_dump"),
        "expansion": ("pii", "gdpr", "audit", "control", "approval", "evidence"),
    },
    "hr_policy": {
        "terms": {"leave", "remote", "remotely", "hybrid", "work", "policy", "employee", "employees", "sick", "manager"},
        "route": "hr_policy_route",
        "strategy": "document_policy_search",
        "stores": ("document_store", "vector_store"),
        "departments": ("HR",),
        "sources": ("pdf_document", "json_log"),
        "expansion": ("leave", "remote work", "employee", "manager approval", "policy"),
    },
}
