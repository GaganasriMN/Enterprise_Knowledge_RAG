import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from rag_assistant.config import DATA_DIR
from rag_assistant.models import Chunk, IngestionReport
from rag_assistant.ingestion_security import (
    update_manifest,
    validate_metadata,
    validate_source_path,
    validate_sql_dump,
)


def chunk_sections(body: str, fallback_title: str) -> list[tuple[str, str]]:
    sections = re.split(r"\n(?=## )", body.strip())
    parsed = []
    for section_text in sections:
        heading_match = re.search(r"^##\s+(.+)$", section_text, flags=re.MULTILINE)
        section = heading_match.group(1).strip() if heading_match else fallback_title
        parsed.append((section, section_text.replace("## ", "").strip()))
    return parsed


def load_markdown_documents() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        if path.name == "hr_policy.md":
            continue
        validate_source_path(path)
        raw = path.read_text(encoding="utf-8")
        metadata_block, body = raw.split("---", 2)[1:]
        metadata = {}
        for line in metadata_block.strip().splitlines():
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        validate_metadata(metadata, path.name)

        allowed_roles = tuple(role.strip() for role in metadata["allowed_roles"].split(","))
        for index, (section, section_text) in enumerate(chunk_sections(body, metadata["title"]), start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}-{index}",
                    source=path.name,
                    source_type=metadata["source_type"],
                    title=metadata["title"],
                    section=section,
                    department=metadata["department"],
                    sensitivity=metadata["sensitivity"],
                    allowed_roles=allowed_roles,
                    text=section_text.replace("## ", "").strip(),
                )
            )
    return chunks


def load_pdf_documents() -> list[Chunk]:
    chunks = []
    for path in sorted(DATA_DIR.glob("*.pdf")):
        validate_source_path(path)
        metadata_path = DATA_DIR / f"{path.stem.replace('_handbook', '')}_metadata.json"
        if not metadata_path.exists():
            metadata_path = DATA_DIR / f"{path.stem}_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_metadata(metadata, metadata_path.name)

        reader = PdfReader(str(path))
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            title_match = re.search(r"Page\s+\d+:\s+(.+)", text)
            section = title_match.group(1).strip() if title_match else f"Page {page_index}"
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}-page-{page_index}",
                    source=path.name,
                    source_type=metadata["source_type"],
                    title=metadata["title"],
                    section=section,
                    department=metadata["department"],
                    sensitivity=metadata["sensitivity"],
                    allowed_roles=tuple(metadata["allowed_roles"]),
                    text=f"Page {page_index}. {text.strip()}",
                )
            )
    return chunks


def load_policy_json() -> list[Chunk]:
    chunks = []
    for path in sorted(DATA_DIR.glob("*_metadata.json")):
        validate_source_path(path)
        metadata = json.loads(path.read_text(encoding="utf-8"))
        validate_metadata(metadata, path.name)
        text = (
            f"Metadata record for {metadata['title']}. Owner: {metadata.get('owner', 'unknown')}. "
            f"Retention: {metadata.get('retention', 'unknown')}. "
            f"Policy domains: {', '.join(metadata.get('policy_domains', []))}."
        )
        chunks.append(
            Chunk(
                chunk_id=f"{path.stem}-metadata",
                source=path.name,
                source_type="json_metadata",
                title=metadata["title"],
                section="metadata",
                department=metadata["department"],
                sensitivity=metadata["sensitivity"],
                allowed_roles=tuple(metadata["allowed_roles"]),
                text=text,
            )
        )
    return chunks


def load_json_logs() -> list[Chunk]:
    path = DATA_DIR / "audit_logs.json"
    validate_source_path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for record in records:
        text = (
            f"Log {record['log_id']} from {record['timestamp']}. "
            f"System: {record['system']}. Severity: {record['severity']}. "
            f"Event: {record['event']}. Details: {record['details']}. "
            f"Owner: {record['owner_department']}."
        )
        chunks.append(
            Chunk(
                chunk_id=record["log_id"],
                source=path.name,
                source_type="json_log",
                title="Security and System Audit Logs",
                section=record["system"],
                department=record["owner_department"],
                sensitivity=record["sensitivity"],
                allowed_roles=tuple(record["allowed_roles"]),
                text=text,
            )
        )
    return chunks


def load_csv_projects() -> list[Chunk]:
    path = DATA_DIR / "employee_projects.csv"
    validate_source_path(path)
    chunks = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            text = (
                f"Project {row['project_id']} is {row['project_name']}, owned by "
                f"{row['department']}. Status: {row['status']}. Budget: {row['budget']}. "
                f"Business note: {row['business_note']}."
            )
            chunks.append(
                Chunk(
                    chunk_id=row["project_id"],
                    source=path.name,
                    source_type="csv",
                    title="Employee Project Portfolio",
                    section=row["project_name"],
                    department=row["department"],
                    sensitivity=row["sensitivity"],
                    allowed_roles=tuple(role.strip() for role in row["allowed_roles"].split("|")),
                    text=text,
                )
            )
    return chunks


def load_sql_records() -> list[Chunk]:
    path = DATA_DIR / "enterprise_records.sql"
    validate_source_path(path)
    sql_text = path.read_text(encoding="utf-8")
    validate_sql_dump(sql_text, path.name)
    connection = sqlite3.connect(":memory:")
    connection.executescript(sql_text)
    rows = connection.execute(
        """
        SELECT record_id, title, department, category, sensitivity, allowed_roles, summary
        FROM enterprise_records
        """
    ).fetchall()
    connection.close()

    chunks = []
    for record_id, title, department, category, sensitivity, allowed_roles, summary in rows:
        chunks.append(
            Chunk(
                chunk_id=record_id,
                source=path.name,
                source_type="sql_dump",
                title=title,
                section=category,
                department=department,
                sensitivity=sensitivity,
                allowed_roles=tuple(role.strip() for role in allowed_roles.split("|")),
                text=f"{title}. Category: {category}. {summary}",
            )
        )
    return chunks


def source_paths() -> list[Path]:
    return sorted(path for path in DATA_DIR.iterdir() if path.is_file() and path.suffix.lower() in {".md", ".pdf", ".json", ".csv", ".sql"})


def build_ingestion_report(chunks: list[Chunk], changed_sources: tuple[str, ...], skipped_sources: tuple[str, ...]) -> IngestionReport:
    return IngestionReport(
        source_count=len({chunk.source for chunk in chunks}),
        chunk_count=len(chunks),
        source_types=dict(Counter(chunk.source_type for chunk in chunks)),
        departments=dict(Counter(chunk.department for chunk in chunks)),
        sensitivities=dict(Counter(chunk.sensitivity for chunk in chunks)),
        changed_sources=changed_sources,
        skipped_sources=skipped_sources,
    )


def load_corpus_with_report() -> tuple[list[Chunk], IngestionReport]:
    paths = source_paths()
    _, changed_sources = update_manifest(paths)
    skipped_sources = ("hr_policy.md replaced by hr_policy_handbook.pdf",)
    chunks = (
        load_markdown_documents()
        + load_pdf_documents()
        + load_policy_json()
        + load_json_logs()
        + load_csv_projects()
        + load_sql_records()
    )
    return chunks, build_ingestion_report(chunks, changed_sources, skipped_sources)


def load_corpus() -> list[Chunk]:
    chunks, _ = load_corpus_with_report()
    return chunks
