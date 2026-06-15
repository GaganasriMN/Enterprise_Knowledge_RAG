import hashlib
import json
import re
from pathlib import Path

from rag_assistant.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_SENSITIVITIES,
    ARTIFACTS_DIR,
    DATA_DIR,
    MAX_SOURCE_BYTES,
    ROLES,
)


MANIFEST_PATH = ARTIFACTS_DIR / "ingestion_manifest.json"


def validate_source_path(path: Path) -> None:
    resolved = path.resolve()
    data_root = DATA_DIR.resolve()
    if data_root not in resolved.parents and resolved != data_root:
        raise ValueError(f"Source path escapes data directory: {path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported source extension: {path.suffix}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError(f"Source exceeds max size: {path.name}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def update_manifest(paths: list[Path]) -> tuple[dict, tuple[str, ...]]:
    previous = load_manifest()
    current = {}
    changed = []
    for path in paths:
        validate_source_path(path)
        digest = file_sha256(path)
        current[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
        if previous.get(path.name, {}).get("sha256") != digest:
            changed.append(path.name)
    save_manifest(current)
    return current, tuple(changed)


def validate_metadata(metadata: dict, source_name: str) -> None:
    required = {"title", "source_type", "department", "sensitivity", "allowed_roles"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"{source_name} missing metadata: {', '.join(sorted(missing))}")

    if metadata["sensitivity"] not in ALLOWED_SENSITIVITIES:
        raise ValueError(f"{source_name} has unsupported sensitivity: {metadata['sensitivity']}")

    roles = metadata["allowed_roles"]
    if isinstance(roles, str):
        roles = [role.strip() for role in roles.replace("|", ",").split(",")]
    unknown_roles = set(roles).difference(ROLES)
    if unknown_roles:
        raise ValueError(f"{source_name} has unsupported roles: {', '.join(sorted(unknown_roles))}")


def validate_sql_dump(sql_text: str, source_name: str) -> None:
    lowered = sql_text.lower()
    forbidden = ["attach ", "detach ", "pragma ", "drop ", "delete ", "update ", "alter ", "vacuum"]
    if any(token in lowered for token in forbidden):
        raise ValueError(f"{source_name} contains forbidden SQL statement")
    allowed_statement = re.compile(r"^\s*(create table|insert into|--|/\*|\*/|\s*$)", re.IGNORECASE)
    for statement in sql_text.split(";"):
        if statement.strip() and not allowed_statement.match(statement):
            raise ValueError(f"{source_name} contains non-ingestion SQL: {statement[:40]}")
