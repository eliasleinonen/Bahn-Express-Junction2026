"""Versioned, append-only decision records for offline dispatcher research.

The writer deliberately accepts only decision-time snapshots supplied by the
caller.  Later outcomes are separate label records, so model training cannot
accidentally see information that was unavailable when an action was chosen.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping


SCHEMA_VERSION = 1
DecisionKind = Literal["quote", "post", "reprice", "withdraw"]
_DECISION_KINDS = frozenset({"quote", "post", "reprice", "withdraw"})


def _json_copy(value: Any) -> Any:
    """Return a JSON-only deep copy, rejecting non-reproducible values early."""
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _record_id(body: Mapping[str, Any]) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def make_decision_record(*, kind: DecisionKind, world_id: str, at: int,
                         snapshot: Mapping[str, Any], action: Mapping[str, Any],
                         run: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create one immutable decision-time record.

    ``snapshot`` must contain only facts known immediately before ``action`` was
    chosen.  The module cannot verify time availability itself, so callers are
    required to build the snapshot before sending an action to the service.
    """
    if kind not in _DECISION_KINDS:
        raise ValueError(f"unsupported decision kind: {kind!r}")
    if not world_id or not isinstance(at, int):
        raise ValueError("world_id and integer at are required")
    body = {
        "schemaVersion": SCHEMA_VERSION,
        "recordType": "decision",
        "kind": kind,
        "worldId": world_id,
        "at": at,
        "snapshot": _json_copy(snapshot),
        "action": _json_copy(action),
        "run": _json_copy(run or {}),
    }
    return {"recordId": _record_id(body), **body}


def make_label_record(*, decision_id: str, world_id: str, at: int,
                      labels: Mapping[str, Any]) -> dict[str, Any]:
    """Create a later outcome record without mutating the original decision."""
    if not decision_id or not world_id or not isinstance(at, int):
        raise ValueError("decision_id, world_id and integer at are required")
    body = {
        "schemaVersion": SCHEMA_VERSION,
        "recordType": "label",
        "decisionId": decision_id,
        "worldId": world_id,
        "at": at,
        "labels": _json_copy(labels),
    }
    return {"recordId": _record_id(body), **body}


def validate_record(record: Mapping[str, Any]) -> None:
    """Validate schema shape and deterministic identity before it reaches disk."""
    record_type = record.get("recordType")
    if record.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    if record_type == "decision":
        required = {"recordId", "schemaVersion", "recordType", "kind", "worldId", "at", "snapshot", "action", "run"}
        if not required.issubset(record) or record["kind"] not in _DECISION_KINDS:
            raise ValueError("invalid decision record")
        body = {key: record[key] for key in required if key != "recordId"}
    elif record_type == "label":
        required = {"recordId", "schemaVersion", "recordType", "decisionId", "worldId", "at", "labels"}
        if not required.issubset(record):
            raise ValueError("invalid label record")
        body = {key: record[key] for key in required if key != "recordId"}
    else:
        raise ValueError("unknown record type")
    if record["recordId"] != _record_id(body):
        raise ValueError("record id does not match record body")
    _json_copy(record)


def append_record(path: str | Path, record: Mapping[str, Any]) -> None:
    """Atomically append one validated JSONL record without rewriting history."""
    validate_record(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_copy(record), sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line.encode())
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate an append-only JSONL dataset in recorded order."""
    target = Path(path)
    if not target.exists():
        return []
    records = [json.loads(line) for line in target.read_text().splitlines() if line]
    for record in records:
        validate_record(record)
    return records
