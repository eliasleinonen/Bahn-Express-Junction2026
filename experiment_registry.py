"""Immutable metadata for reproducible dispatcher experiments.

The registry deliberately separates worlds by deterministic role before a
campaign begins.  It freezes the policy hash and all dispatcher environment
settings which can affect a decision; later source or shell changes cannot be
mistaken for the measured candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from offline_data_contract import ROLE_NAMES, SPLIT_VERSION, world_role


SCHEMA_VERSION = 1


def canonical_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Keep only policy knobs, never credentials or unrelated shell state."""
    source = os.environ if environment is None else environment
    return {key: str(source[key]) for key in sorted(source) if key.startswith("V5_") or key.startswith("V52_")}


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def freeze_experiment(directory: Path, *, name: str, policy: Path,
                      worlds: Iterable[str], environment: Mapping[str, str] | None = None,
                      extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create an immutable manifest; refuse to silently mix a different policy."""
    policy = policy.resolve()
    if not policy.is_file():
        raise FileNotFoundError(policy)
    manifest_path = directory / "manifest.json"
    content = policy.read_bytes()
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "splitVersion": SPLIT_VERSION,
        "name": name,
        "policyFile": policy.name,
        "policySha256": hashlib.sha256(content).hexdigest(),
        "environment": canonical_environment(environment),
        "worlds": [{"key": key, "role": world_role(key)} for key in sorted(set(worlds))],
        "extra": dict(extra or {}),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Experiment manifest already exists with different policy/configuration.")
        return existing
    atomic_json(manifest_path, manifest)
    return manifest


def record_metric(directory: Path, *, world_key: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Append one result with its preassigned role; metrics stay structured JSON."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    roles = {row["key"]: row["role"] for row in manifest["worlds"]}
    if world_key not in roles:
        raise ValueError(f"World {world_key!r} is not registered in this experiment.")
    row = {"world": world_key, "role": roles[world_key], "metrics": dict(metrics)}
    path = directory / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("freeze", help="freeze policy/configuration and split worlds")
    init.add_argument("--directory", type=Path, required=True)
    init.add_argument("--name", required=True)
    init.add_argument("--policy", type=Path, required=True)
    init.add_argument("--world", action="append", required=True)
    metric = sub.add_parser("record", help="append a structured result")
    metric.add_argument("--directory", type=Path, required=True)
    metric.add_argument("--world", required=True)
    metric.add_argument("--metrics", required=True, help="JSON object")
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze_experiment(args.directory, name=args.name, policy=args.policy, worlds=args.world), indent=2))
    else:
        value = json.loads(args.metrics)
        if not isinstance(value, dict):
            parser.error("--metrics must be a JSON object")
        print(json.dumps(record_metric(args.directory, world_key=args.world, metrics=value)))


if __name__ == "__main__":
    main()
