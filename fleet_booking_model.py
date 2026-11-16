"""Fit an auditable booked-before-warning prior from saved practice worlds.

The input is only the normalized local analysis database.  It deliberately
does not infer a fee elasticity: historic fee observations are selected by the
old policies and are not a randomized supply experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from offline_data_contract import HOUR, features_from_warning_slack, world_role


SCHEMA_VERSION = 1
def feature_row(row: sqlite3.Row) -> dict[str, str]:
    return runtime_features(
        warning_slack=row["warning_slack"], pickup_private=bool(row["pickup_private"]),
        delivery_private=bool(row["delivery_private"]), premium=bool(row["premium"]),
        electric=bool(row["electric"]), legs=int(row["leg_count"]), arrived_at=row["arrived_at"],
    )


def runtime_features(*, warning_slack: int | None, pickup_private: bool, delivery_private: bool,
                     premium: bool, electric: bool, legs: int, arrived_at: int | None) -> dict[str, str]:
    """Compatibility wrapper; ``arrived_at`` must be the actual decision time."""
    return features_from_warning_slack(
        warning_slack=warning_slack, pickup_private=pickup_private,
        delivery_private=delivery_private, premium=premium, electric=electric,
        legs=legs, decision_at=arrived_at or 0,
    )


def rows(db: sqlite3.Connection) -> Iterable[tuple[str, int, dict[str, str]]]:
    """One order row, label 1 iff it never reached its first warning.

    A warning is emitted only while no driver holds the order, making this a
    conservative observable proxy for booked-before-warning.  Cancellations
    are excluded because neither outcome is meaningful fleet evidence.
    """
    db.row_factory = sqlite3.Row
    sql = """
    SELECT o.world_id, o.order_id, r.arrived_at, r.leg_count,
           o.last_pickup_at - r.arrived_at - 180 AS warning_slack,
           l.pickup_private, l.delivery_private, l.premium, l.electric,
           CASE WHEN EXISTS (
             SELECT 1 FROM events e WHERE e.world_id=o.world_id
             AND e.order_id=o.order_id AND e.kind='deadline_close'
           ) THEN 0 ELSE 1 END AS label
    FROM orders o JOIN requests r ON r.world_id=o.world_id AND r.request_id=o.order_id
    JOIN request_legs l ON l.world_id=o.world_id AND l.request_id=o.order_id AND l.leg=0
    WHERE o.cancelled=0 AND o.last_pickup_at IS NOT NULL AND r.arrived_at IS NOT NULL
    """
    for row in db.execute(sql):
        yield row["world_id"], int(row["label"]), feature_row(row)


def smoothed(success: int, total: int, alpha: float = 8.0, beta: float = 8.0) -> float:
    return (success + alpha) / (total + alpha + beta)


def fit(samples: Iterable[tuple[str, int, dict[str, str]]]) -> dict[str, Any]:
    total = success = 0
    grouped: dict[str, Counter[tuple[str, int]]] = defaultdict(Counter)
    worlds: Counter[str] = Counter()
    for world_id, label, features in samples:
        total += 1
        success += label
        worlds[world_role(world_id)] += 1
        for name, value in features.items():
            grouped[name][(value, label)] += 1
    global_p = smoothed(success, total)
    tables: dict[str, dict[str, dict[str, float]]] = {}
    for name, counts in grouped.items():
        table = {}
        for value in sorted({key[0] for key in counts}):
            n, yes = counts[(value, 0)] + counts[(value, 1)], counts[(value, 1)]
            # Empirical-Bayes shrinkage keeps rare bins from pretending to be facts.
            weight = n / (n + 50.0)
            table[value] = {"n": n, "p": weight * smoothed(yes, n) + (1 - weight) * global_p}
        tables[name] = table
    return {"schemaVersion": SCHEMA_VERSION, "label": "no_deadline_close_before_booking_proxy",
            "global": {"n": total, "p": global_p}, "worldRows": dict(worlds), "tables": tables}


def predict(model: dict[str, Any], features: dict[str, str]) -> float:
    """Average bounded log-odds adjustments from broad, reusable feature bins."""
    base = model["global"]["p"]
    logit = math.log(base / (1 - base))
    contributions = 0
    for name, value in features.items():
        cell = model["tables"].get(name, {}).get(value)
        if cell is None:
            continue
        p = min(.97, max(.03, cell["p"]))
        logit += math.log(p / (1 - p)) - math.log(base / (1 - base))
        contributions += 1
    if contributions:
        logit = math.log(base / (1 - base)) + (logit - math.log(base / (1 - base))) / contributions
    return 1 / (1 + math.exp(-logit))


def evaluate(model: dict[str, Any], samples: Iterable[tuple[str, int, dict[str, str]]], split: str) -> dict[str, float | int]:
    labels: list[tuple[int, float]] = [(label, predict(model, features)) for world, label, features in samples if world_role(world) == split]
    if not labels:
        return {"n": 0}
    return {"n": len(labels), "actual": sum(x for x, _ in labels) / len(labels),
            "predicted": sum(p for _, p in labels) / len(labels),
            "brier": sum((x - p) ** 2 for x, p in labels) / len(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/analysis.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("data/models/fleet_booking_v1.json"))
    parser.add_argument("--include-heldout", action="store_true",
                        help="evaluate held-out worlds explicitly after validation review")
    args = parser.parse_args()
    with sqlite3.connect(args.db) as db:
        samples = list(rows(db))
    train = [sample for sample in samples if world_role(sample[0]) == "train"]
    model = fit(train)
    model["validation"] = evaluate(model, samples, "validation")
    if args.include_heldout:
        model["heldout"] = evaluate(model, samples, "heldout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    report = {"output": str(args.output), "train": model["global"]["n"], "validation": model["validation"]}
    if args.include_heldout:
        report["heldout"] = model["heldout"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
