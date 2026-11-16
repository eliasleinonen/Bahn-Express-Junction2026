"""Train a transparent option-vs-outside demand prior from saved worlds.

Each request is a choice among our offered delivery options and an outside
carrier.  The model is deliberately small: regularised shared feature weights
instead of a separate live-world price table.  It is predictive evidence, not
a causal claim about a new price; controlled shadow probes are still required
before it prices production work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DAY, HOUR = 24 * 60, 60


def world_split(world_id: str) -> str:
    bucket = int(hashlib.sha256(world_id.encode()).hexdigest()[:8], 16) % 10
    return "heldout" if bucket == 0 else "validation" if bucket == 1 else "train"


def price_bin(price: int, listed: int) -> str:
    ratio = price / max(1, listed)
    for label, bound in (("le_0.9", .9), ("le_1.1", 1.1), ("le_1.3", 1.3),
                         ("le_1.5", 1.5), ("le_1.7", 1.7), ("gt_1.7", float("inf"))):
        if ratio <= bound:
            return label
    raise AssertionError("unreachable")


def option_features(*, option: dict[str, Any], price: int, latest_deadline: int,
                    dealer: str, arrived_at: int) -> list[str]:
    speed_days = max(0, latest_deadline - int(option["deadline"])) // DAY
    speed = "fast" if speed_days >= 2 else "medium" if speed_days else "slow"
    listed = int(option["listPriceCents"])
    return [
        "bias", f"price:{price_bin(price, listed)}",
        f"speed:{speed}", f"kind:{option['kind']}", f"daypart:{(arrived_at // (6 * HOUR)) % 4}",
        # Dealer is strongly regularised below and therefore moves only after
        # repeated observations; no opaque lane/place memorisation is used.
        f"dealer:{dealer}",
    ]


def requests(db: sqlite3.Connection) -> Iterable[tuple[str, str, list[list[str]], int]]:
    """Yield world, request id, option feature lists, selected alternative index.

    Index zero is the external carrier.  Quotes without a matching option are
    ignored rather than inventing a counterfactual price.
    """
    sql = """SELECT world_id, request_id, dealer_id, arrived_at, options_json, quote_json,
                     outcome, chosen_option FROM requests WHERE quote_json IS NOT NULL"""
    for world_id, request_id, dealer, arrived, options_raw, quote_raw, outcome, chosen in db.execute(sql):
        options, quotes = json.loads(options_raw), json.loads(quote_raw)
        latest = max(int(option["deadline"]) for option in options)
        alternatives = [[]]
        kinds: list[str] = []
        for option in options:
            price = quotes.get(option["kind"])
            if price is None:
                continue
            kinds.append(option["kind"])
            alternatives.append(option_features(option=option, price=int(price), latest_deadline=latest,
                                                dealer=dealer or "unknown", arrived_at=int(arrived or 0)))
        if len(alternatives) == 1:
            continue
        target = kinds.index(chosen) + 1 if outcome == "won" and chosen in kinds else 0
        yield world_id, request_id, alternatives, target


def softmax(values: list[float]) -> list[float]:
    ceiling = max(values)
    weights = [math.exp(value - ceiling) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def train(rows: Iterable[tuple[str, str, list[list[str]], int]], *, epochs: int = 5,
          rate: float = .075, l2: float = .003) -> dict[str, Any]:
    """Deterministic regularised multinomial SGD with feature-count shrinkage."""
    rows = list(rows)
    weights: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    train_rows = [row for row in rows if world_split(row[0]) == "train"]
    for _ in range(epochs):
        for _, _, alternatives, target in train_rows:
            probabilities = softmax([sum(weights[token] for token in features) for features in alternatives])
            for index, features in enumerate(alternatives[1:], start=1):
                error = (1.0 if index == target else 0.0) - probabilities[index]
                for token in features:
                    counts[token] += 1
                    step = rate / math.sqrt(counts[token])
                    weights[token] += step * (error - l2 * weights[token])
    return {"schemaVersion": SCHEMA_VERSION, "label": "chosen_option_or_outside",
            "epochs": epochs, "rate": rate, "l2": l2, "weights": dict(sorted(weights.items())),
            "featureCounts": dict(counts), "trainRequests": len(train_rows)}


def probabilities(model: dict[str, Any], alternatives: list[list[str]]) -> list[float]:
    weights = model["weights"]
    return softmax([sum(float(weights.get(token, 0.0)) for token in features) for features in alternatives])


def evaluate(model: dict[str, Any], rows: Iterable[tuple[str, str, list[list[str]], int]], split: str) -> dict[str, float | int]:
    chosen = 0
    nll = 0.0
    predicted_win = actual_win = 0.0
    count = 0
    for world, _, alternatives, target in rows:
        if world_split(world) != split:
            continue
        probs = probabilities(model, alternatives)
        nll -= math.log(max(1e-12, probs[target]))
        predicted_win += 1.0 - probs[0]
        actual_win += target != 0
        chosen += max(range(len(probs)), key=probs.__getitem__) == target
        count += 1
    return {"n": count, "nll": nll / max(1, count), "accuracy": chosen / max(1, count),
            "actualWin": actual_win / max(1, count), "predictedWin": predicted_win / max(1, count)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/analysis.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("data/models/market_choice_v1.json"))
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    with sqlite3.connect(args.db) as db:
        rows = list(requests(db))
    model = train(rows, epochs=args.epochs)
    model["validation"] = evaluate(model, rows, "validation")
    model["heldout"] = evaluate(model, rows, "heldout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "train": model["trainRequests"],
                      "validation": model["validation"], "heldout": model["heldout"]}, indent=2))


if __name__ == "__main__":
    main()
