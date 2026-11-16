"""Canonical split and post-time feature contract for offline models.

Any model trained on a decision-time feature must use this module at runtime.
It intentionally does not provide an arrival-time substitute for post time:
legacy records without a post decision timestamp are not equivalent evidence.
"""
from __future__ import annotations

import hashlib


HOUR = 60
SPLIT_VERSION = 1
ROLE_NAMES = ("train", "validation", "heldout")


def world_role(world_key: str) -> str:
    """Stable 80/10/10 assignment shared by every model and experiment."""
    bucket = int(hashlib.sha256(world_key.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "heldout"


def warning_slack_bin(minutes: int | None) -> str:
    if minutes is None or minutes <= 3 * HOUR:
        return "le_3h"
    if minutes <= 6 * HOUR:
        return "le_6h"
    if minutes <= 12 * HOUR:
        return "le_12h"
    return "gt_12h"


def post_time_features(*, last_pickup_at: int, decision_at: int,
                       pickup_private: bool, delivery_private: bool,
                       premium: bool, electric: bool, legs: int) -> dict[str, str]:
    """Broad reusable features known exactly when a gig is posted."""
    return features_from_warning_slack(
        warning_slack=last_pickup_at - decision_at - 3 * HOUR,
        pickup_private=pickup_private, delivery_private=delivery_private,
        premium=premium, electric=electric, legs=legs, decision_at=decision_at,
    )


def features_from_warning_slack(*, warning_slack: int | None,
                                pickup_private: bool, delivery_private: bool,
                                premium: bool, electric: bool, legs: int,
                                decision_at: int) -> dict[str, str]:
    """Feature form for logs which already retain time-to-warning."""
    special = pickup_private or delivery_private or premium
    return {
        "warning_slack": warning_slack_bin(warning_slack),
        "access": "special" if special else "electric" if electric else "standard",
        "legs": "multi" if legs > 1 else "single",
        "daypart": str(decision_at // (6 * HOUR) % 4),
    }
