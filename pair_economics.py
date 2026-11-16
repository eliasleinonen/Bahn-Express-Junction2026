"""Pure, conservative economics for deciding whether two cars share one gig.

This module owns no dispatcher state.  It evaluates an ordered pair against
real stop windows, compares its fee/risk to two individual gigs, and returns a
fully explainable assessment.  V5 can adopt it only after paired-seed tests.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CarPlan:
    """Decision-time facts for a single-car route."""

    order_id: str
    pickup_from: int
    last_pickup_at: int
    delivery_from: int
    deadline: int
    drive_minutes: int
    individual_fee_cents: int
    individual_late_probability: float
    refund_cents: int


@dataclass(frozen=True)
class PairLimits:
    max_drive_minutes: int = 5 * 60
    max_bridge_minutes: int = 60
    max_idle_minutes: int = 60
    min_net_cents: int = 500
    bridge_cost_cents_per_km: int = 80


@dataclass(frozen=True)
class PairAssessment:
    feasible: bool
    reason: str | None
    finish_at: int | None
    drive_minutes: int
    bridge_minutes: int
    idle_minutes: int
    bridge_cost_cents: int
    added_late_loss_cents: int
    net_saving_cents: int


def _append_car(at: int, car: CarPlan) -> tuple[int, int] | None:
    """Return finish time and waiting caused by this car's real stop windows."""
    pickup_idle = max(0, car.pickup_from - at)
    at = max(at, car.pickup_from)
    if at > car.last_pickup_at:
        return None
    arrival = at + car.drive_minutes
    idle = max(0, car.delivery_from - arrival)
    finish = arrival + idle
    if finish > car.deadline:
        return None
    return finish, pickup_idle + idle


def evaluate_pair(*, now: int, first: CarPlan, second: CarPlan,
                  bridge_minutes: int, bridge_km: float, pair_fee_cents: int,
                  paired_late_probabilities: tuple[float, float],
                  limits: PairLimits = PairLimits()) -> PairAssessment:
    """Assess one ordered pair against singles with no hidden heuristics.

    `paired_late_probabilities` must align with `first`, `second`; it is
    supplied by the fleet model, keeping route feasibility independent from
    probability fitting.
    """
    if bridge_minutes > limits.max_bridge_minutes:
        return PairAssessment(False, "bridge_minutes", None, 0, bridge_minutes, 0, 0, 0, 0)
    # Waiting before the first pickup is not pair-induced; idle only measures
    # time introduced between/within the shared route.
    first_result = _append_car(max(now, first.pickup_from), first)
    if first_result is None:
        return PairAssessment(False, "first_window", None, first.drive_minutes, bridge_minutes, 0, 0, 0, 0)
    first_finish, first_idle = first_result
    bridge_arrival = first_finish + bridge_minutes
    second_result = _append_car(bridge_arrival, second)
    if second_result is None:
        return PairAssessment(False, "second_window", None, first.drive_minutes + bridge_minutes + second.drive_minutes,
                              bridge_minutes, first_idle, 0, 0, 0)
    finish, second_idle = second_result
    drive = first.drive_minutes + bridge_minutes + second.drive_minutes
    idle = first_idle + second_idle
    if drive > limits.max_drive_minutes:
        return PairAssessment(False, "drive_minutes", finish, drive, bridge_minutes, idle, 0, 0, 0)
    if idle > limits.max_idle_minutes:
        return PairAssessment(False, "idle_minutes", finish, drive, bridge_minutes, idle, 0, 0, 0)
    bridge_cost = max(0, int(bridge_km * limits.bridge_cost_cents_per_km + 0.5))
    individual_late_loss = (
        first.individual_late_probability * first.refund_cents
        + second.individual_late_probability * second.refund_cents
    )
    paired_late_loss = (
        paired_late_probabilities[0] * first.refund_cents
        + paired_late_probabilities[1] * second.refund_cents
    )
    added_late_loss = int(paired_late_loss - individual_late_loss + 0.5)
    net = first.individual_fee_cents + second.individual_fee_cents - pair_fee_cents - bridge_cost - added_late_loss
    if net < limits.min_net_cents:
        return PairAssessment(False, "negative_net", finish, drive, bridge_minutes, idle,
                              bridge_cost, added_late_loss, net)
    return PairAssessment(True, None, finish, drive, bridge_minutes, idle,
                          bridge_cost, added_late_loss, net)
