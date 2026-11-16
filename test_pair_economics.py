"""Checks for restrictive, explainable two-car economics."""
from pair_economics import CarPlan, PairLimits, evaluate_pair


def car(order_id: str, *, pickup: int = 100, delivery: int = 180, deadline: int = 500,
        drive: int = 60, fee: int = 10_000, late: float = .03) -> CarPlan:
    return CarPlan(order_id, pickup, pickup + 180, delivery, deadline, drive, fee, late, 8_000)


def main() -> None:
    first = car("one")
    second = car("two", pickup=220, delivery=280, deadline=700)
    positive = evaluate_pair(now=0, first=first, second=second, bridge_minutes=20, bridge_km=8,
                             pair_fee_cents=16_000, paired_late_probabilities=(.04, .04))
    assert positive.feasible and positive.net_saving_cents >= 500 and positive.drive_minutes == 140

    bridge = evaluate_pair(now=0, first=first, second=second, bridge_minutes=61, bridge_km=8,
                           pair_fee_cents=16_000, paired_late_probabilities=(.04, .04))
    assert not bridge.feasible and bridge.reason == "bridge_minutes"

    long_second = car("long", pickup=220, delivery=600, deadline=900, drive=230)
    driving = evaluate_pair(now=0, first=first, second=long_second, bridge_minutes=20, bridge_km=8,
                             pair_fee_cents=16_000, paired_late_probabilities=(.04, .04))
    assert not driving.feasible and driving.reason == "drive_minutes"

    idle_second = car("idle", pickup=400, delivery=500, deadline=900)
    idle = evaluate_pair(now=0, first=first, second=idle_second, bridge_minutes=20, bridge_km=8,
                         pair_fee_cents=16_000, paired_late_probabilities=(.04, .04))
    assert not idle.feasible and idle.reason == "idle_minutes"

    poor = evaluate_pair(now=0, first=first, second=second, bridge_minutes=20, bridge_km=30,
                         pair_fee_cents=19_900, paired_late_probabilities=(.20, .20))
    assert not poor.feasible and poor.reason == "negative_net"

    print("pair economics checks passed")


if __name__ == "__main__":
    main()
