"""Integration checks for V5's optional, decision-neutral shadow labels."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import decision_dataset
import dispatch_v5


def make_desk(path: Path | None) -> dispatch_v5.PortfolioDesk:
    if path is None:
        os.environ.pop("V5_DECISION_LOG", None)
    else:
        os.environ["V5_DECISION_LOG"] = str(path)
    world = SimpleNamespace(id="shadow-world", client=SimpleNamespace(map=lambda: {"roads": []}))
    desk = dispatch_v5.PortfolioDesk(world, 8, 480, 1.65, 0.80, 1000, "all")
    order = {"id": "order", "option": "next_day", "deadline": 2_000, "lastPickupAt": 1_000,
             "legs": [{"pickup": {"placeId": "A", "privateAddress": False, "window": {"from": 600, "to": 1_000}},
                       "delivery": {"placeId": "B", "privateAddress": False, "window": {"from": 700, "to": 2_000}},
                       "car": {"premium": False, "electric": False, "chargePercent": None}}],
             "deliveredAt": [None]}
    desk.orders["order"] = order
    desk.requests["order"] = {"id": "order", "dealerId": "dealer", "arrivedAt": 480,
                              "referenceFeeCents": 10_000, "options": [{"kind": "next_day", "deadline": 2_000, "listPriceCents": 20_000}],
                              "legs": order["legs"], "note": None}
    desk.leg_gigs["order"] = {0: "gig"}
    return desk


def install_post_record(desk: dispatch_v5.PortfolioDesk, path: Path) -> str:
    record = decision_dataset.make_decision_record(kind="post", world_id="shadow-world", at=480,
        snapshot={"now": 480}, action={"type": "post", "stops": []})
    decision_dataset.append_record(path, record)
    desk.gig_decisions["gig"] = record["recordId"]
    desk.gig_fee_state["gig"] = {"orderIds": ("order",), "contexts": (), "openingFee": 10_000,
                                  "postedAt": 480, "repriced": False}
    return record["recordId"]


def labels(path: Path) -> list[dict]:
    return [r["labels"] for r in decision_dataset.read_records(path) if r["recordType"] == "label"]


def main() -> None:
    old = os.environ.get("V5_DECISION_LOG")
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "shadow.jsonl"
        early = make_desk(path)
        install_post_record(early, path)
        early.handle({"kind": "gig_booked", "gigId": "gig", "at": 819})
        assert {"booked": True, "bookedBeforeWarning": True} in labels(path)

        late_path = Path(temporary) / "late.jsonl"
        late = make_desk(late_path)
        install_post_record(late, late_path)
        late.handle({"kind": "gig_booked", "gigId": "gig", "at": 820})
        assert {"booked": True, "bookedBeforeWarning": False} in labels(late_path)

        warning_path = Path(temporary) / "warning.jsonl"
        warning = make_desk(warning_path)
        install_post_record(warning, warning_path)
        warning.handle({"kind": "deadline_close", "orderId": "order", "at": 820})
        warning.handle({"kind": "deadline_close", "orderId": "order", "at": 821})
        warning_labels = [label for label in labels(warning_path) if label.get("reachedWarning")]
        assert warning_labels == [{"reachedWarning": True, "bookedBeforeWarning": False, "warningAt": 820}]

        plain = make_desk(None)
        before = plain.fee_scale
        plain.handle({"kind": "gig_booked", "gigId": "missing", "at": 819})
        assert plain.fee_scale == before - 0.004
    if old is None:
        os.environ.pop("V5_DECISION_LOG", None)
    else:
        os.environ["V5_DECISION_LOG"] = old
    print("shadow logging checks passed")


if __name__ == "__main__":
    main()
