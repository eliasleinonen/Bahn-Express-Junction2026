"""Focused checks for the append-only decision dataset."""
from pathlib import Path
from tempfile import TemporaryDirectory

import decision_dataset as dataset


def main() -> None:
    decision = dataset.make_decision_record(
        kind="quote", world_id="world-a", at=120,
        snapshot={"risk": 0.04, "options": [{"kind": "next_day"}]},
        action={"prices": {"next_day": 24_000}},
        run={"policy": "v5.2", "config": {"lateTarget": 0.08}},
    )
    assert decision["recordId"] == dataset.make_decision_record(
        kind="quote", world_id="world-a", at=120,
        snapshot={"risk": 0.04, "options": [{"kind": "next_day"}]},
        action={"prices": {"next_day": 24_000}},
        run={"policy": "v5.2", "config": {"lateTarget": 0.08}},
    )["recordId"]
    label = dataset.make_label_record(
        decision_id=decision["recordId"], world_id="world-a", at=300,
        labels={"bookedBeforeWarning": True, "deliveredLate": False},
    )
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "decisions.jsonl"
        dataset.append_record(path, decision)
        dataset.append_record(path, label)
        assert dataset.read_records(path) == [decision, label]
    broken = {**decision, "at": 121}
    try:
        dataset.validate_record(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered records must be rejected")
    print("Decision dataset checks passed")


if __name__ == "__main__":
    main()
