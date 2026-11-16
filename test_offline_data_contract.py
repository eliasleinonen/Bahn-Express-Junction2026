"""Checks for the shared split and post-time booking feature contract."""
import offline_data_contract as contract


def main() -> None:
    assert contract.world_role("same-world") == contract.world_role("same-world")
    features = contract.post_time_features(
        last_pickup_at=1_000, decision_at=700, pickup_private=False,
        delivery_private=False, premium=False, electric=False, legs=1,
    )
    assert features["warning_slack"] == "le_3h"
    copied = contract.features_from_warning_slack(
        warning_slack=1_000 - 700 - 3 * contract.HOUR,
        pickup_private=False, delivery_private=False, premium=False,
        electric=False, legs=1, decision_at=700,
    )
    assert copied == features
    print("offline data contract checks passed")


if __name__ == "__main__":
    main()
