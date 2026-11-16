"""Checks for immutable experiment manifests and held-out world assignment."""
from pathlib import Path
from tempfile import TemporaryDirectory

import experiment_registry as registry


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        policy = root / "policy.py"
        policy.write_text("print('policy')\n", encoding="utf-8")
        directory = root / "experiment"
        manifest = registry.freeze_experiment(
            directory, name="candidate", policy=policy,
            worlds=["alpha", "beta", "gamma"], environment={"V5_LATE_TARGET": "0.08", "SECRET": "no"},
        )
        assert manifest["environment"] == {"V5_LATE_TARGET": "0.08"}
        assert {item["role"] for item in manifest["worlds"]} <= set(registry.ROLE_NAMES)
        assert registry.world_role("alpha") == registry.world_role("alpha")
        assert registry.freeze_experiment(directory, name="candidate", policy=policy,
                                          worlds=["gamma", "beta", "alpha"],
                                          environment={"V5_LATE_TARGET": "0.08"}) == manifest
        row = registry.record_metric(directory, world_key="beta", metrics={"marginCents": 1234})
        assert row["role"] == registry.world_role("beta")
        try:
            registry.record_metric(directory, world_key="unknown", metrics={})
            raise AssertionError("unregistered world must fail")
        except ValueError:
            pass
    print("experiment registry checks passed")


if __name__ == "__main__":
    main()
