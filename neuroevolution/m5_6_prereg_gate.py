"""M5.6 prospective cohort manifest and fail-closed confirmation gate.

Do NOT run a held-out campaign with development-only policies, a tuned
environment, or an incomplete predeclared factorial. The same manifest
must be reviewed and frozen before reading source/heldout outcomes.
"""
from __future__ import annotations

import hashlib
import json

SOURCE = tuple(range(161000, 161032))
HOLDOUT = tuple(range(162000, 162032))
REPLICATION = tuple(range(163000, 163032))
DEVELOPMENT = tuple(range(1000, 1100))
REQUIRED_ARMS = frozenset({
    "curiosity_rnn_predators", "no_curiosity_rnn_predators",
    "curiosity_memory_reset_predators", "curiosity_rnn_no_predators",
    "shuffled_curiosity_rnn_predators", "scripted_baseline_predators",
})
PRIMARY_EFFECT = "paired_world_survival_curiosity_minus_no_curiosity"
PRIMARY_MIN_GAIN = 0.10
BOOTSTRAP_REPLICATES = 20000


def manifest() -> dict:
    groups = {"development": DEVELOPMENT, "source": SOURCE,
              "holdout": HOLDOUT, "replication": REPLICATION}
    for left, seeds in groups.items():
        if len(seeds) != len(set(seeds)):
            raise AssertionError("Duplicated cohort seed")
        for right, other in groups.items():
            if left != right and set(seeds).intersection(other):
                raise AssertionError("Cohort leakage")
    report = {
        "version": 1,
        "cohorts": {name: list(seeds) for name, seeds in groups.items()},
        "independent_worlds_source": len(SOURCE),
        "independent_worlds_holdout": len(HOLDOUT),
        "independent_worlds_replication": len(REPLICATION),
        "required_arms": sorted(REQUIRED_ARMS),
        "primary_effect": PRIMARY_EFFECT,
        "minimum_effect": PRIMARY_MIN_GAIN,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "status": "provisional_seed_manifest_not_executable_freeze",
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return report


def assert_ready_for_holdout(*, executable_frozen: bool,
                             source_weights_frozen: bool,
                             world_difficulty_frozen: bool,
                             matched_budget_verified: bool,
                             retention_test_implemented: bool,
                             implemented_arms: set[str]) -> None:
    """Intentionally fails until all scientific gates are genuinely satisfied."""
    missing = []
    for ok, label in (
        (executable_frozen, "executable protocol + version pinned"),
        (source_weights_frozen, "numerical source weights committed and frozen"),
        (world_difficulty_frozen, "difficulty calibrated only on development worlds"),
        (matched_budget_verified, "matched learning / reward budgets verified"),
        (retention_test_implemented, "transfer/retention test implemented"),
    ):
        if not ok:
            missing.append(label)
    missing += ["missing arm: " + a for a in
                sorted(REQUIRED_ARMS - implemented_arms)]
    if missing:
        raise RuntimeError("Held-out M5.6 execution BLOCKED: " +
                           "; ".join(missing))
