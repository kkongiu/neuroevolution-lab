"""M5.6 calibrated exploratory comparison, development worlds ONLY.

Calibrated environment was selected with no-curiosity data on seeds 1020-1031.
This script uses NEW development seeds 1032-1055 exclusively, and does not
inspect or open M5.6 source (161xxx), holdout (162xxx), or replication (163xxx).
Seven arms have identical native physical rules; all controls are descriptive,
NOT proof of emergent adaptive curiosity or a confirmatory test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path

from .m5_6_balance_pilot import settings as pilot_settings
from .m5_6_online import simulate_development

SEEDS = tuple(range(1032, 1056))
PREDATORS = 1
ARMS = (
    ("no_curiosity", "no_novelty", 0.0, False),
    ("weak_novelty", "novelty", .25, False),
    ("moderate_novelty", "novelty", 1.0, False),
    ("strong_novelty", "novelty", 4.0, False),
    ("engineered_energy_gate", "novelty", 1.0, True),
    ("randomized_feedback_placebo", "shuffled_novelty", 1.0, False),
    ("no_synaptic_learning", "frozen", 0.0, False),
)
# Selected prospectively by baseline-only calibration; never tune on arm effects.
HORIZON = 60
INITIAL_FOOD = 9
FOOD_REGENERATION = .55


def execute(seeds: tuple[int, ...] = SEEDS) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or any(
            s not in range(1032, 1056) for s in seeds):
        raise ValueError("Only independent calibrated DEVELOPMENT seeds 1032-1055")
    settings = replace(
        pilot_settings(), ticks=HORIZON, initial_food=INITIAL_FOOD,
        regeneration_probability=FOOD_REGENERATION)
    settings.validate()
    rows = []
    for seed in seeds:
        for label, arm, gain, gate in ARMS:
            r = simulate_development(
                seed, settings, arm, predator_count=PREDATORS,
                curiosity_gain=gain, energy_gate=gate)
            if abs(r["energy_residual"]) > 1e-6:
                raise AssertionError("Caloric physical ledger is not balanced")
            rows.append({"condition": label, **r})
    expected = {(seed, label) for seed in seeds for label, *_ in ARMS}
    observed = {(r["seed"], r["condition"]) for r in rows}
    if observed != expected or len(rows) != len(expected):
        raise AssertionError("Incomplete or duplicated paired design")
    by = {(r["seed"], r["condition"]): r for r in rows}
    summary = {}
    for label, *_ in ARMS:
        data = [by[(seed, label)] for seed in seeds]
        control = [by[(seed, "no_curiosity")] for seed in seeds]
        differences = [a["survived"] - b["survived"]
                       for a, b in zip(data, control)]
        summary[label] = {
            "survivor_worlds": sum(r["survived"] for r in data),
            "total_worlds": len(data),
            "survival_fraction": statistics.mean(r["survived"] for r in data),
            "paired_world_survival_difference_vs_no_curiosity":
                statistics.mean(differences),
            "paired_world_differences": differences,
            "mean_final_population": statistics.mean(r["population"] for r in data),
            "mean_food_eaten": statistics.mean(r["food_eaten"] for r in data),
            "mean_predation_kills": statistics.mean(r["predation_kills"] for r in data),
            "mean_first_visits": statistics.mean(
                r["unique_individual_arrivals"] for r in data),
            "mean_moves": statistics.mean(r["physical_moves"] for r in data),
            "mean_novelty_reward": statistics.mean(
                r["novelty_reward_total"] for r in data),
            "mean_weight_update_actions": statistics.mean(
                r["updates_with_nonzero_drift"] for r in data),
        }
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "status": "development_exploratory_not_heldout",
        "inference": "Effect size is descriptive, not confirmatory. "
                     "No learned curiosity-survival arbitration established.",
        "worlds": list(seeds),
        "calibration_worlds_used": [],
        "reserved_holdout_worlds_used": [],
        "horizon": HORIZON, "predators": PREDATORS,
        "initial_food": INITIAL_FOOD,
        "regeneration_probability": FOOD_REGENERATION,
        "physical_trials": len(rows),
        "summary": summary, "raw_rows": rows,
        "rows_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(
        "results/m5-6/calibrated-curiosity-development.json"))
    args = parser.parse_args()
    report = execute()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    print("M5_6_CALIBRATED_DEVELOPMENT " + json.dumps({
        "status": report["status"], "physical_trials": report["physical_trials"],
        "summary": report["summary"], "rows_sha256": report["rows_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
