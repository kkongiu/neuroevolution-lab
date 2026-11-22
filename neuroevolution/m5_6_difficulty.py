"""M5.6 development-only ecological difficulty calibration.

Select environmental difficulty WITHOUT using any curiosity-arm outcome.
Only the original no-curiosity agent is measured on development seeds.
No final-world seeds, no post-selection effect claims, no modified predator
physics. Selection targets 50% population persistence to avoid pilot ceilings.
The selected setting is a proposal for prospective preregistration, not a
permission to consume holdout worlds.
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

SEEDS = tuple(range(1020, 1032))
# Vary only food environment and the observation duration. Predator AI,
# per-tick predation, recurrent learning and caloric physics are unchanged.
CANDIDATES = (
    (36, 6, 0.25), (36, 6, 0.55),
    (36, 9, 0.25), (36, 9, 0.55),
    (48, 6, 0.25), (48, 6, 0.55),
    (48, 9, 0.25), (48, 9, 0.55),
    (60, 6, 0.25), (60, 6, 0.55),
    (60, 9, 0.25), (60, 9, 0.55),
)
PREDATORS = 1
TARGET_PERSISTENCE = 0.5


def calibrate(seeds: tuple[int, ...] = SEEDS) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or any(
            seed not in range(1000, 1100) for seed in seeds):
        raise ValueError("Calibration may only use distinct development seeds")
    rows = []
    candidates = []
    for duration, food, regen in CANDIDATES:
        s = replace(pilot_settings(), ticks=duration, initial_food=food,
                    regeneration_probability=regen)
        s.validate()
        here = []
        for seed in seeds:
            result = simulate_development(seed, s, "no_novelty",
                                          predator_count=PREDATORS)
            if abs(result["energy_residual"]) > 1e-6:
                raise AssertionError("Native physical energy imbalance")
            row = {"duration": duration, "initial_food": food,
                   "regeneration_probability": regen, **result}
            rows.append(row)
            here.append(row)
        persistence = statistics.mean(r["survived"] for r in here)
        candidates.append({
            "duration": duration, "initial_food": food,
            "regeneration_probability": regen,
            "survival_worlds": sum(r["survived"] for r in here),
            "total_worlds": len(here),
            "survival_rate": persistence,
            "mean_final_population": statistics.mean(
                r["population"] for r in here),
            "mean_food_eaten": statistics.mean(r["food_eaten"] for r in here),
            "distance_from_target": abs(persistence - TARGET_PERSISTENCE),
        })
    # Fixed tie-breaking: prefer longer horizon (to allow potential lifetime
    # learning), then lexicographic food and regeneration parameters.
    winner = min(candidates, key=lambda r: (
        r["distance_from_target"], -r["duration"],
        r["initial_food"], r["regeneration_probability"]))
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "status": "development_calibration_only",
        "selection_policy": "no_curiosity_only_survival_nearest_0.5",
        "selection_is_not_survival_effect_estimation": True,
        "seeds": list(seeds), "predators": PREDATORS,
        "num_runs": len(rows),
        "candidates": candidates, "proposed_setting": winner,
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "raw_rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(
        "results/m5-6/development-difficulty-calibration.json"))
    a = p.parse_args()
    result = calibrate()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                     encoding="utf-8")
    print("M5_6_DIFFICULTY_CALIBRATION " + json.dumps({
        "status": result["status"], "num_runs": result["num_runs"],
        "proposed_setting": result["proposed_setting"],
        "candidates": result["candidates"],
        "raw_sha256": result["raw_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
