"""M5.6 DEVELOPMENT-only preregistration preparation: curiosity/survival pilot.

World seeds are restricted to the previously reserved 1000–1099 range.
This is NOT the prospective >=32-world M5.6 held-out campaign, and does not
support any general survival benefit claim. Exact matched initial geometry,
food, predator count, founder genomes, and random event seeds per cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path

from . import m5_5b
from .m5_6_online import simulate_development

DEVELOPMENT_SEEDS = tuple(range(1000, 1012))
PRESSURES = (0, 1, 3)
POLICIES = (
    ("no_curiosity", "no_novelty", 0., False),
    ("weak_curiosity", "novelty", .25, False),
    ("moderate_curiosity", "novelty", 1., False),
    ("strong_curiosity", "novelty", 4., False),
    ("energy_gated_scripted", "novelty", 1., True),
    ("shuffled_reward_control", "shuffled_novelty", 1., False),
    ("frozen_synapses_control", "frozen", 0., False),
)


def settings() -> m5_5b.Settings:
    return m5_5b.Settings(
        seed_start=300, seeds=1, shifted_start=400, shifted_seeds=1,
        ticks=36, size=11, initial_food=9, food_cap=14,
        initial_population=8, max_population=18,
        checkpoint_every=12, no_food_controls=0)


def evaluate(*, seeds: tuple[int, ...] = DEVELOPMENT_SEEDS,
             pressures: tuple[int, ...] = PRESSURES) -> dict:
    if (not seeds or len(seeds) != len(set(seeds)) or
            any(seed not in range(1000, 1100) for seed in seeds)):
        raise ValueError("Only unique reserved M5.6 development seeds allowed")
    if not pressures or len(pressures) != len(set(pressures)) or any(
            p not in PRESSURES for p in pressures):
        raise ValueError("Invalid pressure set")
    s = settings()
    rows = []
    for pressure in pressures:
        for seed in seeds:
            for label, arm, gain, gate in POLICIES:
                row = simulate_development(seed, s, arm,
                    predator_count=pressure,
                    curiosity_gain=gain, energy_gate=gate)
                if abs(row["energy_residual"]) > 1e-6:
                    raise AssertionError("Physical energy created or destroyed")
                rows.append({"policy": label, **row})
    result = {"status": "development_only_not_confirmatory",
              "seeds": list(seeds), "pressures": list(pressures),
              "worlds": len(rows), "ticks": s.ticks,
              "policies": [p[0] for p in POLICIES],
              "by_pressure": {}, "rows": rows}
    for pressure in pressures:
        subset = [r for r in rows if r["predators"] == pressure]
        indexed = {(r["seed"], r["policy"]): r for r in subset}
        if len(indexed) != len(seeds) * len(POLICIES):
            raise AssertionError("Unmatched factorial trial matrix")
        result["by_pressure"][str(pressure)] = {}
        baseline = "no_curiosity"
        for label, *_ in POLICIES:
            group = [indexed[(seed, label)] for seed in seeds]
            diffs = [indexed[(seed, label)]["survived"] -
                     indexed[(seed, baseline)]["survived"]
                     for seed in seeds]
            result["by_pressure"][str(pressure)][label] = {
                "worlds_survived": sum(r["survived"] for r in group),
                "worlds_total": len(group),
                "mean_survival": statistics.mean(r["survived"] for r in group),
                "paired_survival_diff_vs_none": statistics.mean(diffs),
                "paired_world_differences": diffs,
                "mean_population": statistics.mean(r["population"] for r in group),
                "mean_predation_kills": statistics.mean(r["predation_kills"] for r in group),
                "mean_food_eaten": statistics.mean(r["food_eaten"] for r in group),
                "mean_unique_arrivals": statistics.mean(
                    r["unique_individual_arrivals"] for r in group),
                "mean_physical_moves": statistics.mean(
                    r["physical_moves"] for r in group),
                "mean_action_start_energy": statistics.mean(
                    r["mean_action_start_energy"] for r in group),
            }
    digest = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    result["rows_sha256"] = hashlib.sha256(digest.encode()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(
        "results/m5-6/development-curiosity-survival-pilot.json"))
    args = parser.parse_args()
    report = evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print("M5_6_DEVELOPMENT_PILOT " + json.dumps({
        "status": report["status"],
        "worlds": report["worlds"],
        "worlds_per_policy_pressure": len(DEVELOPMENT_SEEDS),
        "by_pressure": report["by_pressure"],
        "rows_sha256": report["rows_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
