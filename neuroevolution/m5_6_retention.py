"""M5.6 learned-synapse retention: paired within-world counterfactual.

Development-only seeds 1056–1079, disjoint from all other M5.6 uses.
Tick 30 divides training and evaluation. Both learned arms have identical
training trajectories. Post switch: intrinsic reward OFF, synaptic updates OFF,
transient RNN activations reset equally. Compare keeping vs resetting learned
synapses, on the same initially evolved world state and physical hunger.
No sealed source/holdout/replication cohort accessed.
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

SEEDS = tuple(range(1056, 1080))
CUTOFF = 30
ARMS = (
    ("trained_curiosity_retained", "novelty", 4., False),
    ("trained_curiosity_reset_synapses", "novelty", 4., True),
    ("trained_no_curiosity_retained", "no_novelty", 0., False),
    ("frozen_entire_life", "frozen", 0., False),
)


def execute(seeds: tuple[int, ...] = SEEDS) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or any(
            seed not in range(1056, 1080) for seed in seeds):
        raise ValueError("Only reserved development retention seeds 1056–1079")
    settings = replace(pilot_settings(), ticks=60, initial_food=9,
                       regeneration_probability=.55)
    rows = []
    for seed in seeds:
        for label, arm, gain, reset in ARMS:
            result = simulate_development(
                seed, settings, arm, predator_count=1, curiosity_gain=gain,
                learning_cutoff_tick=CUTOFF,
                reset_weights_at_cutoff=reset, reset_hidden_at_cutoff=True)
            if abs(result["energy_residual"]) > 1e-6:
                raise AssertionError("Physical energy is not conserved")
            if result["post_cutoff_weight_updates"] != 0:
                raise AssertionError("Synaptic updates after cutoff are forbidden")
            rows.append({"condition": label, **result})
    index = {(r["seed"], r["condition"]): r for r in rows}
    if len(index) != len(seeds) * len(ARMS):
        raise AssertionError("Incomplete paired design")
    for seed in seeds:
        a = index[(seed, "trained_curiosity_retained")]
        b = index[(seed, "trained_curiosity_reset_synapses")]
        # Exact same development history: no modification before split.
        for key in ("survived_at_cutoff", "weights_changed",
                    "physical_food_reward_total"):
            # Physical food reward total is an all-tick aggregate: it may
            # diverge after the cutoff. Only compare prefix-state metrics.
            if key == "physical_food_reward_total":
                continue
            if key == "weights_changed":
                continue
            if a[key] != b[key]:
                raise AssertionError("Learning branch divergence before cutoff")
    summary = {}
    for label, *_ in ARMS:
        rs = [index[(seed, label)] for seed in seeds]
        baseline = [index[(seed, "trained_curiosity_reset_synapses")]
                    for seed in seeds]
        paired = [r["survived"] - q["survived"]
                  for r, q in zip(rs, baseline)]
        summary[label] = {
            "survival_worlds_at_cutoff": sum(r["survived_at_cutoff"] for r in rs),
            "survival_worlds_at_final_tick": sum(r["survived"] for r in rs),
            "paired_survival_delta_vs_reset": statistics.mean(paired),
            "paired_world_differences": paired,
            "mean_post_cutoff_food": statistics.mean(
                r["post_cutoff_food_eaten"] for r in rs),
            "mean_post_cutoff_predation_kills": statistics.mean(
                r["post_cutoff_predation_kills"] for r in rs),
            "mean_post_cutoff_births": statistics.mean(
                r["post_cutoff_births"] for r in rs),
            "post_cutoff_synaptic_updates": sum(
                r["post_cutoff_weight_updates"] for r in rs),
        }
    serialized = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "status": "development_only_retention_not_confirmatory",
        "worlds": list(seeds), "physical_trials": len(rows),
        "training_ticks": CUTOFF, "evaluation_ticks": settings.ticks-CUTOFF,
        "same_history_causal_pair": [
            "trained_curiosity_retained", "trained_curiosity_reset_synapses"],
        "learned_synapses_only_between_pair": True,
        "novelty_reward_during_evaluation": 0,
        "learning_during_evaluation": False,
        "hidden_memory_reset_for_both": True,
        "interpretation": "Retained vs original numerical weights on same "
                          "physical pre-cutoff history; not a proof of "
                          "hunger-adaptive curiosity or generalization.",
        "summary": summary, "raw_rows": rows,
        "rows_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(
        "results/m5-6/development-synapse-retention.json"))
    a = p.parse_args()
    result = execute()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n",
                     encoding="utf-8")
    print("M5_6_SYNAPSE_RETENTION " + json.dumps({
        "status": result["status"],
        "physical_trials": result["physical_trials"],
        "summary": result["summary"],
        "rows_sha256": result["rows_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
