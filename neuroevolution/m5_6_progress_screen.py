"""M5.6 prospective LEARNING-PROGRESS development screen.

New development seeds 1080–1099 (all remaining local development IDs).
Do not open source (161xxx), heldout (162xxx) or replication (163xxx).
Same environment previously calibrated solely on no-curiosity outcomes.
The within-seed progress-retained and progress-reset pair shares exact
training trajectory, and turns off both predictor/curiosity reward and
policy/predictor weight updates at the same 30 tick cutoff.

Exploratory diagnostic: no preregistered M5.6 survival claim permitted.
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

SEEDS = tuple(range(1080, 1100))
ARMS = (
    ("progress_retained", "learning_progress", 4., False),
    ("progress_reset_synapses", "learning_progress", 4., True),
    ("strong_novelty_retained", "novelty", 4., False),
    ("no_intrinsic_reward_retained", "no_novelty", 0., False),
    ("frozen_all_along", "frozen", 0., False),
)
CUTOFF = 30


def execute(seeds: tuple[int, ...] = SEEDS) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or any(
            seed not in range(1080, 1100) for seed in seeds):
        raise ValueError("Only unused development world IDs 1080–1099")
    s = replace(pilot_settings(), ticks=60, initial_food=9,
                regeneration_probability=.55)
    rows = []
    for seed in seeds:
        for label, arm, gain, reset in ARMS:
            r = simulate_development(
                seed, s, arm, predator_count=1, curiosity_gain=gain,
                learning_cutoff_tick=CUTOFF, reset_weights_at_cutoff=reset,
                reset_hidden_at_cutoff=True)
            if abs(r["energy_residual"]) > 1e-6:
                raise AssertionError("Energy ledger failed")
            if r["post_cutoff_weight_updates"] != 0:
                raise AssertionError("Post-cutoff policy updates!")
            rows.append({"condition": label, **r})
    indexed = {(r["seed"], r["condition"]): r for r in rows}
    if len(indexed) != len(seeds)*len(ARMS):
        raise AssertionError("Paired matrix incomplete")
    for seed in seeds:
        a = indexed[(seed, "progress_retained")]
        b = indexed[(seed, "progress_reset_synapses")]
        for key in ("survived_at_cutoff", "predictor_reward_total",
                    "predictor_training_transitions",
                    "predictor_reward_events", "novelty_reward_total"):
            if a[key] != b[key]:
                raise AssertionError("Training histories not identical: "+key)
    summary = {}
    baseline = [indexed[(seed, "progress_reset_synapses")] for seed in seeds]
    for label, *_ in ARMS:
        here = [indexed[(seed, label)] for seed in seeds]
        summary[label] = {
            "survivor_worlds_at_cutoff": sum(r["survived_at_cutoff"] for r in here),
            "survivor_worlds_at_final": sum(r["survived"] for r in here),
            "paired_world_differences_vs_progress_reset": [
                a["survived"]-z["survived"] for a, z in zip(here, baseline)],
            "mean_final_population": statistics.mean(r["population"] for r in here),
            "mean_food_post_cutoff": statistics.mean(
                r["post_cutoff_food_eaten"] for r in here),
            "mean_prediction_reward": statistics.mean(
                r["predictor_reward_total"] for r in here),
            "mean_predictor_training_transitions": statistics.mean(
                r["predictor_training_transitions"] for r in here),
            "mean_predictor_positive_progress_events": statistics.mean(
                r["predictor_reward_events"] for r in here),
            "mean_policy_weight_changing_updates": statistics.mean(
                r["updates_with_nonzero_drift"] for r in here),
        }
    canon=json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "status":"development_only_prediction_progress_not_confirmatory",
        "worlds":list(seeds), "physical_trials":len(rows),
        "learning_ticks":CUTOFF, "evaluation_ticks":s.ticks-CUTOFF,
        "reward_during_evaluation":0,
        "predictor_updates_during_evaluation":0,
        "policy_updates_during_evaluation":0,
        "interpretation":"Within-world trained-vs-reset synaptic coefficients; "
                         "no novel-world transfer or adaptive self-preservation proof",
        "summary":summary, "raw_rows":rows,
        "rows_sha256":hashlib.sha256(canon.encode()).hexdigest(),
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(
        "results/m5-6/development-prediction-progress.json"))
    a=parser.parse_args()
    result=execute()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                     encoding="utf-8")
    print("M5_6_PREDICTION_PROGRESS " + json.dumps({
        "status":result["status"],
        "physical_trials":result["physical_trials"],
        "summary":result["summary"],
        "rows_sha256":result["rows_sha256"],
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
