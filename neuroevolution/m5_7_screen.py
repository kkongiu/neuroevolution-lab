"""M5.7: paired diagnostic learned forward-model action selection.

Use only previously explored M5.6 DEVELOPMENT seeds 1080–1099.
This is deliberately *not* a new independent cohort or prospective proof.
Same 30-tick development history with learning_progress and fixed original
physical world; then disable all reward and weight updates, wipe transient
RNN hidden state, and vary only how the learned model informs movement:
no access / proper action-to-outcome mapping / deliberately permuted mapping.
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

DEV_SEEDS = tuple(range(1080, 1100))
DECISION_MODES = ("disabled", "model", "permuted")
CUTOFF = 30


def evaluate(seeds: tuple[int, ...] = DEV_SEEDS) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or any(
            s not in range(1080, 1100) for s in seeds):
        raise ValueError("Only previously explored development seeds 1080–1099")
    settings = replace(pilot_settings(), ticks=60, initial_food=9,
                       regeneration_probability=.55)
    rows = []
    for seed in seeds:
        for mode in DECISION_MODES:
            r = simulate_development(
                seed, settings, "learning_progress",
                predator_count=1, curiosity_gain=4.,
                learning_cutoff_tick=CUTOFF, reset_hidden_at_cutoff=True,
                model_decision_mode=mode)
            if abs(r["energy_residual"]) > 1e-6 or r["post_cutoff_weight_updates"] != 0:
                raise AssertionError("Post-cutoff update or caloric ledger violation")
            rows.append(r)
    index = {(r["seed"], r["model_decision_mode"]): r for r in rows}
    if len(index) != len(seeds)*len(DECISION_MODES):
        raise AssertionError("Unpaired world-level interventions")
    for seed in seeds:
        group = [index[(seed, mode)] for mode in DECISION_MODES]
        for field in ("survived_at_cutoff", "predictor_reward_total",
                      "predictor_training_transitions",
                      "predictor_reward_events", "novelty_reward_total"):
            if len({r[field] for r in group}) != 1:
                raise AssertionError("Counterfactuals have different training: "+field)
    summary = {}
    baseline = [index[(seed, "disabled")] for seed in seeds]
    for mode in DECISION_MODES:
        group = [index[(seed, mode)] for seed in seeds]
        paired = [r["survived"]-b["survived"]
                  for r, b in zip(group, baseline)]
        summary[mode] = {
            "surviving_worlds_at_cutoff": sum(r["survived_at_cutoff"] for r in group),
            "surviving_worlds_final": sum(r["survived"] for r in group),
            "paired_survival_delta_vs_disabled": statistics.mean(paired),
            "paired_world_differences": paired,
            "mean_final_population": statistics.mean(r["population"] for r in group),
            "mean_post_cutoff_food": statistics.mean(
                r["post_cutoff_food_eaten"] for r in group),
            "mean_post_cutoff_predations": statistics.mean(
                r["post_cutoff_predation_kills"] for r in group),
            "mean_model_informed_decisions": statistics.mean(
                r["post_cutoff_model_decisions"] for r in group),
            "mean_rnn_fallbacks": statistics.mean(
                r["post_cutoff_rnn_fallbacks"] for r in group),
        }
    serialized = json.dumps(rows,sort_keys=True,separators=(",",":"))
    return {
        "status":"development_reused_worlds_diagnostic_not_confirmatory",
        "seeds":list(seeds), "physical_trials":len(rows),
        "training_ticks":CUTOFF, "evaluation_ticks":settings.ticks-CUTOFF,
        "arms":list(DECISION_MODES),
        "world_seed_provenance":"REUSED earlier M5.6 development seed IDs",
        "model_learning_during_evaluation":False,
        "intrinsic_reward_during_evaluation":False,
        "engineered_utility_not_emergent":True,
        "summary":summary, "raw_rows":rows,
        "rows_sha256":hashlib.sha256(serialized.encode()).hexdigest(),
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-7/development-one-step-model-decision.json"))
    a=p.parse_args()
    report=evaluate()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                     encoding="utf-8")
    print("M5_7_DECISION_SCREEN "+json.dumps({
        "status":report["status"],"physical_trials":report["physical_trials"],
        "summary":report["summary"],"rows_sha256":report["rows_sha256"],
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
