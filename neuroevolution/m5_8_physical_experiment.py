"""M5.8: physical curriculum variants and exogenous shocks comparison.

This is exploratory DEVELOPMENT ONLY, not a new validation cohort or a
conclusive intelligence test. Uses all dedicated worlds 2000–2063, 60 ticks.
Compare same exogenous shock events under progressive/uniform/reversed world
layouts, and same learning budget per arm:
- progress model, learned predictions used at evaluation
- progress model, learned predictions NOT used at evaluation

Both train for 30 ticks, then no reward or synaptic modifications.
Equal 30-tick duration does NOT imply equal achieved updates when survival
differs; report actual update counts. Survival is measured per independent
world seed; do not count offspring as independent worlds.
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
from .m5_8_worlds import DEV_SEEDS
from .m5_8_variants import VARIANTS, audit_variants

MODES = ("disabled", "model")


def evaluate(seeds: tuple[int,...]=DEV_SEEDS) -> dict:
    if (not seeds or len(seeds)!=len(set(seeds)) or any(
            type(s) is not int or s not in DEV_SEEDS for s in seeds)):
        raise ValueError("Only unique dedicated M5.8 development seeds")
    settings=replace(pilot_settings(),ticks=60,initial_food=9,
                     regeneration_probability=.55)
    rows=[]
    for seed in seeds:
        for variant in VARIANTS:
            for mode in MODES:
                r=simulate_development(
                    seed,settings,"learning_progress",predator_count=1,
                    curiosity_gain=4.,learning_cutoff_tick=30,
                    reset_hidden_at_cutoff=True,model_decision_mode=mode,
                    progressive_world=True,stochastic_challenges=True,
                    world_variant=variant)
                if abs(r["energy_residual"]) > 1e-6:
                    raise AssertionError("Physical energy conservation broken")
                if r["post_cutoff_weight_updates"] != 0:
                    raise AssertionError("Evaluation learning not frozen")
                rows.append(r)
    expected={(seed,v,m) for seed in seeds for v in VARIANTS for m in MODES}
    index={(r["seed"],r["world_variant"],r["model_decision_mode"]):r
           for r in rows}
    if len(rows)!=len(expected) or set(index)!=expected:
        raise AssertionError("Incomplete/duplicate paired design")
    summary={}
    for v in VARIANTS:
        model=[index[(seed,v,"model")] for seed in seeds]
        baseline=[index[(seed,v,"disabled")] for seed in seeds]
        for a,b in zip(model,baseline):
            for key in ("survived_at_cutoff","world_sha256",
                        "predictor_reward_total",
                        "predictor_training_transitions",
                        "actual_shocks_proposed"):
                if a[key]!=b[key]:
                    raise AssertionError("Pre-cutoff history mismatch: "+key)
        results={}
        for mode,rs in (("model",model),("disabled",baseline)):
            results[mode]={
                "final_world_survival":sum(r["survived"] for r in rs),
                "world_count":len(rs),
                "cutoff_world_survival":sum(r["survived_at_cutoff"] for r in rs),
                "mean_final_population":statistics.mean(r["population"] for r in rs),
                "mean_food_eaten":statistics.mean(r["food_eaten"] for r in rs),
                "mean_predator_kills":statistics.mean(r["predation_kills"] for r in rs),
                "mean_training_policy_updates":statistics.mean(
                    r["updates_with_nonzero_drift"] for r in rs),
                "mean_model_decisions":statistics.mean(
                    r["post_cutoff_model_decisions"] for r in rs),
                "mean_drought_suppressed":statistics.mean(
                    r["regrowth_attempts_suppressed"] for r in rs),
                "mean_predator_extra_steps":statistics.mean(
                    r["predator_extra_steps"] for r in rs),
            }
        results["paired_world_differences_model_minus_disabled"]=[
            a["survived"]-b["survived"] for a,b in zip(model,baseline)]
        summary[v]=results
    data=json.dumps(rows,sort_keys=True,separators=(",",":"))
    return {
        "status":"exploratory_development_with_real_shocks_not_confirmatory",
        "world_seeds":list(seeds),"physical_trials":len(rows),
        "total_comparison_arms":len(VARIANTS)*len(MODES),
        "world_budget_audit":audit_variants(seeds),
        "same_seed_counterfactual_not_same_geometry_between_variants":True,
        "no_future_information":True,
        "summary":summary,"raw_rows":rows,
        "rows_sha256":hashlib.sha256(data.encode()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-8/physical-progressive-comparison.json"))
    args=p.parse_args()
    report=evaluate()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                        encoding="utf-8")
    print("M5_8_PHYSICAL_EXPERIMENT "+json.dumps({
        "status":report["status"],
        "physical_trials":report["physical_trials"],
        "summary":report["summary"],
        "variant_budgets":report["world_budget_audit"],
        "rows_sha256":report["rows_sha256"],
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
