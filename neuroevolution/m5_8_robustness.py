"""M5.8 pressure calibration and novel-world forecast ablation.

PREDECLARED:
* Calibration: ONLY development seeds 2000..2023, baseline RNN with no
  model-based action selection, progressive map, 60 ticks, full vs moderate
  shock pressure. Choose candidate nearest target 0.40 baseline world survival;
  ties choose moderate. No outcome from new-world check can affect the choice.
* Novel-world screen: 32 previously unused seeds 2064..2095, three matched
  spatial variants, paired disabled/true/permuted action-forecast access.
  Same actual training history across 3 modes; all learning ends at tick 30.
  The policy starts from new (not cross-world pretrained) weights in EVERY
  world. Therefore this is **out-of-sample procedural generalization**,
  NOT transfer of stored learned skills/weights to new geography.

All M5.6 source, holdout, replication seeds remain excluded. Report every
world and paired contrast, no selective successful-world reporting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .m5_6_balance_pilot import settings as pilot_settings
from .m5_6_online import simulate_development
from .m5_8_worlds import DEV_SEEDS, TRANSFER_SEEDS
from .m5_8_variants import VARIANTS

CALIBRATION_SEEDS = tuple(range(2000, 2024))
CHECK_SEEDS = TRANSFER_SEEDS
PRESSURES = ("full", "moderate")
DECISION_MODES = ("disabled", "model", "permuted")
TARGET_BASELINE_SURVIVAL = .40


def physical_run(seed: int, variant: str, mode: str,
                 pressure: str) -> dict:
    s = replace(pilot_settings(), ticks=60, initial_food=9,
                regeneration_probability=.55)
    row = simulate_development(
        seed, s, "learning_progress", predator_count=1,
        curiosity_gain=4., learning_cutoff_tick=30,
        reset_hidden_at_cutoff=True, model_decision_mode=mode,
        progressive_world=True, stochastic_challenges=True,
        world_variant=variant, shock_pressure=pressure)
    if abs(row["energy_residual"]) > 1e-6:
        raise AssertionError("World energy ledger broken")
    if row["post_cutoff_weight_updates"] != 0:
        raise AssertionError("Post-cutoff policy changes")
    return row


def calibration(seeds: tuple[int, ...] = CALIBRATION_SEEDS) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or any(
            type(s) is not int or s not in CALIBRATION_SEEDS for s in seeds):
        raise ValueError("Calibration seeds must be development-only, 2000–2023")
    rows = []
    levels = {}
    for pressure in PRESSURES:
        sample = []
        for seed in seeds:
            row = physical_run(seed, "progressive", "disabled", pressure)
            sample.append(row)
            rows.append(row)
        survivors = sum(r["survived"] for r in sample)
        levels[pressure] = {
            "survivor_worlds": survivors,
            "world_count": len(sample),
            "survival_fraction": survivors / len(sample),
            "mean_extra_predator_steps": (
                sum(r["predator_extra_steps"] for r in sample) / len(sample)),
        }
    # Deterministic baseline-only selection; cannot look at model-arm results.
    winner = min(PRESSURES, key=lambda p: (
        abs(levels[p]["survival_fraction"] - TARGET_BASELINE_SURVIVAL),
        0 if p == "moderate" else 1))
    return {
        "chosen_pressure": winner,
        "baseline_only": True,
        "target_survival": TARGET_BASELINE_SURVIVAL,
        "seeds": list(seeds),
        "candidate_metrics": levels,
        "raw_rows": rows,
    }


def evaluate(*, calibration_seeds: tuple[int, ...] = CALIBRATION_SEEDS,
             check_seeds: tuple[int, ...] = CHECK_SEEDS) -> dict:
    if not check_seeds or len(check_seeds) != len(set(check_seeds)) or any(
            type(s) is not int or s not in CHECK_SEEDS for s in check_seeds):
        raise ValueError("Only untouched novel-world development seeds 2064–2095")
    if set(check_seeds) & set(calibration_seeds):
        raise AssertionError("World leakage")
    calibrated = calibration(calibration_seeds)
    pressure = calibrated["chosen_pressure"]
    rows = []
    for seed in check_seeds:
        for variant in VARIANTS:
            for mode in DECISION_MODES:
                rows.append(physical_run(seed, variant, mode, pressure))
    index = {(r["seed"], r["world_variant"], r["model_decision_mode"]):r
             for r in rows}
    if len(rows) != len(check_seeds)*len(VARIANTS)*len(DECISION_MODES):
        raise AssertionError("Missing or duplicate checks")
    summary = {}
    for variant in VARIANTS:
        block = {
            mode: [index[(s, variant, mode)] for s in check_seeds]
            for mode in DECISION_MODES}
        for seed in check_seeds:
            entries = [index[(seed, variant, mode)] for mode in DECISION_MODES]
            for key in ("world_sha256", "survived_at_cutoff",
                        "predictor_reward_total", "predictor_training_transitions",
                        "novelty_reward_total", "actual_shocks_proposed",
                        "shock_pressure"):
                if len({e[key] for e in entries}) != 1:
                    raise AssertionError("Non-matched pre-evaluation history: "+key)
        table = {}
        for mode in DECISION_MODES:
            records = block[mode]
            table[mode] = {
                "survivor_worlds_cutoff": sum(r["survived_at_cutoff"] for r in records),
                "survivor_worlds_final": sum(r["survived"] for r in records),
                "mean_final_population": sum(r["population"] for r in records)/len(records),
                "mean_forecast_driven_decisions": sum(
                    r["post_cutoff_model_decisions"] for r in records)/len(records),
                "mean_food_collected": sum(r["food_eaten"] for r in records)/len(records),
            }
        for mode in ("model", "permuted"):
            table["paired_"+mode+"_minus_disabled"] = [
                a["survived"]-b["survived"] for a,b in
                zip(block[mode], block["disabled"])]
        table["paired_model_minus_permuted"] = [
            a["survived"]-b["survived"] for a,b in
            zip(block["model"], block["permuted"])]
        summary[variant] = table
    canon = json.dumps(rows,sort_keys=True,separators=(",",":"))
    return {
        "status":"novel_development_worlds_exploratory_not_final_holdout",
        "calibration":calibrated,
        "frozen_choice_before_novel_worlds":pressure,
        "new_world_seeds":list(check_seeds),
        "no_cross_world_synaptic_transfer":True,
        "physical_trials":len(rows),
        "world_level_results":summary,
        "raw_rows":rows,
        "rows_sha256":hashlib.sha256(canon.encode()).hexdigest(),
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path(
        "results/m5-8/pressure-and-novel-world-check.json"))
    a=p.parse_args()
    report=evaluate()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                     encoding="utf-8")
    print("M5_8_NOVEL_WORLD_RESULTS "+json.dumps({
        "status":report["status"],
        "calibration":{k:v for k,v in report["calibration"].items()
                       if k!="raw_rows"},
        "physical_trials":report["physical_trials"],
        "world_level_results":report["world_level_results"],
        "rows_sha256":report["rows_sha256"]
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
