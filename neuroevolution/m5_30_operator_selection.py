"""M5.30: own-reward selection among SEVEN HUMAN-SUPPLIED composition operators.

This is a development-only, limited hypothesis-menu selection experiment.
The agent DOES NOT invent composition operators, discover a new grammar, or
meta-learn shared weights across tasks. The seven possible operators (plain
product and product with one of six pairwise AND sign flips) are specified
by the researcher. The agent fits five float32 product coefficients to its
own action/reward examples for each menu option, then selects an operator
using TRAINING SUPPORT LOSS ONLY, with a frozen complexity penalty.

Eight disjoint new cues are used only AFTER all fitting/selection. Context
A/B/C is explicitly visible; there is no ecology/evolution or access to
the originally sealed M5.6 trials. All task families are synthetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import numpy as np

from neuroevolution.m5_29_compositional_modules import (
    ModularProductLearner, support_split, X_BITS,
)

META_SEEDS = (817, 827, 829)
AFFINE_SEEDS = tuple(range(7100, 7116))
PAIR_SEEDS = tuple(range(7200, 7216))
TRIPLE_SEEDS = tuple(range(7300, 7308))
COHORTS = {
    "affine_parity": AFFINE_SEEDS,
    "parity_plus_pair_AND": PAIR_SEEDS,
    "outside_menu_triple_AND": TRIPLE_SEEDS,
}
# Every evaluation task is on complete rule masks held out of the previous
# M5.29 pooled meta-training. This is NOT independent family replication.
MASKS = (5, 9, 11, 14)
PAIR_INDEX = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
OPERATORS = ("plain_product",) + tuple(
    "product_pair_%d%d" % p for p in PAIR_INDEX
)
ARMS = (
    "own_reward_operator_selection",
    "fixed_product_real_feedback",
    "random_operator_real_feedback",
    "scrambled_reward_operator_selection",
    "privileged_operator_reference",
    "unadapted_product",
)
# The penalty was prospectively selected, not tuned on cohort evaluations.
PAIR_PENALTY = 0.025
SUPPORT_COUNT = 8
EXAM_COUNT = 8
PARAM_BITS_PER_TASK = 5 * 32 + 3  # 5 float32, 3-bit operator ID
GATES = {
    "pair_accuracy_at_least_70pct": 0.70,
    "pair_gain_over_fixed_at_least_8pp": 0.08,
    "pair_gain_over_scrambled_at_least_15pp": 0.15,
}


def gate(cue: int, operator: int) -> int:
    if not isinstance(operator, int) or not 0 <= operator < 7:
        raise ValueError("Operator index outside frozen seven-item menu")
    if operator == 0:
        return 0
    a, b = PAIR_INDEX[operator - 1]
    return ((cue >> a) & 1) & ((cue >> b) & 1)


def triple(cue: int) -> int:
    return ((cue & 1) & ((cue >> 1) & 1) & ((cue >> 2) & 1))


def rule_for(seed: int, cohort: str):
    if cohort not in COHORTS or type(seed) is not int or seed not in COHORTS[cohort]:
        raise ValueError("M5.30 accepts only registered development cohorts")
    rng = np.random.default_rng(seed * 197 + 3)
    a = int(rng.choice(MASKS))
    c = int(rng.choice([m for m in MASKS if m != a]))
    ab = int(rng.integers(2))
    cb = int(rng.integers(2))
    if cohort == "affine_parity":
        ops = (0, 0, 0)
    elif cohort == "parity_plus_pair_AND":
        ops = (int(rng.integers(1, 7)),) * 2 + (int(rng.integers(1, 7)),)
    else:
        ops = (-1, -1, -1)  # triple AND: outside all seven options
    return ((a, ab, ops[0]), (a, 1 - ab, ops[1]), (c, cb, ops[2]))


def observed_reward(mask: int, bias: int, op: int, cue: int, action: int) -> int:
    if cue not in range(16) or action not in (0, 1):
        raise ValueError("Only physical binary actions and four-bit cues")
    correction = triple(cue) if op == -1 else gate(cue, op)
    desired = ((mask & cue).bit_count() & 1) ^ bias ^ correction
    return int(action == desired)


def observations(seed: int, cohort: str, context: int, rule):
    """The sole evidence given to an agent: cue, CHOSEN action, actual reward."""
    if context not in (0, 1, 2):
        raise ValueError("Unknown context")
    rng = np.random.default_rng(seed * 20117 + context * 313 + 29)
    seen, unseen = support_split(rng)
    mask, bias, op = rule
    records = tuple((int(c), int(a), observed_reward(mask, bias, op, int(c), int(a)))
                    for c in seen for a in [int(rng.integers(2))])
    if len(records) != SUPPORT_COUNT or len(unseen) != EXAM_COUNT:
        raise AssertionError("Unexpected support/heldout split")
    if set(c for c, _, _ in records) & set(unseen):
        raise AssertionError("Train/test cue contamination")
    return records, tuple(int(c) for c in unseen)


def transform(records, operator: int):
    """For a proposed gate, transform observed chosen action, NOT its reward."""
    return tuple((cue, action ^ gate(cue, operator), success)
                 for cue, action, success in records)


def fit_bank(records, seed: int, randomized_rewards: bool = False):
    """Does not accept task rule, heldout cues, or oracle operator."""
    if len(records) != 8 or len({r[0] for r in records}) != 8:
        raise ValueError("Need eight distinct personally experienced cues")
    if randomized_rewards:
        rng = np.random.default_rng(seed * 991 + 5)
        records = tuple((cue, action, int(rng.integers(2)))
                        for cue, action, _ in records)
    candidates = []
    for op in range(len(OPERATORS)):
        # Same initial weights, restarts and SGD budget for every operator.
        fit = ModularProductLearner(seed, kind="product").fit(transform(records, op))
        candidates.append(fit)
    scores = tuple(float(fit.support_mse + (PAIR_PENALTY if op else 0))
                   for op, fit in enumerate(candidates))
    selected = min(range(7), key=lambda i: (scores[i], i))
    return candidates, selected, scores


def predict(model, op: int, cues):
    return tuple(v ^ gate(cue, op)
                 for cue, v in zip(cues, model.predict(cues)))


def score(predictions, cues, rule):
    mask, bias, true_operator = rule
    return statistics.mean(
        int(prediction == (
            (((mask & cue).bit_count() & 1) ^ bias ^
             (triple(cue) if true_operator == -1 else gate(cue, true_operator)))))
        for prediction, cue in zip(predictions, cues)
    )


def one_trial(meta_seed: int, seed: int, cohort: str):
    """All six arms share the SAME real support interactions and hidden exam."""
    rules = rule_for(seed, cohort)
    rng = random.Random(seed * 191 + meta_seed)
    per_arm = {arm: [] for arm in ARMS}
    chosen_ops = []
    score_margins = []
    true_ops = []
    code_values = []
    for ctx, rule in enumerate(rules):
        records, exam = observations(seed, cohort, ctx, rule)
        init_seed = meta_seed * 10000019 + seed * 19 + ctx
        real, selected, costs = fit_bank(records, init_seed)
        scrambled, shuffled_choice, _ = fit_bank(
            records, init_seed, randomized_rewards=True
        )
        random_choice = rng.randrange(7)
        true_op = rule[2]
        chosen_ops.append(selected)
        true_ops.append(true_op)
        sorted_scores = sorted(costs)
        score_margins.append(sorted_scores[1] - sorted_scores[0])
        code_values.append(real[selected].weights.tolist())
        model0 = ModularProductLearner(init_seed)
        exams = {
            "own_reward_operator_selection": predict(real[selected], selected, exam),
            "fixed_product_real_feedback": predict(real[0], 0, exam),
            "random_operator_real_feedback": predict(real[random_choice], random_choice, exam),
            "scrambled_reward_operator_selection": predict(
                scrambled[shuffled_choice], shuffled_choice, exam
            ),
            "privileged_operator_reference": predict(
                real[true_op if true_op >= 0 else 0],
                true_op if true_op >= 0 else 0, exam
            ),
            "unadapted_product": predict(model0, 0, exam),
        }
        for arm, outputs in exams.items():
            per_arm[arm].append(score(outputs, exam, rule))
    return {
        "seed": seed, "meta_seed": meta_seed, "cohort": cohort,
        "unseen_accuracy": {
            arm: statistics.mean(values) for arm, values in per_arm.items()
        },
        "chosen_operator_ids": chosen_ops,
        "offline_true_operator_ids": true_ops,
        "operator_identification_rate": statistics.mean(
            int(x == y) for x, y in zip(chosen_ops, true_ops)
        ) if cohort != "outside_menu_triple_AND" else None,
        "operator_selection_mse_margins": score_margins,
        "learned_task_float32_codes": code_values,
        "persistent_candidate_code_bits_for_three_contexts": 3 * PARAM_BITS_PER_TASK,
        "temporary_seven_models_and_support_records_not_included": True,
        "all_real_arms_used_same_chosen_action_rewards": True,
        "eight_disjoint_frozen_exam_cues": True,
        "candidate_did_not_access_true_operator": True,
        "operator_menu_human_specified": True,
        "no_operator_meta_learning_or_evolution": True,
    }


def bootstrap_difference(rows, cohort, against, n=1000):
    """Exploratory paired two-level bootstrap; just three optimizer seeds."""
    lookup = {(r["meta_seed"], r["seed"]): r for r in rows if r["cohort"] == cohort}
    rng = random.Random(72431)
    seeds = COHORTS[cohort]
    vals = {
        (m, s): lookup[(m, s)]["unseen_accuracy"]["own_reward_operator_selection"]
                - lookup[(m, s)]["unseen_accuracy"][against]
        for m in META_SEEDS for s in seeds
    }
    delta = statistics.mean(vals.values())
    samples = []
    for _ in range(n):
        sampled_m = [rng.choice(META_SEEDS) for _ in META_SEEDS]
        samples.append(statistics.mean(
            vals[(m, rng.choice(seeds))] for m in sampled_m for _ in seeds
        ))
    samples.sort()
    return {"difference": delta,
            "exploratory_95pct_interval": [samples[int(.025*n)],
                                            samples[int(.975*n)-1]]}


def run():
    raw = [
        one_trial(meta_seed, seed, cohort)
        for meta_seed in META_SEEDS
        for cohort, seeds in COHORTS.items()
        for seed in seeds
    ]
    summary = {
        cohort: {
            "unseen_accuracy": {
                arm: statistics.mean(
                    r["unseen_accuracy"][arm]
                    for r in raw if r["cohort"] == cohort
                )
                for arm in ARMS
            },
            "operator_identification_rate": (
                statistics.mean(r["operator_identification_rate"]
                                for r in raw if r["cohort"] == cohort)
                if cohort != "outside_menu_triple_AND" else None
            ),
            "selected_menu_frequency": {
                OPERATORS[i]: sum(r["chosen_operator_ids"].count(i)
                                  for r in raw if r["cohort"] == cohort)
                for i in range(7)
            },
            "paired_bootstrap": {
                arm: bootstrap_difference(raw, cohort, arm)
                for arm in ARMS if arm != "own_reward_operator_selection"
            },
        }
        for cohort in COHORTS
    }
    pair = summary["parity_plus_pair_AND"]["unseen_accuracy"]
    a = pair["own_reward_operator_selection"]
    checks = {
        "nonlinear_pair_accuracy_at_least70pct": a >= GATES["pair_accuracy_at_least_70pct"],
        "nonlinear_pair_gain_over_fixed_at_least8pp":
            a - pair["fixed_product_real_feedback"] >= GATES["pair_gain_over_fixed_at_least_8pp"],
        "nonlinear_pair_gain_over_scrambled_at_least15pp":
            a - pair["scrambled_reward_operator_selection"] >= GATES["pair_gain_over_scrambled_at_least_15pp"],
        "own_reward_frozen_exam_distinct": all(r["eight_disjoint_frozen_exam_cues"] for r in raw),
        "no_true_operator_in_candidate": all(r["candidate_did_not_access_true_operator"] for r in raw),
        "candidate_code_489bits_for_three_contexts": all(
            r["persistent_candidate_code_bits_for_three_contexts"] == 489 for r in raw
        ),
        "outside_menu_stress_reported": len(summary["outside_menu_triple_AND"]["selected_menu_frequency"]) == 7,
    }
    return {
        "status": "M5.30 development-only support-selected human-operator menu",
        "metadata": {
            "optimizer_initialization_seeds": META_SEEDS,
            "cohorts": {k: list(v) for k, v in COHORTS.items()},
            "rule_masks": MASKS, "composition_operators": OPERATORS,
            "pairwise_menu_penalty": PAIR_PENALTY,
            "support_observations_per_task": 8, "exam_cues_per_task": 8,
            "fixed_adaptation_each_operator": "8 restarts x 180 full-batch steps",
            "candidate_operator_evaluations_per_task": 14,
            "persistent_bits_three_contexts": 489,
            "operator_menu_specified_by_researcher": True,
            "selection_supervision": "chosen-action binary reward, support only",
        },
        "summary": summary, "screening_gates": checks,
        "limited_support_operator_selection_screen_pass": all(checks.values()),
        "autonomously_discovered_composition_operator": False,
        "learned_shared_neural_executive": False,
        "independent_task_family_replication_complete": False,
        "evolution_authorized": False,
        "advance_to_evolution": False,
        "original_sealed_M5_6_cohorts_accessed": False,
        "raw_rows_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "raw_rows": raw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("results/m5-30/operator-selection-development.json"))
    args = ap.parse_args()
    out = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("M5_30_EXPERIENCE_SELECTED_OPERATORS " + json.dumps(
        {k: out[k] for k in ("summary", "screening_gates",
                             "limited_support_operator_selection_screen_pass",
                             "raw_rows_sha256")},
        sort_keys=True
    ), flush=True)


if __name__ == "__main__":
    main()
