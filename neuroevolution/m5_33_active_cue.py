"""M5.33: active *sensory cue selection* and exact identifiability study.

The active version-space sensor selector uses a FULL HUMAN-DESIGNED
HYPOTHESIS GRAMMAR including the true synthetic family. It never
receives the true rule ID, but knows the possible family. It is NOT
learned curiosity, neural operator invention or open-ended cognition.

A separately learned M5.32 two-neuron model (same number of real
rewards/parameters/SGD updates in ALL arms) tests whether increased
information is useful to the neural learner. Actions are precommitted
random responses by the test harness, not chosen by an organism.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

from neuroevolution.m5_31_synaptic_growth import (
    AFFINE_MASKS, DISJOINT_PAIRS, wanted_action,
)
from neuroevolution.m5_32_dual_soft_synapses import DualSoftSynapses

COHORTS = {
    "plain_parity": tuple(range(12100, 12108)),
    "pair_interaction": tuple(range(12200, 12212)),
    "triple_interaction": tuple(range(12300, 12312)),
    "two_disjoint_pairs": tuple(range(12400, 12412)),
}
OPTIMIZER_SEEDS = (1031, 1033)
POLICIES = (
    "random_cues",
    "coverage_only",
    "family_version_space_active",
    "scrambled_belief_active",
)
START_CUES = 2
SUPPORT_BUDGET = 8
CUE_COUNT = 16
BITS = 4
SGD_DIVERSITY = 0.0
CUES = tuple(range(CUE_COUNT))
INTERACTIONS = (
    (0,)
    + tuple(1 << a | 1 << b for a, b in itertools.combinations(range(4), 2))
    + tuple(sum(1 << x for x in triple)
            for triple in itertools.combinations(range(4), 3))
    + DISJOINT_PAIRS
)
CHECK_THRESHOLDS = {
    "posterior_relative_reduction": .20,
    "identifiable_gain": .10,
    "two_pair_full16_gain_vs_random": .05,
    "two_pair_heldout8_gain_vs_random": .05,
    "two_pair_full16_gain_vs_scrambled_planner": .05,
    "two_pair_full16_gain_vs_coverage": .03,
    "two_pair_full16_accuracy": .60,
}


def fresh_rule_for(seed: int, cohort: str):
    """Entirely new DEV seeds, SAME M5.31 synthetic family grammar."""
    if cohort not in COHORTS or type(seed) is not int or seed not in COHORTS[cohort]:
        raise ValueError("M5.33 unregistered development seed")
    rng = np.random.default_rng(seed * 181081 + 131)
    a = int(rng.choice(AFFINE_MASKS))
    c = int(rng.choice([v for v in AFFINE_MASKS if v != a]))
    bias_a, bias_c = int(rng.integers(2)), int(rng.integers(2))
    if cohort == "plain_parity":
        effects = (0, 0, 0)
    elif cohort in ("pair_interaction", "triple_interaction"):
        n = 2 if cohort == "pair_interaction" else 3
        def sample_effect():
            return sum(1 << int(i) for i in rng.choice(4, n, replace=False))
        x, y = sample_effect(), sample_effect()
        effects = (x, x, y)
    else:
        x = DISJOINT_PAIRS[int(rng.integers(3))]
        y = DISJOINT_PAIRS[int(rng.integers(3))]
        effects = (x, x, y)
    return ((a, bias_a, effects[0]),
            (a, 1 - bias_a, effects[1]),
            (c, bias_c, effects[2]))


def candidate_catalog():
    """Exact set of possible 16-cue functions, without a target rule ID."""
    signatures = set()
    for parity_mask in AFFINE_MASKS:
        for bias in (0, 1):
            for effect in INTERACTIONS:
                rule = (parity_mask, bias, effect)
                signatures.add(tuple(wanted_action(c, rule) for c in CUES))
    # Unique fully observed functions rather than duplicated syntax.
    return np.array(sorted(signatures), dtype=np.uint8)


CATALOG = candidate_catalog()


def prior_has_truth(rule):
    truth = np.array([wanted_action(c, rule) for c in CUES], dtype=np.uint8)
    return bool(np.any(np.all(CATALOG == truth[None, :], axis=1)))


def posterior_indices(records):
    """Filter researcher-supplied function possibilities by actual choices and rewards.

    This routine knows only support triples. The exact true rule and
    any exam outcomes are NEVER arguments.
    """
    valid = np.ones(len(CATALOG), dtype=bool)
    for cue, action, success in records:
        implied_label = action ^ (1 - success)
        valid &= CATALOG[:, cue] == implied_label
    return np.flatnonzero(valid)


def _min_hamming_distance(cue, visited):
    return min((cue ^ other).bit_count() for other in visited)


def choose_next_cue(records, policy: str, tie_order):
    """No target, oracle, heldout answers or environment access."""
    if policy not in POLICIES:
        raise ValueError("Invalid exploration arm")
    if len(records) < START_CUES or len(records) >= SUPPORT_BUDGET:
        raise ValueError("First two shared observations; exactly eight total")
    visited = {r[0] for r in records}
    if len(visited) != len(records):
        raise AssertionError("Duplicate cue already observed")
    tie_priority = {cue: rank for rank, cue in enumerate(tie_order)}
    remaining = [c for c in tie_order if c not in visited]
    if policy == "random_cues":
        return remaining[0], 0
    if policy == "coverage_only":
        return max(
            remaining,
            key=lambda c: (_min_hamming_distance(c, visited),
                           -tie_priority[c])
        ), 0
    survivor = posterior_indices(records)
    resets = 0
    if len(survivor) == 0:
        # Only false-belief control can produce a contradiction.
        if policy != "scrambled_belief_active":
            raise AssertionError("Actual truth removed from candidate grammar")
        survivor = np.arange(len(CATALOG))
        resets = 1
    probs = CATALOG[survivor, :].mean(axis=0)
    # Exact binary entropy is monotonic with min(p,1-p);
    # maximize prior model disagreement using support only.
    return max(
        remaining,
        key=lambda c: (
            min(float(probs[c]), float(1. - probs[c])),
            _min_hamming_distance(c, visited),
            -tie_priority[c],
        ),
    ), resets


def _seed_plan(task_seed: int, ctx: int):
    """Same initial cues + the same random action at any cue for all arms."""
    rng = np.random.default_rng(task_seed * 120011 + ctx * 479 + 53)
    order = tuple(int(c) for c in rng.permutation(CUE_COUNT))
    actions = tuple(int(a) for a in rng.integers(2, size=CUE_COUNT))
    false_rewards = tuple(int(a) for a in rng.integers(2, size=CUE_COUNT))
    return order, actions, false_rewards


def acquire_own_support(task_seed: int, ctx: int, rule, policy: str):
    """Environment only supplies the reward AFTER the policy selected cue."""
    if policy not in POLICIES:
        raise ValueError("Invalid policy")
    order, actions, false_rewards = _seed_plan(task_seed, ctx)
    authentic = []
    imagined = []
    log = []
    reset_count = 0
    for step in range(SUPPORT_BUDGET):
        if step < START_CUES:
            cue, reset = order[step], 0
        else:
            planner_records = (imagined if policy == "scrambled_belief_active"
                               else authentic)
            cue, reset = choose_next_cue(planner_records, policy, order)
        if any(cue == r[0] for r in authentic):
            raise AssertionError("Revisiting a cue is prohibited")
        action = actions[cue]
        # Only environment-side transition sees rule truth.
        observed_reward = int(action == wanted_action(cue, rule))
        false_reward = false_rewards[cue]
        authentic.append((cue, action, observed_reward))
        imagined.append((cue, action, false_reward))
        reset_count += reset
        survivors = posterior_indices(authentic)
        if len(survivors) == 0:
            raise AssertionError("Reality must be possible under human grammar")
        log.append({
            "step": step + 1,
            "requested_cue": cue,
            "actual_chosen_action": action,
            "actual_observed_reward": observed_reward,
            "planner_feedback": (false_reward if policy == "scrambled_belief_active"
                                 else observed_reward),
            "true_observation_posterior_count": int(len(survivors)),
            "planner_belief_reset": bool(reset),
        })
    exam = tuple(c for c in CUES if c not in {r[0] for r in authentic})
    if len(exam) != 8 or len(authentic) != 8:
        raise AssertionError("Eight observed cues and disjoint eight-cue exam")
    return tuple(authentic), exam, log, reset_count


def theory_diagnostics(records, rule):
    """Perfectly grammar-informed posterior, used for OFFLINE diagnosis."""
    surviving = posterior_indices(records)
    if len(surviving) == 0:
        raise AssertionError("Family catalog excludes the true synthetic task")
    all_truth = tuple(wanted_action(c, rule) for c in CUES)
    present = any(all(CATALOG[i, c] == all_truth[c] for c in CUES)
                  for i in surviving)
    predictions = tuple(
        int(CATALOG[surviving, c].mean() >= .5) for c in CUES
    )
    return {
        "posterior_distinct_truth_tables": int(len(surviving)),
        "true_function_retained": present,
        "exact_identifiable": int(len(surviving) == 1),
        "grammar_bayes_all16_accuracy": sum(
            p == t for p, t in zip(predictions, all_truth)
        ) / 16,
    }


def one_context(task_seed: int, ctx: int, cohort: str, optimizer_seed: int):
    """Only this offline evaluation boundary sees the hidden true rule."""
    if optimizer_seed not in OPTIMIZER_SEEDS:
        raise ValueError("Unregistered optimizer seed")
    rule = fresh_rule_for(task_seed, cohort)[ctx]
    if not prior_has_truth(rule):
        raise AssertionError("Preregistered human family catalog incomplete")
    all_labels = tuple(wanted_action(c, rule) for c in CUES)
    arms = {}
    previous_start = None
    action_schedule = _seed_plan(task_seed, ctx)[1]
    for policy in POLICIES:
        records, exam, trace, resets = acquire_own_support(
            task_seed, ctx, rule, policy
        )
        if previous_start is None:
            previous_start = tuple(r[0] for r in records[:2])
        if tuple(r[0] for r in records[:2]) != previous_start:
            raise AssertionError("Shared initial two cues violated")
        fitting_seed = optimizer_seed * 1000003 + task_seed * 79 + ctx * 167
        model = DualSoftSynapses(fitting_seed, SGD_DIVERSITY).fit(records)
        all_prediction = model.predictions(CUES)
        theoretical = theory_diagnostics(records, rule)
        arms[policy] = {
            "observations": [list(x) for x in records],
            "stepwise_selection_trace": trace,
            "unseen_cues": list(exam),
            "no_support_exam_overlap": not bool(set(exam) & {r[0] for r in records}),
            "same_precommitted_binary_actions_for_all_arms": all(
                action_schedule[r[0]] == r[1] for r in records
            ),
            "all16_frozen_predictions": list(all_prediction),
            "all16_accuracy_including_seen_support": sum(
                int(p == y) for p, y in zip(all_prediction, all_labels)
            ) / 16.,
            "own_unseen8_accuracy_different_exam_set_per_arm": sum(
                int(all_prediction[c] == all_labels[c]) for c in exam
            ) / 8.,
            "support8_accuracy_not_frozen_exam": sum(
                int(all_prediction[cue] == all_labels[cue])
                for cue, _, _ in records
            ) / 8.,
            "theoretical_identifiability": theoretical,
            "false_planner_resets": resets,
            "frozen_float32_weights": model.frozen_code.tolist(),
            "optimizer_gradient_updates": model.gradient_updates,
            "frozen_code_bytes": model.frozen_code.nbytes,
            "same_15_parameter_neural_fit_each_arm": True,
        }
    return {
        "task_seed": task_seed,
        "optimizer_seed": optimizer_seed,
        "context": ctx,
        "cohort": cohort,
        "report_only_true_rule": [
            rule[0], rule[1],
            list(rule[2]) if isinstance(rule[2], tuple) else rule[2]
        ],
        "report_only_true_all16_labels": list(all_labels),
        "shared_two_initial_cues": list(previous_start),
        "precommitted_action_by_cue": list(action_schedule),
        "arms": arms,
    }


def paired_ci(rows, cohort, field, other, n=1000):
    """Bootstrap task seeds then optimizer replicates, contexts clustered."""
    tasks = COHORTS[cohort]
    lookup = {}
    for row in rows:
        if row["cohort"] == cohort:
            k = row["optimizer_seed"], row["task_seed"]
            lookup.setdefault(k, []).append(row)
    diffs = {
        (opt, seed): statistics.mean(
            r["arms"]["family_version_space_active"][field] -
            r["arms"][other][field]
            for r in lookup[opt, seed]
        ) for opt in OPTIMIZER_SEEDS for seed in tasks
    }
    observed = statistics.mean(diffs.values())
    r = random.Random(533091)
    draws = []
    for _ in range(n):
        sampled_optimizers = [r.choice(OPTIMIZER_SEEDS) for _ in OPTIMIZER_SEEDS]
        draws.append(statistics.mean(
            diffs[opt, r.choice(tasks)] for opt in sampled_optimizers
            for _ in tasks
        ))
    draws.sort()
    return {
        "difference": observed,
        "exploratory_hierarchical_boot95": [
            draws[int(n * .025)], draws[int(n * .975)-1]
        ]
    }


def run():
    raw = [
        one_context(seed, ctx, cohort, opt)
        for opt in OPTIMIZER_SEEDS
        for cohort, seeds in COHORTS.items()
        for seed in seeds
        for ctx in range(3)
    ]
    summary = {}
    for cohort in COHORTS:
        group = [r for r in raw if r["cohort"] == cohort]
        summary[cohort] = {
            "context_optimizer_rows": len(group),
            "arms": {},
            "paired_accuracies_active_vs_controls": {
                comparator: {
                    metric: paired_ci(raw, cohort, metric, comparator)
                    for metric in (
                        "all16_accuracy_including_seen_support",
                        "own_unseen8_accuracy_different_exam_set_per_arm"
                    )
                }
                for comparator in POLICIES if comparator != "family_version_space_active"
            },
        }
        for p in POLICIES:
            cases = [r["arms"][p] for r in group]
            summary[cohort]["arms"][p] = {
                "full16_neural_accuracy": statistics.mean(
                    a["all16_accuracy_including_seen_support"] for a in cases
                ),
                "heldout8_neural_accuracy": statistics.mean(
                    a["own_unseen8_accuracy_different_exam_set_per_arm"] for a in cases
                ),
                "support8_neural_accuracy": statistics.mean(
                    a["support8_accuracy_not_frozen_exam"] for a in cases
                ),
                "mean_posterior_count_after8": statistics.mean(
                    a["theoretical_identifiability"]["posterior_distinct_truth_tables"]
                    for a in cases
                ),
                "true_function_exactly_identifiable_rate": statistics.mean(
                    a["theoretical_identifiability"]["exact_identifiable"]
                    for a in cases
                ),
                "grammar_bayes_all16_accuracy": statistics.mean(
                    a["theoretical_identifiability"]["grammar_bayes_all16_accuracy"]
                    for a in cases
                ),
                "total_false_planner_resets": sum(
                    a["false_planner_resets"] for a in cases
                ),
            }
    two = summary["two_disjoint_pairs"]["arms"]
    active, random_c, coverage, scrambled = (
        two["family_version_space_active"],
        two["random_cues"],
        two["coverage_only"],
        two["scrambled_belief_active"],
    )
    checks = {
        "two_pair_posterior_count_at_least20pct_lower_than_random":
            active["mean_posterior_count_after8"] <=
            .8 * random_c["mean_posterior_count_after8"],
        "two_pair_identifiable_improvement_at_least10pp":
            active["true_function_exactly_identifiable_rate"] -
            random_c["true_function_exactly_identifiable_rate"] >=
            CHECK_THRESHOLDS["identifiable_gain"],
        "two_pair_full16_neural_gain_vs_random_at_least5pp":
            active["full16_neural_accuracy"] -
            random_c["full16_neural_accuracy"] >=
            CHECK_THRESHOLDS["two_pair_full16_gain_vs_random"],
        "two_pair_unseen8_neural_gain_vs_random_at_least5pp":
            active["heldout8_neural_accuracy"] -
            random_c["heldout8_neural_accuracy"] >=
            CHECK_THRESHOLDS["two_pair_heldout8_gain_vs_random"],
        "two_pair_full16_neural_gain_vs_scrambled_planner_at_least5pp":
            active["full16_neural_accuracy"] -
            scrambled["full16_neural_accuracy"] >=
            CHECK_THRESHOLDS["two_pair_full16_gain_vs_scrambled_planner"],
        "two_pair_full16_neural_gain_vs_coverage_at_least3pp":
            active["full16_neural_accuracy"] -
            coverage["full16_neural_accuracy"] >=
            CHECK_THRESHOLDS["two_pair_full16_gain_vs_coverage"],
        "two_pair_full16_accuracy_at_least60pct":
            active["full16_neural_accuracy"] >=
            CHECK_THRESHOLDS["two_pair_full16_accuracy"],
        "all_true_rules_remain_in_reality_posterior":
            all(a["theoretical_identifiability"]["true_function_retained"]
                for row in raw for a in row["arms"].values()),
        "same_initial_two_and_eight_distinct_genuine_observations":
            all(
                tuple(a["observations"][i][0] for i in range(2)) ==
                tuple(row["shared_two_initial_cues"])
                and len({o[0] for o in a["observations"]}) == 8
                and len(a["observations"]) == 8
                and a["same_precommitted_binary_actions_for_all_arms"]
                for row in raw for a in row["arms"].values()
            ),
        "raw_complete_eight_unseen_no_leak_and_all_controls":
            all(
                set(row["arms"]) == set(POLICIES)
                and all(
                    a["no_support_exam_overlap"]
                    and len(a["unseen_cues"]) == 8
                    and len(a["stepwise_selection_trace"]) == 8
                    and len(a["all16_frozen_predictions"]) == 16
                    for a in row["arms"].values()
                ) for row in raw
            ),
        "same_real_15_float32_codes_and_evolution_stop":
            all(
                a["frozen_code_bytes"] == 60
                and len(a["frozen_float32_weights"]) == 15
                and a["optimizer_gradient_updates"] == 8 * 300
                for row in raw for a in row["arms"].values()
            ),
    }
    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "status": "M5.33 active cue selection: fresh synthetic DEV seeds, same known rule family",
        "metadata": {
            "cohorts": {k: list(v) for k,v in COHORTS.items()},
            "optimizer_seeds": list(OPTIMIZER_SEEDS),
            "total_task_context_optimizations_per_arm": len(raw),
            "distinct_environmental_contexts": len(raw)//len(OPTIMIZER_SEEDS),
            "possible_truth_tables_human_grammar": len(CATALOG),
            "initial_random_shared_cues": START_CUES,
            "sequentially_selected_cues": SUPPORT_BUDGET - START_CUES,
            "total_own_binary_rewards": SUPPORT_BUDGET,
            "all16_primary_includes_training_cues": True,
            "heldout8_complements_differ_per_policy": True,
            "original_M531_family_reused_but_seed_list_new": True,
            "neural_code_float32_bits_per_context": 480,
            "matched_neural_updates_but_not_planner_compute": True,
            "catalog_enumeration_is_human_familiy_privileged": True,
            "neural_action_policy_not_learned": True,
        },
        "summary": summary,
        "screening_gates": checks,
        "local_active_cue_screen_pass": all(checks.values()),
        "learned_general_curiosity_policy": False,
        "autonomously_invented_neural_operator": False,
        "shared_meta_plastic_controller_learned": False,
        "independent_task_family_replication_complete": False,
        "continual_memory_retention_verified": False,
        "evolution_authorized": False,
        "advance_to_evolution": False,
        "sealed_M5_6_ecology_accessed": False,
        "raw_rows_sha256": digest,
        "raw_rows": raw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/m5-33/active-cue-development.json"))
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    print("M5_33_ACTIVE_CUE " + json.dumps({
        k: result[k] for k in (
            "status", "metadata", "summary", "screening_gates",
            "local_active_cue_screen_pass", "raw_rows_sha256"
        )
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
