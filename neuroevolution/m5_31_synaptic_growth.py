"""M5.31: reward-conditioned soft synapses, consolidated as one sparse neuron.

DEVELOPMENT ONLY. A soft neuron with an explicitly researcher-designed
conjunction activation learns four incoming synapse gates (and a scalar
strength) by reward-only gradient descent. The learner then grows/prunes
a discrete incoming-edge mask from soft gates using SUPPORT DATA ONLY,
refitting five float32 product module weights. This is NOT discovery of
the conjunction operator, a shared meta-learner, or evolutionary growth.

The eight binary support actions are randomized script choices (not
autonomous active-exploration decisions); their observed reward is real.
All eight heldout cue outcomes are generated only for offline grading.
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
    ModularProductLearner, X_BITS, support_split,
    sign_targets_from_observations,
)
from neuroevolution.m5_30_operator_selection import fit_bank as menu_fit_bank, predict as menu_predict

MODEL_SEEDS = (881, 883, 887)
COHORTS = {
    "plain_parity": tuple(range(11100, 11112)),
    "pair_interaction": tuple(range(11200, 11216)),
    "triple_interaction": tuple(range(11300, 11316)),
    "two_disjoint_pairs_outside_one_neuron": tuple(range(11400, 11408)),
}
AFFINE_MASKS = (5, 9, 11, 14)
DISJOINT_PAIRS = ((3, 12), (5, 10), (9, 6))
ARMS = (
    "learned_synapse_growth",
    "fixed_product",
    "scrambled_feedback_growth",
    "random_structural_growth",
    "m5_30_fixed_operator_menu",
    "human_privileged_topology",
    "continuous_soft_unpruned",
)
SOFT_RESTARTS = 16
SOFT_STEPS = 300
SOFT_LR_W = .30
SOFT_LR_ALPHA = .08
SOFT_LR_GATES = .09
ALPHA_SUPPORT_PENALTY = .01
EDGE_THRESHOLD = .50
ALPHA_THRESHOLD = .50
EDGE_COMPLEXITY_PENALTY = .012
TOP_SOFT_CANDIDATES = 8
PERSISTENT_BITS_PER_CONTEXT = 5 * 32 + 4
X = X_BITS.astype(np.float64)
GATES = {
    "pair_unseen_at_least70": .70,
    "triple_unseen_at_least70": .70,
    "pair_gain_over_fixed_at_least8pp": .08,
    "triple_gain_over_M530_menu_at_least5pp": .05,
    "pair_gain_over_corruption_at_least15pp": .15,
    "pair_growth_rate_at_least20pct": .20,
}


def is_conjunction(cue: int, mask: int) -> int:
    if mask == 0:
        return 0
    if not 0 <= mask <= 15 or cue not in range(16):
        raise ValueError("Invalid sensory bitmask or cue")
    return int((cue & mask) == mask)


def rule_for(seed: int, cohort: str):
    if cohort not in COHORTS or type(seed) is not int or seed not in COHORTS[cohort]:
        raise ValueError("Unregistered M5.31 development seed/cohort")
    rng = np.random.default_rng(seed * 1079 + 13)
    amask = int(rng.choice(AFFINE_MASKS))
    cmask = int(rng.choice([m for m in AFFINE_MASKS if m != amask]))
    abias, cbias = int(rng.integers(2)), int(rng.integers(2))
    if cohort == "plain_parity":
        effects = (0, 0, 0)
    elif cohort in ("pair_interaction", "triple_interaction"):
        n = 2 if cohort == "pair_interaction" else 3
        def fresh_mask():
            return sum(1 << int(b) for b in rng.choice(4, n, replace=False))
        first, third = fresh_mask(), fresh_mask()
        effects = (first, first, third)
    else:
        first = DISJOINT_PAIRS[int(rng.integers(len(DISJOINT_PAIRS)))]
        third = DISJOINT_PAIRS[int(rng.integers(len(DISJOINT_PAIRS)))]
        effects = (first, first, third)
    return ((amask, abias, effects[0]),
            (amask, 1 - abias, effects[1]),
            (cmask, cbias, effects[2]))


def wanted_action(cue: int, rule) -> int:
    amask, bias, effect = rule
    out = (int(amask & cue).bit_count() & 1) ^ bias
    if isinstance(effect, tuple):
        out ^= is_conjunction(cue, effect[0]) ^ is_conjunction(cue, effect[1])
    else:
        out ^= is_conjunction(cue, effect)
    return out


def own_experiences(seed: int, cohort: str, ctx: int, rule):
    if ctx not in (0, 1, 2):
        raise ValueError("Unknown context")
    rng = np.random.default_rng(seed * 37781 + ctx * 197 + 17)
    seen, query = support_split(rng)
    records = []
    for cue in seen:
        cue = int(cue)
        chosen = int(rng.integers(2))
        reward = int(chosen == wanted_action(cue, rule))
        records.append((cue, chosen, reward))
    if len(records) != 8 or len(query) != 8:
        raise AssertionError("Bad sample count")
    if set(cue for cue, _, _ in records) & set(query):
        raise AssertionError("Exam cue leaked into support")
    return tuple(records), tuple(int(q) for q in query)


def corrupt_records(records, seed: int):
    rng = np.random.default_rng(seed * 991 + 5)
    return tuple((cue, action, int(rng.integers(2)))
                 for cue, action, _ in records)


def corrected_records(records, mask: int):
    return tuple((cue, action ^ is_conjunction(cue, mask), reward)
                 for cue, action, reward in records)


def fit_product(records, seed: int, mask: int):
    model = ModularProductLearner(seed, "product")
    model.fit(corrected_records(records, mask))
    if model.weights.dtype != np.float32 or model.weights.nbytes != 20:
        raise AssertionError("Five final weights must really be float32")
    return model


def evaluate_hard(model, mask: int, cues):
    return tuple(int(p) ^ is_conjunction(cue, mask)
                 for p, cue in zip(model.predict(cues), cues))


class SoftSynapticNeuron:
    """Unconsolidated potential synapses; no supplied target interaction."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-.9, .9, (SOFT_RESTARTS, 5))
        self.w[0] = .5
        self.g = rng.uniform(.2, .95, (SOFT_RESTARTS, 4))
        self.g[0] = .5
        self.alpha = rng.uniform(.4, .98, SOFT_RESTARTS)
        self.alpha[0] = .8
        self.soft_support_loss = None
        self.rank = None
        self.best_index = None
        self.gradient_steps = 0

    def forward(self, w, alpha, gates, cues):
        x = X[list(cues)]
        terms = np.where(x[None, :, :].astype(bool),
                         w[:, None, 1:], 1.)
        base = w[:, 0, None] * np.prod(terms, axis=2)
        z = 1. - gates[:, None, :] * (1. - x[None, :, :])
        h = np.prod(z, axis=2)
        return base * (1. - 2. * alpha[:, None] * h)

    def fit(self, records):
        if len(records) != 8 or len(set(r[0] for r in records)) != 8:
            raise ValueError("Exactly eight distinct own support experiences required")
        cues = tuple(c for c, _, _ in records)
        x = X[list(cues)]
        targets = sign_targets_from_observations(records)
        w, gates, alpha = self.w, self.g, self.alpha
        for _ in range(SOFT_STEPS):
            active = np.where(x[None, :, :].astype(bool),
                              w[:, None, 1:], 1.)
            base_factors = np.prod(active, axis=2)
            base = w[:, 0, None] * base_factors
            z = 1. - gates[:, None, :] * (1. - x[None, :, :])
            h = np.prod(z, axis=2)
            factor = 1. - 2. * alpha[:, None] * h
            prediction = base * factor
            delta = 2. / len(cues) * (prediction - targets[None, :])
            gradw = np.empty_like(w)
            gradw[:, 0] = np.sum(delta * base_factors * factor, axis=1)
            for k in range(4):
                others = np.prod(np.delete(active, k, axis=2), axis=2)
                gradw[:, k + 1] = np.sum(
                    delta * w[:, 0, None] * x[None, :, k] *
                    others * factor, axis=1)
            gradalpha = np.sum(delta * base * (-2. * h), axis=1)
            gradg = np.empty_like(gates)
            for k in range(4):
                others = np.prod(np.delete(z, k, axis=2), axis=2)
                gradg[:, k] = np.sum(
                    delta * base * 2. * alpha[:, None] *
                    (1. - x[None, :, k]) * others, axis=1)
            w = np.clip(w - SOFT_LR_W * gradw, -1., 1.)
            alpha = np.clip(alpha - SOFT_LR_ALPHA * gradalpha, 0., 1.)
            gates = np.clip(gates - SOFT_LR_GATES * gradg, 0., 1.)
        self.w, self.g, self.alpha = w, gates, alpha
        predicted = self.forward(w, alpha, gates, cues)
        self.soft_support_loss = np.mean(
            (predicted - targets[None, :]) ** 2, axis=1
        ) + ALPHA_SUPPORT_PENALTY * alpha
        self.rank = tuple(int(x) for x in np.argsort(self.soft_support_loss))
        self.best_index = self.rank[0]
        self.gradient_steps = SOFT_RESTARTS * SOFT_STEPS
        return self

    def hard_mask(self, restart: int):
        if self.alpha[restart] < ALPHA_THRESHOLD:
            return 0
        mask = sum(1 << k for k in range(4)
                   if self.g[restart, k] >= EDGE_THRESHOLD)
        return mask if mask.bit_count() >= 2 else 0

    def frozen_soft_predict(self, cues):
        j = self.best_index
        if j is None:
            raise RuntimeError("Fit before frozen evaluation")
        # No query update or outcome is accepted.
        y = self.forward(self.w[j:j + 1], self.alpha[j:j + 1],
                         self.g[j:j + 1], cues)[0]
        return tuple(int(s < 0.) for s in y)


def consolidate(records, seed: int):
    """After reward-only soft optimization, stabilize/prune sparse edges.

    No rule, topology oracle, exam cue, query reward or context label
    is an argument. Candidate comparison is SUPPORT-ONLY.
    """
    soft = SoftSynapticNeuron(seed).fit(records)
    suggested_masks = sorted(set(
        soft.hard_mask(j) for j in soft.rank[:TOP_SOFT_CANDIDATES]
    ) | {0})
    choices = {}
    for mask in suggested_masks:
        model = fit_product(records, seed, mask)
        score = model.support_mse + EDGE_COMPLEXITY_PENALTY * mask.bit_count()
        choices[mask] = (score, model)
    selected = min(choices, key=lambda mask: (choices[mask][0], mask))
    return {
        "mask": selected,
        "model": choices[selected][1],
        "soft": soft,
        "candidate_masks": suggested_masks,
        "selection_support_costs": {str(k): v[0] for k, v in choices.items()},
        "best_soft_alpha": float(soft.alpha[soft.best_index]),
        "best_soft_gates": soft.g[soft.best_index].tolist(),
        "new_synapses": selected.bit_count(),
        "pruned_potential_synapses": 4 - selected.bit_count(),
    }


def grade(predictions, cues, rule):
    return statistics.mean(
        int(pred == wanted_action(cue, rule))
        for pred, cue in zip(predictions, cues)
    )


def one_trial(model_seed: int, seed: int, cohort: str):
    # This is the only simulator scope with true rules; true topologies
    # never enter consolidate(), fit_product() or SoftSynapticNeuron.fit().
    rules = rule_for(seed, cohort)
    rng = random.Random(model_seed * 191 + seed)
    accuracies = {arm: [] for arm in ARMS}
    learned_mask = []
    selected_costs = []
    soft_alpha = []
    soft_gates = []
    pruned = []
    true_effects = []
    for ctx, rule in enumerate(rules):
        records, exam = own_experiences(seed, cohort, ctx, rule)
        s = model_seed * 1000019 + seed * 31 + ctx * 97
        growth = consolidate(records, s)
        fixed = fit_product(records, s, 0)
        scrambled = consolidate(corrupt_records(records, s), s)
        random_mask = rng.choice([mask for mask in range(16)
                                  if 2 <= mask.bit_count() <= 3])
        random_model = fit_product(records, s, random_mask)
        menu, menu_choice, _ = menu_fit_bank(records, s)
        privileged_mask = rule[2] if isinstance(rule[2], int) else 0
        privileged = fit_product(records, s, privileged_mask)
        outputs = {
            "learned_synapse_growth": evaluate_hard(
                growth["model"], growth["mask"], exam),
            "fixed_product": evaluate_hard(fixed, 0, exam),
            "scrambled_feedback_growth": evaluate_hard(
                scrambled["model"], scrambled["mask"], exam),
            "random_structural_growth": evaluate_hard(
                random_model, random_mask, exam),
            "m5_30_fixed_operator_menu": menu_predict(
                menu[menu_choice], menu_choice, exam),
            "human_privileged_topology": evaluate_hard(
                privileged, privileged_mask, exam),
            "continuous_soft_unpruned": growth["soft"].frozen_soft_predict(exam),
        }
        for arm in ARMS:
            accuracies[arm].append(grade(outputs[arm], exam, rule))
        learned_mask.append(growth["mask"])
        selected_costs.append(growth["selection_support_costs"])
        soft_alpha.append(growth["best_soft_alpha"])
        soft_gates.append(growth["best_soft_gates"])
        pruned.append(growth["pruned_potential_synapses"])
        true_effects.append(list(rule[2]) if isinstance(rule[2], tuple)
                            else rule[2])
    return {
        "model_seed": model_seed, "seed": seed, "cohort": cohort,
        "unseen_accuracy": {a: statistics.mean(v) for a, v in accuracies.items()},
        "learned_synapse_masks": learned_mask,
        "report_only_true_interactions": true_effects,
        "new_synapse_counts": [m.bit_count() for m in learned_mask],
        "pruned_potential_edge_counts": pruned,
        "best_soft_alpha": soft_alpha,
        "best_soft_gate_values": soft_gates,
        "support_selection_costs": selected_costs,
        "three_task_code_bits": 3 * PERSISTENT_BITS_PER_CONTEXT,
        "soft_workspace_and_restart_memory_excluded": True,
        "randomized_chosen_support_actions_not_learned_policy": True,
        "own_actual_support_rewards_unseen_to_exams": True,
        "candidate_never_received_true_interaction": True,
        "soft_to_hard_synapses_researcher_engineered": True,
        "no_shared_meta_learning": True,
        "original_sealed_ecology_untouched": True,
    }


def paired_ci(rows, cohort: str, against: str, n=1000):
    lookup = {(r["model_seed"], r["seed"]): r
              for r in rows if r["cohort"] == cohort}
    seeds = COHORTS[cohort]
    diffs = {
        (m, s): (
            lookup[(m, s)]["unseen_accuracy"]["learned_synapse_growth"] -
            lookup[(m, s)]["unseen_accuracy"][against]
        ) for m in MODEL_SEEDS for s in seeds
    }
    rng = random.Random(573199)
    observed = statistics.mean(diffs.values())
    samples = []
    for _ in range(n):
        chosen_models = [rng.choice(MODEL_SEEDS) for _ in MODEL_SEEDS]
        samples.append(statistics.mean(
            diffs[(m, rng.choice(seeds))]
            for m in chosen_models for _ in seeds
        ))
    samples.sort()
    return {"difference": observed,
            "exploratory_boot95": [samples[int(.025*n)], samples[int(.975*n) - 1]]}


def run():
    raw = [one_trial(model_seed, seed, cohort)
           for model_seed in MODEL_SEEDS
           for cohort, seeds in COHORTS.items()
           for seed in seeds]
    summary = {}
    for cohort in COHORTS:
        subset = [r for r in raw if r["cohort"] == cohort]
        summary[cohort] = {
            "unseen_accuracy": {
                arm: statistics.mean(r["unseen_accuracy"][arm] for r in subset)
                for arm in ARMS
            },
            "growth_frequency": statistics.mean(
                int(m.bit_count() >= 2) for r in subset
                for m in r["learned_synapse_masks"]
            ),
            "mean_consolidated_edges": statistics.mean(
                m.bit_count() for r in subset
                for m in r["learned_synapse_masks"]
            ),
            "exact_interaction_mask_frequency": (
                statistics.mean(
                    int(m == t) for r in subset
                    for m, t in zip(r["learned_synapse_masks"],
                                    r["report_only_true_interactions"])
                ) if cohort != "two_disjoint_pairs_outside_one_neuron" else None
            ),
            "paired_bootstrap": {
                against: paired_ci(raw, cohort, against)
                for against in ARMS if against != "learned_synapse_growth"
            },
        }
    pair = summary["pair_interaction"]
    triple = summary["triple_interaction"]
    pair_acc = pair["unseen_accuracy"]
    triple_acc = triple["unseen_accuracy"]
    checks = {
        "pair_new_cue_accuracy_ge70": pair_acc["learned_synapse_growth"] >= GATES["pair_unseen_at_least70"],
        "triple_new_cue_accuracy_ge70": triple_acc["learned_synapse_growth"] >= GATES["triple_unseen_at_least70"],
        "pair_beats_fixed_product_ge8pp":
            pair_acc["learned_synapse_growth"] -
            pair_acc["fixed_product"] >= GATES["pair_gain_over_fixed_at_least8pp"],
        "triple_beats_m530_pair_menu_ge5pp":
            triple_acc["learned_synapse_growth"] -
            triple_acc["m5_30_fixed_operator_menu"] >= GATES["triple_gain_over_M530_menu_at_least5pp"],
        "pair_beats_scrambled_reward_growth_ge15pp":
            pair_acc["learned_synapse_growth"] -
            pair_acc["scrambled_feedback_growth"] >= GATES["pair_gain_over_corruption_at_least15pp"],
        "pair_synapses_consolidated_at_least20pct":
            pair["growth_frequency"] >= GATES["pair_growth_rate_at_least20pct"],
        "eight_support_cues_disjoint_from_eight_exam_cues":
            all(r["own_actual_support_rewards_unseen_to_exams"] for r in raw),
        "candidate_receives_no_true_interaction":
            all(r["candidate_never_received_true_interaction"] for r in raw),
        "persistent_A_B_C_codes_exactly_492bits":
            all(r["three_task_code_bits"] == 492 for r in raw),
        "outside_single_conjunction_stress_reported":
            len(summary["two_disjoint_pairs_outside_one_neuron"]["unseen_accuracy"]) == len(ARMS),
    }
    return {
        "status": "M5.31 fresh-development own-reward structural-synapse-consolidation",
        "metadata": {
            "cohorts": {k: list(v) for k, v in COHORTS.items()},
            "optimizer_initializations": list(MODEL_SEEDS),
            "soft_restarts": SOFT_RESTARTS,
            "soft_training_steps_per_restart": SOFT_STEPS,
            "candidate_projected_soft_restarts": TOP_SOFT_CANDIDATES,
            "edge_threshold": EDGE_THRESHOLD,
            "alpha_threshold": ALPHA_THRESHOLD,
            "l0_edge_penalty": EDGE_COMPLEXITY_PENALTY,
            "support_per_context": 8, "unseen_exam_per_context": 8,
            "observed_actions_random_not_exploration_policy": True,
            "human_supplied_conjunction_node_primitive": True,
            "cross_task_plasticity_not_trained": True,
            "persistent_bits_three_contexts": 492,
            "dev_pilot_seeds_excluded": True,
        },
        "summary": summary,
        "screening_gates": checks,
        "local_structural_plasticity_screen_pass": all(checks.values()),
        "autonomously_invented_neural_operator": False,
        "shared_meta_plastic_controller_learned": False,
        "independent_task_family_replication_complete": False,
        "evolution_authorized": False,
        "advance_to_evolution": False,
        "original_sealed_M5_6_ecology_accessed": False,
        "raw_rows_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "raw_rows": raw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=Path,
        default=Path("results/m5-31/synaptic-growth-development.json")
    )
    args = ap.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    print("M5_31_SYNAPTIC_GROWTH " + json.dumps({
        k: result[k] for k in (
            "status", "summary", "screening_gates",
            "local_structural_plasticity_screen_pass", "raw_rows_sha256"
        )
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
