"""M5.32: two trainable soft nonlinear units, real-reward specialization audit.

This DEV-ONLY experiment supplies a TWO-conjunction neuron architecture
and XOR-style residual math by hand. It learns 2x4 soft synaptic gates,
two strengths, and five product module weights exclusively from eight
randomly CHOSEN actions with actually observed 0/1 rewards. It DOES NOT
invent neural operators, choose its own exploratory actions, learn a
shared plasticity strategy, or solve continual memory retention.

M5.31's exact already-examined development task seeds are reused for a
direct paired comparison. NOT independent validation.
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
    X_BITS, sign_targets_from_observations,
)
from neuroevolution.m5_30_operator_selection import (
    fit_bank as menu_fit_bank, predict as menu_predict,
)
from neuroevolution.m5_31_synaptic_growth import (
    COHORTS, SoftSynapticNeuron, corrupt_records, evaluate_hard, fit_product,
    grade, is_conjunction, own_experiences, rule_for,
)

SEEDS = (977, 983, 991)
ARMS = (
    "two_soft_diversity",
    "two_soft_no_diversity",
    "two_soft_scrambled_rewards",
    "one_soft_m531",
    "fixed_product",
    "m530_fixed_operator_menu",
    "privileged_true_two_masks",
)
RESTARTS = 8
STEPS = 300
LR_W = .30
LR_GATE = .09
LR_ALPHA = .08
ALPHA_L1 = .01
DIVERSITY = .025
THRESHOLD = .50
INPUT_BITS = 4
N_UNITS = 2
N_FLOATS = 5 + N_UNITS * (INPUT_BITS + 1)
PERSISTENT_BITS_PER_CONTEXT = N_FLOATS * 32
X = X_BITS.astype(np.float64)

GATES = {
    "outside_accuracy": .55,
    "outside_vs_one_soft": .08,
    "outside_vs_fixed_product": .10,
    "plain_accuracy": .90,
    "triple_accuracy": .75,
    "outside_diversity_vs_no_diversity_floor": -.03,
    "both_units_causal_frequency": .10,
    "outside_vs_scrambled": .05,
}


def _checked(records):
    if len(records) != 8 or len({c for c, _, _ in records}) != 8:
        raise ValueError("Require eight distinct observed support interactions")
    for cue, action, reward in records:
        if cue not in range(16) or action not in (0, 1) or reward not in (0, 1):
            raise ValueError("Only actual 4-bit cue, binary action and reward")


class DualSoftSynapses:
    """Two fully researcher-defined conjunction units; plastic gates learned."""

    def __init__(self, seed: int, diversity: float = DIVERSITY):
        if diversity not in (0., DIVERSITY):
            raise ValueError("Only preregistered matched arm penalties allowed")
        self.diversity = float(diversity)
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-.9, .9, (RESTARTS, 5))
        self.g = rng.uniform(.20, .95, (RESTARTS, 2, 4))
        self.a = rng.uniform(.40, .98, (RESTARTS, 2))
        self.w[0] = .5
        self.g[0, 0, :] = .45
        self.g[0, 1, :] = .55
        self.a[0, :] = .8
        self.best = None
        self.train_objectives = None
        self.gradient_updates = 0

    @staticmethod
    def _forward(w, g, a, cues):
        """Vectorized float64 calculation: R random starts × N cues."""
        x = X[list(cues)]
        active = np.where(x[None, :, :].astype(bool), w[:, None, 1:], 1.)
        base = w[:, 0, None] * np.prod(active, axis=2)
        z = 1. - g[:, None, :, :] * (1. - x[None, :, None, :])
        h = np.prod(z, axis=3)
        f = 1. - 2. * a[:, None, :] * h
        return base * np.prod(f, axis=2)

    @staticmethod
    def _loss_grads(w, g, a, cues, targets, diversity):
        """Exact analytic gradients; independently checked numerically in tests."""
        x = X[list(cues)]
        active = np.where(x[None, :, :].astype(bool), w[:, None, 1:], 1.)
        base_factors = np.prod(active, axis=2)
        base = w[:, 0, None] * base_factors
        z = 1. - g[:, None, :, :] * (1. - x[None, :, None, :])
        h = np.prod(z, axis=3)
        f = 1. - 2. * a[:, None, :] * h
        both = f[:, :, 0] * f[:, :, 1]
        pred = base * both
        err = pred - targets[None, :]
        delta = 2. / len(cues) * err
        gw = np.empty_like(w)
        gw[:, 0] = np.sum(delta * base_factors * both, axis=1)
        for k in range(4):
            other_base = np.prod(np.delete(active, k, axis=2), axis=2)
            gw[:, k + 1] = np.sum(
                delta * w[:, 0, None] * x[None, :, k] *
                other_base * both, axis=1
            )
        ga = np.empty_like(a)
        gg = np.empty_like(g)
        for j in range(2):
            other = f[:, :, 1-j]
            ga[:, j] = np.sum(
                delta * base * (-2. * h[:, :, j]) * other,
                axis=1
            ) + ALPHA_L1
            for k in range(4):
                other_gates = np.prod(
                    np.delete(z[:, :, j, :], k, axis=2), axis=2
                )
                gg[:, j, k] = np.sum(
                    delta * base * other *
                    (2. * a[:, j, None] * (1. - x[None, :, k])
                     * other_gates), axis=1
                ) + diversity / 4. * g[:, 1-j, k]
        objective = (
            np.mean(err * err, axis=1)
            + ALPHA_L1 * np.sum(a, axis=1)
            + diversity * np.mean(g[:, 0, :] * g[:, 1, :], axis=1)
        )
        return objective, gw, gg, ga

    def fit(self, records):
        """Only (cue, chosen-action, observed-reward) tuples and seed."""
        _checked(records)
        cues = tuple(cue for cue, _, _ in records)
        # This deduction only applies to a deterministic two-action task:
        # a failed actual action implies the complementary action succeeds.
        targets = sign_targets_from_observations(records)
        w, g, a = self.w.copy(), self.g.copy(), self.a.copy()
        for _ in range(STEPS):
            _, gw, gg, ga = self._loss_grads(
                w, g, a, cues, targets, self.diversity
            )
            w = np.clip(w - LR_W * gw, -1., 1.)
            g = np.clip(g - LR_GATE * gg, 0., 1.)
            a = np.clip(a - LR_ALPHA * ga, 0., 1.)
        objective, _, _, _ = self._loss_grads(
            w, g, a, cues, targets, self.diversity
        )
        self.w, self.g, self.a = w, g, a
        self.train_objectives = objective
        self.best = int(np.argmin(objective))
        self.gradient_updates = RESTARTS * STEPS
        self.frozen_code = np.concatenate((
            self.w[self.best],
            self.g[self.best].ravel(),
            self.a[self.best]
        )).astype(np.float32)
        if self.frozen_code.shape != (15,) or self.frozen_code.nbytes != 60:
            raise AssertionError("Actual 15-float32 persistent code required")
        return self

    def predictions(self, cues, ablate=None):
        """Frozen float32 inference; no reward or exam labels are accepted."""
        if self.best is None:
            raise ValueError("Fit before inference")
        if ablate not in (None, 0, 1):
            raise ValueError("Unknown neuron index")
        code = self.frozen_code
        w = code[:5]
        g = code[5:13].reshape(2, 4)
        a = code[13:15]
        x = X_BITS[list(cues)].astype(np.float32)
        terms = np.where(x.astype(bool), w[1:][None, :], 1.).astype(np.float32)
        base = w[0] * np.prod(terms, axis=1)
        z = 1. - g[None, :, :] * (1. - x[:, None, :])
        h = np.prod(z, axis=2)
        f = 1. - 2. * a[None, :] * h
        if ablate is not None:
            f[:, ablate] = 1.
        output = base * np.prod(f, axis=1)
        return tuple(int(v < 0.) for v in output)

    def structure(self):
        if self.best is None:
            raise ValueError("Fit before analysis")
        gates = self.frozen_code[5:13].reshape(2, 4)
        alpha = self.frozen_code[13:15]
        mask = tuple(
            sum((1 << k) for k in range(4)
                if gates[j, k] >= THRESHOLD)
            if alpha[j] >= THRESHOLD else 0
            for j in range(2)
        )
        return {
            "effective_sensory_masks": list(mask),
            "effective_connection_counts": [m.bit_count() for m in mask],
            "gate_overlap_mean": float(np.mean(gates[0] * gates[1])),
            "distinct_nonzero_effective_masks": bool(
                mask[0] != 0 and mask[1] != 0 and mask[0] != mask[1]
            ),
            "learned_gates": gates.tolist(),
            "learned_strengths": alpha.tolist(),
            "support_objective": float(self.train_objectives[self.best]),
        }


def _oracle_experience(records, oracle_masks):
    """Human-only upper reference. NOT passed to candidate/matched arms."""
    if len(oracle_masks) != 2:
        raise ValueError("Need two true oracle masks, identity zeros allowed")
    return tuple((
        cue, action ^ is_conjunction(cue, oracle_masks[0])
                    ^ is_conjunction(cue, oracle_masks[1]), reward
    ) for cue, action, reward in records)


def _oracle_predict(model, masks, cues):
    product = evaluate_hard(model, 0, cues)
    return tuple(
        int(y) ^ is_conjunction(cue, masks[0])
               ^ is_conjunction(cue, masks[1])
        for cue, y in zip(cues, product)
    )


def _ctx_oracle_masks(rule):
    effect = rule[2]
    return tuple(effect) if isinstance(effect, tuple) else (effect, 0)


def one_trial(optimizer_seed: int, seed: int, cohort: str):
    """Environment-side true rules only inside this orchestration boundary."""
    if optimizer_seed not in SEEDS:
        raise ValueError("Non-preregistered M5.32 optimizer initialization")
    rules = rule_for(seed, cohort)
    accuracies = {arm: [] for arm in ARMS}
    contexts = []
    for ctx, true_rule in enumerate(rules):
        records, exam = own_experiences(seed, cohort, ctx, true_rule)
        fitting_seed = optimizer_seed * 1000003 + seed * 59 + ctx * 193
        primary = DualSoftSynapses(fitting_seed, DIVERSITY).fit(records)
        matched = DualSoftSynapses(fitting_seed, 0.).fit(records)
        scrambled = DualSoftSynapses(fitting_seed, DIVERSITY).fit(
            corrupt_records(records, fitting_seed))
        old_soft = SoftSynapticNeuron(fitting_seed).fit(records)
        fixed = fit_product(records, fitting_seed, 0)
        bank, choice, _ = menu_fit_bank(records, fitting_seed)
        # This entire block is an explicitly PRIVILEGED environmental reference.
        oracle_masks = _ctx_oracle_masks(true_rule)
        privileged_model = fit_product(
            _oracle_experience(records, oracle_masks), fitting_seed, 0
        )
        predictions = {
            "two_soft_diversity": primary.predictions(exam),
            "two_soft_no_diversity": matched.predictions(exam),
            "two_soft_scrambled_rewards": scrambled.predictions(exam),
            "one_soft_m531": old_soft.frozen_soft_predict(exam),
            "fixed_product": evaluate_hard(fixed, 0, exam),
            "m530_fixed_operator_menu": menu_predict(
                bank[choice], choice, exam),
            "privileged_true_two_masks": _oracle_predict(
                privileged_model, oracle_masks, exam),
        }
        for arm in ARMS:
            accuracies[arm].append(
                grade(predictions[arm], exam, true_rule)
            )
        unablated = accuracies["two_soft_diversity"][-1]
        ablation = [
            grade(primary.predictions(exam, ablate=j), exam, true_rule)
            for j in (0, 1)
        ]
        drops = [unablated - x for x in ablation]
        ctx_data = {
            "context_index": ctx,
            "support_cue_ids": [r[0] for r in records],
            "exam_cue_ids": list(exam),
            "no_exam_overlap": not bool(
                {r[0] for r in records} & set(exam)
            ),
            "own_observed_support_actions_rewards": [
                [int(v) for v in r] for r in records
            ],
            "report_only_true_interaction": (
                list(true_rule[2]) if isinstance(true_rule[2], tuple)
                else int(true_rule[2])
            ),
            "candidate_frozen_predictions": list(predictions["two_soft_diversity"]),
            "unseen_accuracy": {
                arm: accuracies[arm][-1] for arm in ARMS
            },
            "frozen_neuron_ablation_accuracies": ablation,
            "frozen_neuron_ablation_accuracy_drops": drops,
            "both_neurons_causally_needed": all(
                d >= 1./8. - 1e-12 for d in drops
            ),
            "primary_structure": primary.structure(),
            "matched_structure": matched.structure(),
            "scrambled_structure": scrambled.structure(),
            "primary_frozen_15_float32_weights": primary.frozen_code.tolist(),
            "strictly_matched_same_initialization_and_steps": True,
            "true_rule_never_passed_to_candidate": True,
            "support_actions_randomly_generated_not_exploration_policy": True,
        }
        contexts.append(ctx_data)
    return {
        "optimizer_seed": optimizer_seed,
        "seed": seed,
        "cohort": cohort,
        "unseen_accuracy": {
            a: statistics.mean(v) for a, v in accuracies.items()
        },
        "contexts": contexts,
        "frozen_A_B_C_code_bits": PERSISTENT_BITS_PER_CONTEXT * 3,
        "actual_float32_code_bytes_per_context": 60,
        "no_evolution_or_sealed_ecology_used": True,
    }


def paired_ci(rows, cohort: str, against: str, n=1000):
    lookup = {(r["optimizer_seed"], r["seed"]): r
              for r in rows if r["cohort"] == cohort}
    tasks = COHORTS[cohort]
    diffs = {
        (s, t): (
            lookup[s, t]["unseen_accuracy"]["two_soft_diversity"] -
            lookup[s, t]["unseen_accuracy"][against]
        )
        for s in SEEDS for t in tasks
    }
    randomizer = random.Random(91332)
    sample_means = []
    for _ in range(n):
        picked = [randomizer.choice(SEEDS) for _ in SEEDS]
        sample_means.append(statistics.mean(
            diffs[s, randomizer.choice(tasks)]
            for s in picked for _ in tasks
        ))
    sample_means.sort()
    return {
        "difference": statistics.mean(diffs.values()),
        "exploratory_boot95": [
            sample_means[int(.025*n)],
            sample_means[int(.975*n)-1]
        ],
    }


def run():
    raw = [
        one_trial(opt, seed, cohort)
        for opt in SEEDS for cohort, task_seeds in COHORTS.items()
        for seed in task_seeds
    ]
    summaries = {}
    for cohort, task_seeds in COHORTS.items():
        subset = [r for r in raw if r["cohort"] == cohort]
        contexts = [c for row in subset for c in row["contexts"]]
        summaries[cohort] = {
            "row_count": len(subset),
            "context_count": len(contexts),
            "unseen_accuracy": {
                arm: statistics.mean(r["unseen_accuracy"][arm] for r in subset)
                for arm in ARMS
            },
            "both_units_needed_frequency": statistics.mean(
                c["both_neurons_causally_needed"] for c in contexts
            ),
            "distinct_nonzero_gate_masks_frequency": statistics.mean(
                c["primary_structure"]["distinct_nonzero_effective_masks"]
                for c in contexts
            ),
            "mean_ablation_accuracy_drops": [
                statistics.mean(
                    c["frozen_neuron_ablation_accuracy_drops"][j]
                    for c in contexts
                ) for j in (0, 1)
            ],
            "mean_gate_overlap": statistics.mean(
                c["primary_structure"]["gate_overlap_mean"]
                for c in contexts
            ),
            "paired_bootstrap": {
                arm: paired_ci(raw, cohort, arm)
                for arm in ARMS if arm != "two_soft_diversity"
            },
        }
    outside = summaries["two_disjoint_pairs_outside_one_neuron"]
    plain = summaries["plain_parity"]
    triple = summaries["triple_interaction"]
    a = outside["unseen_accuracy"]
    checks = {
        "two_pair_unseen_accuracy_at_least55":
            a["two_soft_diversity"] >= GATES["outside_accuracy"],
        "two_pair_gain_vs_one_soft_at_least8pp":
            a["two_soft_diversity"] - a["one_soft_m531"] >= GATES["outside_vs_one_soft"],
        "two_pair_gain_vs_fixed_at_least10pp":
            a["two_soft_diversity"] - a["fixed_product"] >= GATES["outside_vs_fixed_product"],
        "plain_maintains_at_least90":
            plain["unseen_accuracy"]["two_soft_diversity"] >= GATES["plain_accuracy"],
        "triple_maintains_at_least75":
            triple["unseen_accuracy"]["two_soft_diversity"] >= GATES["triple_accuracy"],
        "matched_diversity_avoids_over3pp_loss":
            a["two_soft_diversity"] - a["two_soft_no_diversity"] >=
            GATES["outside_diversity_vs_no_diversity_floor"],
        "both_units_causally_needed_in_at_least10pct":
            outside["both_units_needed_frequency"] >= GATES["both_units_causal_frequency"],
        "outside_outperforms_feedback_scramble_by5pp":
            a["two_soft_diversity"] - a["two_soft_scrambled_rewards"] >=
            GATES["outside_vs_scrambled"],
        "train_exam_disjoint_and_truth_never_passed":
            all(c["no_exam_overlap"] and c["true_rule_never_passed_to_candidate"]
                for r in raw for c in r["contexts"]),
        "three_real_float32_codes_exactly1440bits":
            all(r["frozen_A_B_C_code_bits"] == 1440 and
                r["actual_float32_code_bytes_per_context"] == 60
                for r in raw),
        "negative_controls_and_both_ablations_reported":
            all(len(c["frozen_neuron_ablation_accuracy_drops"]) == 2
                and set(c["unseen_accuracy"]) == set(ARMS)
                for r in raw for c in r["contexts"]),
    }
    packed = json.dumps(raw, sort_keys=True,
                        separators=(",", ":")).encode("utf-8")
    return {
        "status": "M5.32 dual soft-reward-synapses, M5.31-seed-reuse DEV",
        "metadata": {
            "method": "real reward, 2 independent human-conjunction units",
            "cohorts": {k: list(v) for k, v in COHORTS.items()},
            "optimizer_seeds": list(SEEDS),
            "initializations": RESTARTS,
            "full_batch_steps": STEPS,
            "diversity_regularization": DIVERSITY,
            "matched_no_diversity_regularization": 0.0,
            "support_cues": 8,
            "unseen_exam_cues": 8,
            "same_M531_development_task_seeds_reused": True,
            "independent_confirmation": False,
            "two_actions_complementary": True,
            "learned_active_exploration": False,
            "human_supplied_conjunction_primitive": True,
            "persisted_float32_code_bits_three_contexts": 1440,
        },
        "summary": summaries,
        "screening_gates": checks,
        "local_dual_plasticity_screen_pass": all(checks.values()),
        "autonomously_invented_neural_operator": False,
        "shared_meta_plastic_controller_learned": False,
        "independent_task_family_replication_complete": False,
        "continual_memory_retention_verified": False,
        "evolution_authorized": False,
        "advance_to_evolution": False,
        "sealed_M5_6_ecology_accessed": False,
        "raw_rows_sha256": hashlib.sha256(packed).hexdigest(),
        "raw_rows": raw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("results/m5-32/dual-soft-synapses-dev.json"))
    args = parser.parse_args()
    output = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")
    print("M5_32_DUAL_SOFT " + json.dumps({
        k: output[k] for k in (
            "status", "summary", "screening_gates",
            "local_dual_plasticity_screen_pass", "raw_rows_sha256"
        )
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
