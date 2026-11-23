"""M5.34: cross-task learned neural representations, whole-function OOD test.

A 32->64->16 ordinary MLP learns to infer unknown query-cue labels
from a support-only 16-cue partial truth table and mask. PRETRAINING
uses entirely different truth functions and their FULL training
labels; TEST fitting makes NO parameter updates and receives only
eight actually observed action/reward pairs. All train/test functions
come from the same researcher-designed Boolean family; **not** true
independent family generalization or autonomously learned curiosity.

The human 112-function grammar is used ONLY in the environment's
active cue selector and the explicitly privileged symbolic oracle.
NEVER as neural architecture features, labels or an optimizer input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import numpy as np

from neuroevolution.m5_31_synaptic_growth import (
    AFFINE_MASKS, DISJOINT_PAIRS, wanted_action,
)
from neuroevolution.m5_32_dual_soft_synapses import DualSoftSynapses
from neuroevolution.m5_33_active_cue import (
    INTERACTIONS, CATALOG, CUES, _seed_plan, choose_next_cue,
    posterior_indices,
)

SPLIT_SEED = 3401
OPTIMIZER_SEEDS = (1301, 1303)
EPISODES_PER_HELDOUT_FUNCTION = 3
HIDDEN = 64
PRETRAIN_STEPS = 3000
BATCH = 32
LR = .003
ADAM_BETA1 = .9
ADAM_BETA2 = .999
ADAM_EPS = 1e-8
CUE_COUNT = 16
SUPPORT = 8
ARMS = (
    "shared_meta_representation",
    "scrambled_meta_pretraining",
    "untrained_same_network",
    "m532_independent_dual_soft",
    "human_privileged_exact_grammar",
)
POLICIES = ("random_cues", "family_version_space_active")
HOLDOUT = {
    "plain_parity": 1,
    "pair_interaction": 12,
    "triple_interaction": 8,
    "two_disjoint_pairs": 6,
}
COHORT_EFFECTS = {
    "plain_parity": tuple(e for e in INTERACTIONS if isinstance(e, int) and e == 0),
    "pair_interaction": tuple(e for e in INTERACTIONS if isinstance(e, int) and e.bit_count() == 2),
    "triple_interaction": tuple(e for e in INTERACTIONS if isinstance(e, int) and e.bit_count() == 3),
    "two_disjoint_pairs": DISJOINT_PAIRS,
}
CRITERIA = {
    "two_pair_unseen_accuracy": .60,
    "two_pair_gain_scrambled": .08,
    "two_pair_gain_untrained": .08,
    "two_pair_gain_dual_soft": .08,
    "four_family_unseen_accuracy": .70,
    "pair_unseen_accuracy": .70,
    "triple_unseen_accuracy": .75,
}


def entire_function_split():
    """No heldout truth function is ever in the cross-task training set."""
    rng = np.random.default_rng(SPLIT_SEED)
    train, heldout = [], []
    for cohort, effects in COHORT_EFFECTS.items():
        functions = []
        for parity in AFFINE_MASKS:
            for bias in (0, 1):
                for effect in effects:
                    rule = (parity, bias, effect)
                    truth = tuple(int(wanted_action(c, rule)) for c in CUES)
                    functions.append({
                        "cohort": cohort,
                        "rule": rule,
                        "truth": truth,
                        "signature": "".join(map(str, truth)),
                    })
        functions.sort(key=lambda f: f["signature"])
        permutation = rng.permutation(len(functions))
        n_hold = HOLDOUT[cohort]
        heldout.extend(functions[int(i)] for i in permutation[:n_hold])
        train.extend(functions[int(i)] for i in permutation[n_hold:])
    train_signatures = [f["signature"] for f in train]
    eval_signatures = [f["signature"] for f in heldout]
    if (len(train) != 85 or len(heldout) != 27
            or len(set(train_signatures + eval_signatures)) != 112
            or set(train_signatures) & set(eval_signatures)):
        raise AssertionError("Entire-function train/test split compromised")
    if not all(any(np.array_equal(CATALOG[k], f["truth"])
                   for k in range(len(CATALOG))) for f in train + heldout):
        raise AssertionError("Unexpected function outside known synthetic family")
    for cohort, n_hold in HOLDOUT.items():
        if sum(f["cohort"] == cohort for f in heldout) != n_hold:
            raise AssertionError("Stratified heldout split violated")
    return train, heldout


def input_from_records(records):
    """Infer observed binary label from each actually chosen action and reward."""
    if len(records) != SUPPORT or len({c for c, _, _ in records}) != SUPPORT:
        raise ValueError("Exactly 8 distinct support cue observations required")
    features = np.zeros(32, dtype=np.float32)
    for cue, action, success in records:
        if cue not in CUES or action not in (0, 1) or success not in (0, 1):
            raise ValueError("Invalid support triple")
        implied = int(action) ^ (1-int(success))
        features[cue] = np.float32(1-2*implied)
        features[16 + cue] = np.float32(1)
    return features


def heldout_support(function, episode_index: int, policy: str):
    """Environment-side only: true function generates chosen-action reward."""
    if not 0 <= episode_index < EPISODES_PER_HELDOUT_FUNCTION:
        raise ValueError("Unknown heldout episode")
    if policy not in POLICIES:
        raise ValueError("Unknown sampling arm")
    # Stable signature-derived seed; no learned parameters used to pick cues.
    key = int(hashlib.sha256(function["signature"].encode()).hexdigest()[:9], 16)
    order, actions, _ = _seed_plan(key, episode_index)
    records = []
    trace = []
    for step in range(SUPPORT):
        if step < 2 or policy == "random_cues":
            cue = order[step]
        else:
            cue, _ = choose_next_cue(
                records, "family_version_space_active", order
            )
        assert cue not in {r[0] for r in records}
        action = int(actions[cue])
        # The TRUE task labels enter ONLY this environmental transition.
        reward = int(action == function["truth"][cue])
        records.append((cue, action, reward))
        trace.append({
            "requested_cue": int(cue),
            "chosen_action": action,
            "observed_reward": reward,
            "posterior_surviving_human_functions": int(
                len(posterior_indices(records))
            ),
        })
    exam = tuple(cue for cue in CUES if cue not in {r[0] for r in records})
    if len(exam) != 8 or len(trace) != 8:
        raise AssertionError("Complete eight support / eight unseen rule violated")
    return tuple(records), exam, trace


class SharedRepresentation:
    """Ordinary supervised episodic MLP. Full TRAIN truth labels for meta training."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": (rng.standard_normal((32, HIDDEN))*np.sqrt(1/32)).astype(np.float32),
            "b1": np.zeros((HIDDEN,), dtype=np.float32),
            "w2": (rng.standard_normal((HIDDEN, 16))*np.sqrt(1/HIDDEN)).astype(np.float32),
            "b2": np.zeros((16,), dtype=np.float32),
        }
        self.steps = 0
        self.training_source_signatures = None
        self.pretraining_pairs_correct = None

    @staticmethod
    def _predict_batch(x, p):
        hidden = np.tanh(x @ p["w1"] + p["b1"])
        out = np.tanh(hidden @ p["w2"] + p["b2"])
        return hidden, out

    def pretrain(self, training_functions, corrupted=False):
        """Full truth used for TRAIN functions ONLY, never evaluation functions.

        The *corrupted* control uses identical support batches/seed
        and target training budget but a DIFFERENT TRAIN function's
        query labels, breaking support->target correspondence.
        """
        train_signatures = tuple(f["signature"] for f in training_functions)
        if len(train_signatures) != 85 or len(set(train_signatures)) != 85:
            raise ValueError("Expected 85 unique, separate TRAIN functions")
        labels = np.asarray([f["truth"] for f in training_functions], dtype=np.float32)
        if labels.shape != (85, 16):
            raise AssertionError("Wrong pretraining function dimensions")
        # Both real and negative controls get the SAME sampled support inputs.
        # Only corrupted control draws an independent target function.
        param_digest = int(hashlib.sha256(
            self.params["w1"].tobytes()).hexdigest()[:8], 16)
        sampling_rng = np.random.default_rng(param_digest + 14341)
        corrupted_rng = np.random.default_rng(param_digest + 14913)
        adam_m = {k: np.zeros_like(v) for k,v in self.params.items()}
        adam_v = {k: np.zeros_like(v) for k,v in self.params.items()}
        index = np.arange(BATCH)[:, None]
        for step in range(1, PRETRAIN_STEPS+1):
            indexes = sampling_rng.integers(0, len(labels), size=BATCH)
            source = labels[indexes]
            choice = sampling_rng.random((BATCH, 16)).argsort(axis=1)[:, :8]
            visible = np.zeros((BATCH, 16), dtype=np.float32)
            visible[index, choice] = 1.
            sign_source = 1. - 2. * source
            inputs = np.concatenate(
                (sign_source * visible, visible), axis=1
            ).astype(np.float32)
            if corrupted:
                # Different training function for every sampled episode.
                different = (
                    indexes + corrupted_rng.integers(1, len(labels), size=BATCH)
                ) % len(labels)
                targets = 1. - 2. * labels[different]
            else:
                targets = sign_source
            hidden, output = self._predict_batch(inputs, self.params)
            derivative = (
                2. / (BATCH * 8)
                * (output-targets)
                * (1.-visible)
                * (1.-output*output)
            )
            grads = {
                "w2": hidden.T @ derivative,
                "b2": derivative.sum(axis=0),
            }
            dh = (derivative @ self.params["w2"].T) * (1.-hidden*hidden)
            grads["w1"] = inputs.T @ dh
            grads["b1"] = dh.sum(axis=0)
            for key in self.params:
                grad = grads[key].astype(np.float32)
                adam_m[key] = (ADAM_BETA1*adam_m[key]
                               +(1.-ADAM_BETA1)*grad)
                adam_v[key] = (ADAM_BETA2*adam_v[key]
                               +(1.-ADAM_BETA2)*(grad*grad))
                corrected_m = adam_m[key] / (1.-ADAM_BETA1**step)
                corrected_v = adam_v[key] / (1.-ADAM_BETA2**step)
                self.params[key] = (
                    self.params[key] -
                    LR*corrected_m/(np.sqrt(corrected_v) + ADAM_EPS)
                ).astype(np.float32)
        self.steps = PRETRAIN_STEPS
        self.training_source_signatures = train_signatures
        self.pretraining_pairs_correct = not corrupted
        if sum(p.nbytes for p in self.params.values()) != 3152*4:
            raise AssertionError("Shared model contains 3152 float32 parameters")
        return self

    def predictions(self, records):
        """Frozen model: accepts ONLY actual observed 8 support records."""
        x = input_from_records(records)
        hidden, output = self._predict_batch(x[None, :], self.params)
        pred = [int(score < 0.) for score in output[0]]
        for cue, action, reward in records:
            # Same deterministic two-action support clamping for all MLPs.
            pred[cue] = int(action) ^ (1-int(reward))
        return tuple(pred), float(np.linalg.norm(hidden[0]))

    def checkpoint(self):
        return {
            "step_count": self.steps,
            "pretraining_pairs_correct": self.pretraining_pairs_correct,
            "source_function_count": (
                len(self.training_source_signatures)
                if self.training_source_signatures is not None else 0
            ),
            "parameters": {k:v.tolist() for k,v in self.params.items()},
            "float32_parameter_bytes": sum(v.nbytes for v in self.params.values()),
            "sha256": hashlib.sha256(
                b"".join(self.params[k].tobytes()
                         for k in ("w1","b1","w2","b2"))
            ).hexdigest(),
        }


def human_grammar_reference(records):
    candidates = posterior_indices(records)
    if len(candidates) == 0:
        raise AssertionError("Human grammar omitted original true function")
    return tuple(
        int(CATALOG[candidates, cue].mean() >= .5) for cue in CUES
    ), int(len(candidates))


def offline_score(pred, truth, exam, records):
    seen = [cue for cue, _, _ in records]
    return {
        "unseen8_accuracy": sum(int(pred[i] == truth[i]) for i in exam)/8.,
        "support8_accuracy": sum(int(pred[i] == truth[i]) for i in seen)/8.,
        "all16_accuracy_including_support": sum(
            int(pred[i] == truth[i]) for i in CUES
        ) / 16.,
    }


def one_trial(function, episode, policy, optimizer_seed, pretrained, corrupted, scratch):
    # None of pretrained/corrupted/scratch receives function/true answers.
    records, exam, selection_trace = heldout_support(function, episode, policy)
    predictions = {}
    hidden_norms = {}
    for arm, model in (
        ("shared_meta_representation", pretrained),
        ("scrambled_meta_pretraining", corrupted),
        ("untrained_same_network", scratch),
    ):
        pred, norm = model.predictions(records)
        predictions[arm] = pred
        hidden_norms[arm] = norm
    key = int(hashlib.sha256(
        function["signature"].encode()
    ).hexdigest()[:10], 16)
    dual = DualSoftSynapses(
        optimizer_seed * 1000003 + key * 31 + episode * 163,
        0.
    ).fit(records)
    predictions["m532_independent_dual_soft"] = dual.predictions(CUES)
    oracle, remaining = human_grammar_reference(records)
    predictions["human_privileged_exact_grammar"] = oracle
    return {
        "heldout_signature": function["signature"],
        "cohort": function["cohort"],
        "episode": episode,
        "sampling_policy": policy,
        "shared_pretraining_seed": optimizer_seed,
        "support_chosen_action_reward_triples": [list(r) for r in records],
        "eight_unseen_exam_ids": list(exam),
        "cue_selection_trace": selection_trace,
        "true_rule_OFFLINE_GRADING_ONLY": [
            function["rule"][0], function["rule"][1],
            (list(function["rule"][2]) if isinstance(function["rule"][2], tuple)
             else function["rule"][2])
        ],
        "true_all16_labels_OFFLINE_GRADING_ONLY": list(function["truth"]),
        "human_privileged_posterior_count": remaining,
        "arm_accuracies": {
            arm: offline_score(pred, function["truth"], exam, records)
            for arm, pred in predictions.items()
        },
        "arm_all16_frozen_predictions": {
            arm: list(p) for arm,p in predictions.items()
        },
        "hidden_representation_norms": hidden_norms,
        "same_support_across_all_learners": True,
        "no_heldout_function_used_to_pretrain": all(
            function["signature"] not in m.training_source_signatures
            for m in (pretrained, corrupted)
        ),
        "no_query_seen_in_any_training_support": not bool(
            {c for c,_,_ in records} & set(exam)
        ),
        "same_real_eight_rewards_all_learners": True,
        "dual_gradient_updates": dual.gradient_updates,
        "model_training_source_count": len(pretrained.training_source_signatures),
    }


def paired_bootstrap(raw, cohort, against, sample_policy=None, draws=1000):
    """Bootstrap functions, not individual correlated seed×episode trials."""
    filtered = [
        r for r in raw if r["cohort"] == cohort
        and (sample_policy is None or r["sampling_policy"] == sample_policy)
    ]
    signatures = sorted({r["heldout_signature"] for r in filtered})
    per_function = {
        s: statistics.mean(
            r["arm_accuracies"]["shared_meta_representation"]["unseen8_accuracy"] -
            r["arm_accuracies"][against]["unseen8_accuracy"]
            for r in filtered if r["heldout_signature"] == s
        ) for s in signatures
    }
    rng = random.Random(340134)
    samples = sorted(
        statistics.mean(
            per_function[rng.choice(signatures)] for _ in signatures
        ) for _ in range(draws)
    )
    return {
        "difference": statistics.mean(per_function.values()),
        "exploratory_function_cluster_boot95": [
            samples[int(draws*.025)],samples[int(draws*.975)-1]
        ],
    }


def run():
    train, test = entire_function_split()
    train_signatures = set(f["signature"] for f in train)
    test_signatures = set(f["signature"] for f in test)
    if train_signatures & test_signatures:
        raise AssertionError("Train-evaluation truth function leakage")
    models = {}
    for seed in OPTIMIZER_SEEDS:
        models[seed] = {
            "shared_meta_representation":
                SharedRepresentation(seed).pretrain(train, corrupted=False),
            "scrambled_meta_pretraining":
                SharedRepresentation(seed).pretrain(train, corrupted=True),
            "untrained_same_network": SharedRepresentation(seed),
        }
    raw = [
        one_trial(
            function, episode, policy, seed,
            models[seed]["shared_meta_representation"],
            models[seed]["scrambled_meta_pretraining"],
            models[seed]["untrained_same_network"],
        )
        for seed in OPTIMIZER_SEEDS
        for function in test
        for episode in range(EPISODES_PER_HELDOUT_FUNCTION)
        for policy in POLICIES
    ]
    summary = {}
    for cohort in HOLDOUT:
        subset = [r for r in raw if r["cohort"] == cohort]
        summary[cohort] = {
            "row_count": len(subset),
            "test_function_count": HOLDOUT[cohort],
            "arms": {
                arm: {
                    metric: statistics.mean(
                        r["arm_accuracies"][arm][metric] for r in subset
                    ) for metric in (
                        "unseen8_accuracy",
                        "support8_accuracy",
                        "all16_accuracy_including_support",
                    )
                } for arm in ARMS
            },
            "paired_bootstrap_against": {
                arm: paired_bootstrap(raw, cohort, arm)
                for arm in ARMS if arm != "shared_meta_representation"
            },
            "sampling_policy_breakdown": {
                policy: {
                    arm: statistics.mean(
                        r["arm_accuracies"][arm]["unseen8_accuracy"]
                        for r in subset if r["sampling_policy"] == policy
                    ) for arm in ARMS
                } for policy in POLICIES
            },
            "mean_privileged_posterior_candidates": statistics.mean(
                r["human_privileged_posterior_count"] for r in subset
            ),
        }
    two = summary["two_disjoint_pairs"]["arms"]
    meta = two["shared_meta_representation"]["unseen8_accuracy"]
    metrics = {
        "pooled_four_family_mean_meta_unseen8": statistics.mean(
            summary[c]["arms"]["shared_meta_representation"]["unseen8_accuracy"]
            for c in HOLDOUT
        ),
        "two_pair_meta_unseen8": meta,
        "two_pair_gain_vs_scrambled": meta -
            two["scrambled_meta_pretraining"]["unseen8_accuracy"],
        "two_pair_gain_vs_scratch": meta -
            two["untrained_same_network"]["unseen8_accuracy"],
        "two_pair_gain_vs_dual": meta -
            two["m532_independent_dual_soft"]["unseen8_accuracy"],
    }
    checks = {
        "two_pair_unseen8_ge60":
            meta >= CRITERIA["two_pair_unseen_accuracy"],
        "two_pair_gain_vs_scrambled_ge8pp":
            metrics["two_pair_gain_vs_scrambled"] >=
                CRITERIA["two_pair_gain_scrambled"],
        "two_pair_gain_vs_scratch_ge8pp":
            metrics["two_pair_gain_vs_scratch"] >=
                CRITERIA["two_pair_gain_untrained"],
        "two_pair_gain_vs_dual_soft_ge8pp":
            metrics["two_pair_gain_vs_dual"] >=
                CRITERIA["two_pair_gain_dual_soft"],
        "four_family_avg_unseen8_ge70":
            metrics["pooled_four_family_mean_meta_unseen8"] >=
                CRITERIA["four_family_unseen_accuracy"],
        "pair_unseen8_ge70":
            summary["pair_interaction"]["arms"]["shared_meta_representation"]
                ["unseen8_accuracy"] >= CRITERIA["pair_unseen_accuracy"],
        "triple_unseen8_ge75":
            summary["triple_interaction"]["arms"]["shared_meta_representation"]
                ["unseen8_accuracy"] >= CRITERIA["triple_unseen_accuracy"],
        "whole_function_train_test_disjoint_85_vs27":
            len(train_signatures) == 85 and len(test_signatures) == 27
            and not (train_signatures & test_signatures),
        "all_eight_support_rewards_and_eight_exam_unseen":
            all(len(r["support_chosen_action_reward_triples"]) == 8
                and len(r["eight_unseen_exam_ids"]) == 8
                and r["no_query_seen_in_any_training_support"]
                for r in raw),
        "test_targets_absent_from_meta_pretraining_and_controls_matched":
            all(r["no_heldout_function_used_to_pretrain"] and
                r["model_training_source_count"] == 85 and
                r["same_support_across_all_learners"] for r in raw)
            and all(models[s]["shared_meta_representation"].steps ==
                    models[s]["scrambled_meta_pretraining"].steps ==
                    PRETRAIN_STEPS for s in OPTIMIZER_SEEDS),
        "all_negative_results_raw_and_evolution_stop":
            len(raw) == 324 and
            all(set(r["arm_accuracies"]) == set(ARMS) and
                set(r["arm_all16_frozen_predictions"]) == set(ARMS)
                for r in raw),
    }
    checkpoints = {
        str(seed): {arm: model.checkpoint() for arm,model in
                    models[seed].items()}
        for seed in OPTIMIZER_SEEDS
    }
    raw_hash = hashlib.sha256(
        json.dumps(raw,sort_keys=True,separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "status": "M5.34 shared learned representation, whole unseen functions DEV",
        "metadata": {
            "split_seed": SPLIT_SEED,
            "train_functions": 85,
            "test_disjoint_functions": 27,
            "train_signatures": sorted(train_signatures),
            "heldout_signatures": sorted(test_signatures),
            "heldout_by_family": HOLDOUT,
            "optimizer_seeds": OPTIMIZER_SEEDS,
            "episode_repeats": EPISODES_PER_HELDOUT_FUNCTION,
            "sampling_policies": POLICIES,
            "support_per_task": SUPPORT,
            "unseen_exam_per_task": 8,
            "neural_pretrain_steps": PRETRAIN_STEPS,
            "neural_pretrain_batch": BATCH,
            "meta_pretrain_full_labels_ONLY_from_training_functions": True,
            "known_task_grammar_shared_but_test_truth_table_ids_disjoint": True,
            "human_privileged_active_sampler_and_oracle": True,
            "neural_meta_predictor_no_test_gradient_adaptation": True,
            "matched_final_support_but_NOT_training_compute": True,
        },
        "summary": summary,
        "metrics": metrics,
        "screening_gates": checks,
        "local_reusable_representation_screen_pass": all(checks.values()),
        "learned_general_curiosity_policy": False,
        "autonomously_invented_neural_operator": False,
        "shared_meta_plastic_controller_learned": False,
        "independent_task_family_replication_complete": False,
        "continual_memory_retention_verified": False,
        "evolution_authorized": False,
        "advance_to_evolution": False,
        "sealed_M5_6_ecology_accessed": False,
        "raw_rows_sha256": raw_hash,
        "raw_rows": raw,
        "full_pretrained_float32_model_checkpoints": checkpoints,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/m5-34/shared-representation-dev.json")
    )
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                        encoding="utf-8")
    print("M5_34_SHARED_REPRESENTATION "+json.dumps({
        k:result[k] for k in (
            "status","summary","metrics","screening_gates",
            "local_reusable_representation_screen_pass","raw_rows_sha256"
        )
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
