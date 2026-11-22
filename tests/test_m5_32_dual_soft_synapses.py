"""M5.32 gradient, no-oracle, causal-ablation and budget-matching tests."""
import inspect
import unittest

import numpy as np

from neuroevolution.m5_32_dual_soft_synapses import (
    ALPHA_L1, ARMS, COHORTS, DIVERSITY, GATES, N_FLOATS,
    PERSISTENT_BITS_PER_CONTEXT, RESTARTS, SEEDS, STEPS,
    DualSoftSynapses, _ctx_oracle_masks, _oracle_experience,
    one_trial,
)
from neuroevolution.m5_31_synaptic_growth import own_experiences, rule_for


class M532Tests(unittest.TestCase):
    def sample(self):
        rule = rule_for(11400, "two_disjoint_pairs_outside_one_neuron")[0]
        support, exam = own_experiences(
            11400, "two_disjoint_pairs_outside_one_neuron", 0, rule
        )
        return support, exam, rule

    def test_registered_dev_comparison_and_frozen_protocol(self):
        self.assertEqual(SEEDS, (977, 983, 991))
        self.assertEqual([len(v) for v in COHORTS.values()], [12, 16, 16, 8])
        self.assertEqual(N_FLOATS, 15)
        self.assertEqual(PERSISTENT_BITS_PER_CONTEXT, 480)
        self.assertEqual(RESTARTS, 8)
        self.assertEqual(STEPS, 300)
        self.assertEqual(DIVERSITY, .025)
        self.assertEqual(ALPHA_L1, .01)
        self.assertEqual(len(ARMS), 7)
        self.assertEqual(GATES["both_units_causal_frequency"], .10)
        self.assertEqual(
            set(inspect.signature(DualSoftSynapses.fit).parameters),
            {"self", "records"}
        )

    def test_real_reward_only_and_identical_matched_starts(self):
        support, exam, rule = self.sample()
        self.assertEqual(len(support), 8)
        self.assertEqual(len(exam), 8)
        self.assertFalse({c for c, _, _ in support} & set(exam))
        primary = DualSoftSynapses(64317, DIVERSITY)
        matched = DualSoftSynapses(64317, 0.)
        np.testing.assert_array_equal(primary.w, matched.w)
        np.testing.assert_array_equal(primary.a, matched.a)
        np.testing.assert_array_equal(primary.g, matched.g)
        primary.fit(support)
        matched.fit(support)
        self.assertEqual(primary.gradient_updates, RESTARTS * STEPS)
        self.assertEqual(matched.gradient_updates, RESTARTS * STEPS)
        self.assertEqual(primary.frozen_code.dtype, np.float32)
        self.assertEqual(primary.frozen_code.nbytes, 60)
        self.assertEqual(len(primary.predictions(exam)), 8)
        self.assertEqual(len(primary.predictions(exam, ablate=0)), 8)
        self.assertEqual(len(primary.predictions(exam, ablate=1)), 8)
        self.assertEqual(len(primary.structure()["learned_gates"]), 2)
        self.assertEqual(len(primary.structure()["learned_strengths"]), 2)
        np.testing.assert_array_equal(
            primary.frozen_code,
            DualSoftSynapses(64317, DIVERSITY).fit(support).frozen_code,
        )
        with self.assertRaises(ValueError):
            DualSoftSynapses(33, .4)
        with self.assertRaises(ValueError):
            DualSoftSynapses(33).fit(support[:-1])
        with self.assertRaises(ValueError):
            DualSoftSynapses(33).predictions(exam)
        with self.assertRaises(ValueError):
            primary.predictions(exam, ablate=2)

    def test_exact_gradient_finite_difference(self):
        support, _, _ = self.sample()
        learner = DualSoftSynapses(641, DIVERSITY)
        w, g, a = learner.w.copy(), learner.g.copy(), learner.a.copy()
        cues = tuple(c for c, _, _ in support)
        from neuroevolution.m5_29_compositional_modules import (
            sign_targets_from_observations,
        )
        targets = sign_targets_from_observations(support)
        loss, gw, gg, ga = learner._loss_grads(
            w, g, a, cues, targets, DIVERSITY
        )
        self.assertTrue(np.isfinite(loss).all())
        self.assertTrue(np.isfinite(gw).all())
        self.assertTrue(np.isfinite(gg).all())
        self.assertTrue(np.isfinite(ga).all())
        eps = 1e-6
        # Every module class, both independent neurons and strength are
        # checked numerically at a non-boundary random initialization.
        cases = [
            (w, gw, (3, 0)), (w, gw, (3, 2)), (w, gw, (3, 4)),
            (g, gg, (3, 0, 0)), (g, gg, (3, 0, 3)),
            (g, gg, (3, 1, 0)), (g, gg, (3, 1, 2)),
            (a, ga, (3, 0)), (a, ga, (3, 1)),
        ]
        for arr, grads, idx in cases:
            arr[idx] += eps
            plus = learner._loss_grads(
                w, g, a, cues, targets, DIVERSITY)[0][3]
            arr[idx] -= 2 * eps
            minus = learner._loss_grads(
                w, g, a, cues, targets, DIVERSITY)[0][3]
            arr[idx] += eps
            numeric = (plus - minus) / (2 * eps)
            self.assertAlmostEqual(
                numeric, grads[idx], delta=2e-6,
                msg=f"Mismatch at {idx}: numeric {numeric}, analytic {grads[idx]}"
            )

    def test_environment_only_oracle_is_separate_from_fit(self):
        support, _, rule = self.sample()
        oracle = _ctx_oracle_masks(rule)
        self.assertEqual(len(oracle), 2)
        self.assertEqual((oracle[0] & oracle[1]), 0)
        transformed = _oracle_experience(support, oracle)
        self.assertEqual(
            [r[2] for r in transformed],
            [r[2] for r in support],
        )
        self.assertTrue(any(
            transformed[i][1] != support[i][1]
            for i in range(len(support))
        ))
        self.assertEqual(
            set(inspect.signature(DualSoftSynapses._loss_grads).parameters),
            {"w", "g", "a", "cues", "targets", "diversity"}
        )

    def test_single_full_trial_auditable_and_nonleaky(self):
        row = one_trial(
            977, 11400, "two_disjoint_pairs_outside_one_neuron"
        )
        self.assertEqual(set(row["unseen_accuracy"]), set(ARMS))
        self.assertEqual(len(row["contexts"]), 3)
        self.assertEqual(row["frozen_A_B_C_code_bits"], 1440)
        for c in row["contexts"]:
            self.assertTrue(c["no_exam_overlap"])
            self.assertTrue(c["true_rule_never_passed_to_candidate"])
            self.assertTrue(c["strictly_matched_same_initialization_and_steps"])
            self.assertTrue(
                c["support_actions_randomly_generated_not_exploration_policy"]
            )
            self.assertEqual(len(c["frozen_neuron_ablation_accuracies"]), 2)
            self.assertEqual(len(c["frozen_neuron_ablation_accuracy_drops"]), 2)
            self.assertIn(c["random_unit_ablation_control_index"], (0, 1))
            self.assertEqual(
                c["random_unit_ablation_control_accuracy"],
                c["frozen_neuron_ablation_accuracies"][c["random_unit_ablation_control_index"]],
            )
            self.assertEqual(
                len(c["primary_frozen_15_float32_weights"]), 15
            )
            self.assertEqual(len(c["support_cue_ids"]), 8)
            self.assertEqual(len(c["exam_cue_ids"]), 8)
            self.assertEqual(set(c["unseen_accuracy"]), set(ARMS))
            self.assertTrue(all(
                0 <= x <= 1 for x in c["unseen_accuracy"].values()
            ))


if __name__ == "__main__":
    unittest.main()
