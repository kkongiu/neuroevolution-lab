"""M5.33 active sensory exploration preregistration and leakage tests."""
import inspect
import unittest
import numpy as np

from neuroevolution.m5_33_active_cue import (
    CATALOG, COHORTS, OPTIMIZER_SEEDS, POLICIES,
    START_CUES, SUPPORT_BUDGET, CUES,
    _seed_plan, acquire_own_support, candidate_catalog,
    choose_next_cue, fresh_rule_for, one_context,
    posterior_indices, prior_has_truth, theory_diagnostics,
)
from neuroevolution.m5_31_synaptic_growth import wanted_action


class M533Tests(unittest.TestCase):
    def test_protocol_and_true_family_in_human_prior(self):
        self.assertEqual([len(x) for x in COHORTS.values()], [8, 12, 12, 12])
        self.assertEqual(len(set().union(*map(set, COHORTS.values()))), 44)
        self.assertEqual(OPTIMIZER_SEEDS, (1031, 1033))
        self.assertEqual(len(POLICIES), 4)
        self.assertEqual(START_CUES, 2)
        self.assertEqual(SUPPORT_BUDGET, 8)
        self.assertEqual(len(CUES), 16)
        self.assertEqual(CATALOG.shape[1], 16)
        self.assertGreater(len(CATALOG), 15)
        np.testing.assert_array_equal(candidate_catalog(), CATALOG)
        self.assertEqual(
            len(set(tuple(x) for x in CATALOG.tolist())), len(CATALOG)
        )
        for cohort, seeds in COHORTS.items():
            with self.subTest(cohort=cohort):
                for seed in seeds:
                    rules = fresh_rule_for(seed, cohort)
                    self.assertEqual(len(rules), 3)
                    self.assertEqual(rules[0][0], rules[1][0])
                    self.assertEqual(rules[0][1] ^ rules[1][1], 1)
                    for rule in rules:
                        self.assertTrue(prior_has_truth(rule))
                with self.assertRaises(ValueError):
                    fresh_rule_for(7100, cohort)

    def test_posterior_from_own_action_and_reward_only(self):
        seed, cohort = 12400, "two_disjoint_pairs"
        rule = fresh_rule_for(seed, cohort)[0]
        prior = posterior_indices(())
        self.assertEqual(len(prior), len(CATALOG))
        records = []
        for cue in (2, 7, 1, 11):
            actual_action = cue % 2
            reward = int(actual_action == wanted_action(cue, rule))
            records.append((cue, actual_action, reward))
            candidates = posterior_indices(records)
            self.assertGreater(len(candidates), 0)
            full = [wanted_action(c, rule) for c in CUES]
            self.assertTrue(
                any(all(CATALOG[i, c] == full[c] for c in CUES)
                    for i in candidates)
            )
        self.assertLessEqual(len(posterior_indices(records)), len(prior))
        self.assertEqual(
            set(inspect.signature(choose_next_cue).parameters),
            {"records", "policy", "tie_order"}
        )
        self.assertEqual(
            set(inspect.signature(posterior_indices).parameters),
            {"records"}
        )

    def test_four_policies_share_initial_cues_actions_reward_and_budget(self):
        seed, cohort = 12401, "two_disjoint_pairs"
        rule = fresh_rule_for(seed, cohort)[0]
        order, actions, false_rewards = _seed_plan(seed, 0)
        first_two = None
        by_policy = {}
        for policy in POLICIES:
            own, exam, trace, resets = acquire_own_support(
                seed, 0, rule, policy
            )
            by_policy[policy] = own
            if first_two is None:
                first_two = tuple(r[0] for r in own[:2])
            self.assertEqual(tuple(r[0] for r in own[:2]), first_two)
            self.assertEqual(first_two, order[:2])
            self.assertEqual(len(own), 8)
            self.assertEqual(len(set(r[0] for r in own)), 8)
            self.assertEqual(len(exam), 8)
            self.assertFalse(set(exam) & {r[0] for r in own})
            self.assertEqual(len(trace), 8)
            self.assertTrue(all(log["requested_cue"] == rec[0]
                                for log, rec in zip(trace, own)))
            for cue, action, observed_reward in own:
                self.assertEqual(action, actions[cue])
                self.assertEqual(
                    observed_reward, int(action == wanted_action(cue, rule))
                )
            self.assertTrue(theory_diagnostics(own, rule)["true_function_retained"])
            self.assertEqual(resets, sum(x["planner_belief_reset"] for x in trace))
        self.assertEqual(tuple(r[0] for r in by_policy["random_cues"]), order[:8])
        self.assertEqual(
            len({tuple(r[0] for r in own) for own in by_policy.values()}) >= 2,
            True
        )

    def test_planning_is_not_oracle_learner_signature(self):
        self.assertEqual(
            set(inspect.signature(acquire_own_support).parameters),
            {"task_seed", "ctx", "rule", "policy"}
        )
        self.assertEqual(
            set(inspect.signature(choose_next_cue).parameters),
            {"records", "policy", "tie_order"}
        )
        with self.assertRaises(ValueError):
            choose_next_cue((), "family_version_space_active", range(16))
        with self.assertRaises(ValueError):
            choose_next_cue(((1, 0, 1), (2, 0, 0)), "unrecognized", range(16))
        with self.assertRaises(ValueError):
            choose_next_cue(
                ((1, 0, 1), (1, 1, 0)), "family_version_space_active",
                range(16)
            )

    def test_full_context_records_all_four_arms_and_neural_budget(self):
        row = one_context(12400, 0, "two_disjoint_pairs", 1031)
        self.assertEqual(set(row["arms"]), set(POLICIES))
        self.assertEqual(len(row["precommitted_action_by_cue"]), 16)
        for policy in POLICIES:
            a = row["arms"][policy]
            self.assertEqual(len(a["observations"]), 8)
            self.assertEqual(len(a["unseen_cues"]), 8)
            self.assertTrue(a["no_support_exam_overlap"])
            self.assertTrue(a["same_precommitted_binary_actions_for_all_arms"])
            self.assertTrue(a["same_15_parameter_neural_fit_each_arm"])
            self.assertEqual(a["frozen_code_bytes"], 60)
            self.assertEqual(a["optimizer_gradient_updates"], 2400)
            self.assertEqual(len(a["frozen_float32_weights"]), 15)
            self.assertEqual(len(a["all16_frozen_predictions"]), 16)
            self.assertEqual(len(a["stepwise_selection_trace"]), 8)
            self.assertTrue(a["theoretical_identifiability"]["true_function_retained"])
            self.assertTrue(0 <= a["all16_accuracy_including_seen_support"] <= 1)
            self.assertTrue(
                0 <= a["own_unseen8_accuracy_different_exam_set_per_arm"] <= 1
            )


if __name__ == "__main__":
    unittest.main()
