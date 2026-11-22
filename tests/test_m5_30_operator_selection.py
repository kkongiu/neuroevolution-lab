"""M5.30 invariants: own chosen-action feedback, frozen support and operator menu."""
import unittest

import numpy as np

from neuroevolution.m5_30_operator_selection import (
    AFFINE_SEEDS, PAIR_SEEDS, TRIPLE_SEEDS, COHORTS, META_SEEDS, MASKS,
    OPERATORS, ARMS, PAIR_INDEX, PAIR_PENALTY, PARAM_BITS_PER_TASK,
    gate, triple, rule_for, observed_reward, observations, transform,
    fit_bank, predict, one_trial,
)


class M530Tests(unittest.TestCase):
    def test_operator_menu_and_fresh_development_cohorts(self):
        self.assertEqual(len(OPERATORS), 7)
        self.assertEqual(len(PAIR_INDEX), 6)
        self.assertEqual(len(META_SEEDS), 3)
        self.assertEqual((len(AFFINE_SEEDS), len(PAIR_SEEDS), len(TRIPLE_SEEDS)),
                         (16, 16, 8))
        self.assertEqual(MASKS, (5, 9, 11, 14))
        self.assertEqual(PARAM_BITS_PER_TASK, 163)
        self.assertEqual(PAIR_PENALTY, .025)
        self.assertEqual(len(ARMS), 6)
        self.assertEqual(len(set().union(*map(set, COHORTS.values()))), 40)
        with self.assertRaises(ValueError):
            rule_for(6200, "parity_plus_pair_AND")
        with self.assertRaises(ValueError):
            gate(3, 7)

    def test_real_actions_complementary_and_heldout_firewall(self):
        for cohort, seeds in COHORTS.items():
            for seed in seeds[:2]:
                rules = rule_for(seed, cohort)
                self.assertEqual(rules[0][0], rules[1][0])
                self.assertNotEqual(rules[0][1], rules[1][1])
                for ctx, rule in enumerate(rules):
                    records, exam = observations(seed, cohort, ctx, rule)
                    self.assertEqual(len(records), 8)
                    self.assertEqual(len(exam), 8)
                    self.assertFalse({x for x, _, _ in records} & set(exam))
                    for cue, action, reward in records:
                        self.assertEqual(reward, observed_reward(*rule, cue, action))
                        self.assertNotEqual(
                            observed_reward(*rule, cue, action ^ 1), reward
                        )
                if cohort == "affine_parity":
                    self.assertEqual(tuple(r[2] for r in rules), (0, 0, 0))
                elif cohort == "parity_plus_pair_AND":
                    self.assertTrue(all(1 <= r[2] <= 6 for r in rules))
                else:
                    self.assertTrue(all(r[2] == -1 for r in rules))

    def test_pair_and_triple_corrections_are_real(self):
        for op in range(1, 7):
            a, b = PAIR_INDEX[op - 1]
            self.assertEqual(gate((1 << a) | (1 << b), op), 1)
            self.assertEqual(gate(0, op), 0)
        self.assertEqual(triple(7), 1)
        self.assertEqual(triple(3), 0)
        self.assertEqual(gate(15, 0), 0)
        self.assertNotEqual(observed_reward(0, 0, 1, 3, 0),
                            observed_reward(0, 0, 0, 3, 0))

    def test_support_only_operator_fits_and_float32_storage(self):
        rule = rule_for(PAIR_SEEDS[0], "parity_plus_pair_AND")[0]
        records, unseen = observations(
            PAIR_SEEDS[0], "parity_plus_pair_AND", 0, rule
        )
        transformed = transform(records, 2)
        for (cue, action, reward), (newcue, newaction, newreward) in zip(
            records, transformed
        ):
            self.assertEqual(cue, newcue)
            self.assertEqual(newaction, action ^ gate(cue, 2))
            self.assertEqual(reward, newreward)
        bank, winner, scores = fit_bank(records, 9124)
        self.assertEqual(len(bank), 7)
        self.assertIn(winner, range(7))
        self.assertEqual(len(scores), 7)
        self.assertTrue(all(np.isfinite(scores)))
        self.assertTrue(all(fit.weights.dtype == np.float32 for fit in bank))
        self.assertTrue(all(fit.weights.nbytes == 20 for fit in bank))
        self.assertTrue(all(fit.gradient_steps == 1440 for fit in bank))
        self.assertEqual(len(predict(bank[winner], winner, unseen)), 8)
        bank2, winner2, scores2 = fit_bank(records, 9124)
        self.assertEqual(winner, winner2)
        self.assertEqual(scores, scores2)
        corrupted, _, _ = fit_bank(records, 9124, randomized_rewards=True)
        self.assertTrue(any(not np.array_equal(a.weights, b.weights)
                            for a, b in zip(bank, corrupted)))
        with self.assertRaises(ValueError):
            fit_bank(records[:7], 9124)

    def test_frozen_exam_six_arms_reporting_and_stop_flags(self):
        row = one_trial(META_SEEDS[0], AFFINE_SEEDS[0], "affine_parity")
        self.assertEqual(set(row["unseen_accuracy"]), set(ARMS))
        self.assertEqual(len(row["chosen_operator_ids"]), 3)
        self.assertEqual(row["persistent_candidate_code_bits_for_three_contexts"],
                         489)
        self.assertTrue(row["all_real_arms_used_same_chosen_action_rewards"])
        self.assertTrue(row["candidate_did_not_access_true_operator"])
        self.assertTrue(row["eight_disjoint_frozen_exam_cues"])
        self.assertTrue(row["operator_menu_human_specified"])
        self.assertTrue(row["no_operator_meta_learning_or_evolution"])
        self.assertTrue(all(0 <= x <= 1 for x in row["unseen_accuracy"].values()))


if __name__ == "__main__":
    unittest.main()
