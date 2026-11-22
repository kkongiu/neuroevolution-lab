"""M5.31: support-only structural consolidation and real synapse audits."""
import inspect
import unittest
import numpy as np

from neuroevolution.m5_31_synaptic_growth import (
    COHORTS, MODEL_SEEDS, AFFINE_MASKS, DISJOINT_PAIRS, ARMS,
    SOFT_RESTARTS, SOFT_STEPS, EDGE_THRESHOLD, PERSISTENT_BITS_PER_CONTEXT,
    SoftSynapticNeuron, consolidate, corrected_records, fit_product,
    grade, is_conjunction, one_trial, own_experiences, rule_for, wanted_action,
    corrupt_records, evaluate_hard,
)


class M531Tests(unittest.TestCase):
    def test_registered_fresh_synthetic_development_cohorts(self):
        self.assertEqual(
            [len(v) for v in COHORTS.values()], [12, 16, 16, 8]
        )
        self.assertEqual(len(set().union(*[set(v) for v in COHORTS.values()])), 52)
        self.assertEqual(MODEL_SEEDS, (881, 883, 887))
        self.assertEqual(PERSISTENT_BITS_PER_CONTEXT, 164)
        self.assertEqual(SOFT_RESTARTS, 16)
        self.assertEqual(SOFT_STEPS, 300)
        self.assertEqual(EDGE_THRESHOLD, .50)
        self.assertEqual(len(ARMS), 7)
        self.assertTrue(all(m in range(16) for m in AFFINE_MASKS))
        for pair in DISJOINT_PAIRS:
            self.assertEqual(pair[0] & pair[1], 0)
        with self.assertRaises(ValueError):
            rule_for(9000, "triple_interaction")
        with self.assertRaises(ValueError):
            rule_for(11300.0, "triple_interaction")
        with self.assertRaises(ValueError):
            is_conjunction(2, 16)

    def test_true_reward_and_frozen_exam_firewall(self):
        for cohort, seeds in COHORTS.items():
            for seed in seeds[:2]:
                rules = rule_for(seed, cohort)
                self.assertEqual(rules[0][0], rules[1][0])
                self.assertEqual(rules[0][1] ^ rules[1][1], 1)
                for ctx, rule in enumerate(rules):
                    support, unseen = own_experiences(seed, cohort, ctx, rule)
                    self.assertEqual(len(support), 8)
                    self.assertEqual(len(unseen), 8)
                    self.assertEqual(len(set(x for x, _, _ in support)), 8)
                    self.assertFalse({x for x, _, _ in support} & set(unseen))
                    for cue, action, success in support:
                        self.assertEqual(success, int(action == wanted_action(cue, rule)))
                    if cohort == "pair_interaction":
                        self.assertEqual(rule[2].bit_count(), 2)
                    elif cohort == "triple_interaction":
                        self.assertEqual(rule[2].bit_count(), 3)
                    elif cohort == "plain_parity":
                        self.assertEqual(rule[2], 0)
                    else:
                        self.assertTrue(isinstance(rule[2], tuple))
                        self.assertEqual(rule[2][0] & rule[2][1], 0)
        self.assertEqual(set(inspect.signature(consolidate).parameters),
                         {"records", "seed"})
        self.assertEqual(set(inspect.signature(SoftSynapticNeuron.fit).parameters),
                         {"self", "records"})

    def test_soft_synapse_gradients_and_hard_pruning_are_real(self):
        rng = np.random.default_rng(451)
        records = tuple((cue, int(action), int(action == (cue & 1)))
                        for cue, action in zip(range(8), rng.integers(2, size=8)))
        soft = SoftSynapticNeuron(317).fit(records)
        self.assertEqual(soft.gradient_steps, SOFT_RESTARTS * SOFT_STEPS)
        self.assertEqual(soft.w.shape, (16, 5))
        self.assertEqual(soft.g.shape, (16, 4))
        self.assertEqual(soft.alpha.shape, (16,))
        self.assertTrue(np.isfinite(soft.soft_support_loss).all())
        self.assertTrue(np.isfinite(soft.g).all())
        self.assertTrue(np.all((0 <= soft.g) & (soft.g <= 1)))
        self.assertEqual(len(soft.frozen_soft_predict(tuple(range(8, 16)))), 8)
        self.assertTrue(all(
            soft.hard_mask(j) == 0 or soft.hard_mask(j).bit_count() >= 2
            for j in range(SOFT_RESTARTS)
        ))
        # Same deterministic support evidence means identical learned
        # continuous synapses. No hidden rule coefficient is provided.
        repeated = SoftSynapticNeuron(317).fit(records)
        self.assertTrue(np.array_equal(soft.g, repeated.g))
        self.assertEqual(soft.rank, repeated.rank)

    def test_consolidation_support_only_and_storage_audit(self):
        rule = rule_for(11200, "pair_interaction")[0]
        records, exam = own_experiences(11200, "pair_interaction", 0, rule)
        obj = consolidate(records, 1789)
        self.assertTrue(obj["mask"] in range(16))
        self.assertEqual(obj["model"].weights.dtype, np.float32)
        self.assertEqual(obj["model"].weights.nbytes, 20)
        self.assertIn(obj["mask"], obj["candidate_masks"])
        self.assertIn(0, obj["candidate_masks"])
        self.assertEqual(obj["new_synapses"] + obj["pruned_potential_synapses"], 4)
        self.assertEqual(len(obj["best_soft_gates"]), 4)
        self.assertEqual(len(evaluate_hard(obj["model"], obj["mask"], exam)), 8)
        self.assertEqual(obj["mask"], consolidate(records, 1789)["mask"])
        fake = corrupt_records(records, 1789)
        self.assertEqual(tuple(c for c, _, _ in fake),
                         tuple(c for c, _, _ in records))
        self.assertEqual(tuple(a for _, a, _ in fake),
                         tuple(a for _, a, _ in records))
        self.assertTrue(any(a != b for (_, _, a), (_, _, b) in zip(fake, records)))
        changed = consolidate(fake, 1789)
        self.assertNotEqual(
            sorted(obj["selection_support_costs"].values()),
            sorted(changed["selection_support_costs"].values())
        )
        for cue, action, reward in corrected_records(records, 3):
            self.assertIn(cue, range(16))
            self.assertIn(action, (0, 1))
            self.assertIn(reward, (0, 1))
        fixed = fit_product(records, 1789, 0)
        self.assertEqual(fixed.weights.nbytes, 20)

    def test_one_trial_positive_and_negative_arms_reported(self):
        for cohort in COHORTS:
            row = one_trial(MODEL_SEEDS[0], COHORTS[cohort][0], cohort)
            self.assertEqual(set(row["unseen_accuracy"]), set(ARMS))
            self.assertEqual(len(row["learned_synapse_masks"]), 3)
            self.assertEqual(len(row["new_synapse_counts"]), 3)
            self.assertEqual(len(row["best_soft_alpha"]), 3)
            self.assertEqual(row["three_task_code_bits"], 492)
            self.assertTrue(row["candidate_never_received_true_interaction"])
            self.assertTrue(row["own_actual_support_rewards_unseen_to_exams"])
            self.assertTrue(row["randomized_chosen_support_actions_not_learned_policy"])
            self.assertTrue(row["soft_to_hard_synapses_researcher_engineered"])
            self.assertTrue(row["original_sealed_ecology_untouched"])
            self.assertTrue(all(0 <= v <= 1 for v in row["unseen_accuracy"].values()))


if __name__ == "__main__":
    unittest.main()
