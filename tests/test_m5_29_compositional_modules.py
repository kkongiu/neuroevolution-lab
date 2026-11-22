"""M5.29: neural task-gradient modules, own-action reward firewall, true heldout."""
import unittest
import numpy as np

from neuroevolution.m5_29_compositional_modules import (
    META_SEEDS,KNOWN_SEEDS,HELD_SEEDS,STRESS_SEEDS,COHORTS,
    TRAIN_MASKS,HELD_MASKS,TRAIN_RULES,HELD_RULES,ARMS,
    TASK_RESTARTS,ADAPT_STEPS,ADAPT_LR,ModularProductLearner,
    support_task,task_rules,actual_reward,one_arm,
    train_pooled_reference,seed_check,sign_targets_from_observations,
)
from neuroevolution.m5_28_neural_meta_concepts import NeuralConceptEncoder


class M529Tests(unittest.TestCase):
    def test_whole_rule_holdout_and_new_dev_seed_firewall(self):
        self.assertEqual(HELD_MASKS,(5,9,11,14))
        self.assertTrue(set(HELD_MASKS).isdisjoint(TRAIN_MASKS))
        self.assertEqual(len(TRAIN_RULES),24)
        self.assertEqual(len(HELD_RULES),8)
        self.assertEqual(len(META_SEEDS),3)
        self.assertEqual(len(KNOWN_SEEDS),24)
        self.assertEqual(len(HELD_SEEDS),24)
        self.assertEqual(len(STRESS_SEEDS),12)
        self.assertEqual(len(ARMS),7)
        self.assertFalse(set(HELD_SEEDS)&set(KNOWN_SEEDS))
        self.assertFalse(set(STRESS_SEEDS)&set(HELD_SEEDS))
        for cohort,seeds in COHORTS.items():
            for seed in seeds[:3]:
                seed_check(seed,cohort)
                rules=task_rules(seed,cohort)
                self.assertEqual(rules[0][0],rules[1][0])
                self.assertEqual(rules[0][1]^rules[1][1],1)
                if cohort=="fully_heldout_combinations":
                    self.assertTrue(all(mask in HELD_MASKS
                                        for mask,bias in rules))
                else:
                    self.assertTrue(all(mask in TRAIN_MASKS
                                        for mask,bias in rules))
        for invalid in (5900,6200.0,6224,6100,161000):
            with self.assertRaises(ValueError):
                seed_check(invalid,"fully_heldout_combinations")

    def test_no_query_leak_and_actual_feedback_for_support(self):
        for cohort,seeds in COHORTS.items():
            for seed in seeds[:2]:
                for context,rule in enumerate(task_rules(seed,cohort)):
                    records,held=support_task(seed,cohort,context,rule)
                    self.assertEqual(len(records),8)
                    self.assertEqual(len(held),8)
                    self.assertFalse({x for x,_,_ in records}&set(held))
                    for cue,action,success in records:
                        self.assertEqual(success,actual_reward(
                            rule[0],rule[1],cue,action,
                            cohort=="nonaffine_AND_stress"))
                        self.assertEqual(
                            sign_targets_from_observations(
                                ((cue,action,success),))[0],
                            1.-2.*(action^(1-success)))
        with self.assertRaises(ValueError):
            actual_reward(3,0,2,2)

    def test_gradient_modules_train_on_support_without_oracle(self):
        # Train-mask-only controlled fixtures; no heldout-mask tuning.
        train_mask=3
        seed=KNOWN_SEEDS[0]
        records,unseen=support_task(
            seed,"familiar_complete_rules",0,(train_mask,1))
        initial=ModularProductLearner(177,"product")
        trained=ModularProductLearner(177,"product")
        self.assertEqual(initial.digest(),trained.digest())
        trained.fit(records)
        self.assertEqual(trained.gradient_steps,TASK_RESTARTS*ADAPT_STEPS)
        self.assertEqual(trained.weights.shape,(5,))
        self.assertTrue(np.isfinite(trained.weights).all())
        self.assertLess(trained.support_mse,.05)
        self.assertNotEqual(trained.digest(),initial.digest())
        self.assertEqual(len(trained.predict(unseen)),8)
        self.assertTrue(set(trained.predict(unseen)).issubset({0,1}))
        # Exact same initial point, corrupted experienced outcomes:
        scrambled=ModularProductLearner(177,"product").fit(
            records,corrupt=True,corruption_seed=97)
        self.assertNotEqual(trained.digest(),scrambled.digest())
        with self.assertRaises(ValueError):
            ModularProductLearner(177,"symbolic_parity_solver")
        self.assertEqual(TASK_RESTARTS,8)
        self.assertEqual(ADAPT_STEPS,180)
        self.assertGreater(ADAPT_LR,0)

    def test_difference_between_additive_and_product_structure(self):
        p=ModularProductLearner(7123,"product")
        a=ModularProductLearner(7123,"additive")
        self.assertTrue(np.array_equal(p.initial,a.initial))
        # The alternative module grammar is different even with
        # identical scalar trainable parameters, independent of truth.
        p.weights=np.array([1.,-1.,-1.,1.,1.])
        a.weights=p.weights.copy()
        self.assertNotEqual(p.predict(tuple(range(16))),
                            a.predict(tuple(range(16))))

    def test_pooled_baseline_train_and_test_no_weight_mutation(self):
        pooled=train_pooled_reference(META_SEEDS[0],episodes=24)
        self.assertEqual(pooled.steps,24)
        before=pooled.digest()
        # Only familiar-function seed in unit tests, not the primary
        # held-out-function test cohort.
        for arm in ARMS:
            row=one_arm(
                pooled,META_SEEDS[0],KNOWN_SEEDS[0],
                "familiar_complete_rules",arm)
            self.assertEqual(row["unseen_queries_per_context"],8)
            self.assertEqual(row["support_experiences_per_context"],8)
            self.assertTrue(row["no_shared_neural_weight_updates_during_evaluation"])
            self.assertTrue(row["query_outcomes_unavailable_at_evaluation"])
            self.assertTrue(row["compositional_product_preprogrammed"])
            self.assertTrue(0<=row["unseen_mean"]<=1)
            self.assertEqual(pooled.digest(),before)
            if arm=="modular_product_sgd":
                self.assertEqual(row["task_code_float32_bits"],480)
                self.assertEqual(len(row["module_fit_diagnostics"]),3)
            if arm=="symbolic_GF2_oracle_prior":
                self.assertEqual(row["unseen_mean"],1)
        with self.assertRaises(ValueError):
            one_arm(pooled,META_SEEDS[0],KNOWN_SEEDS[0],
                    "familiar_complete_rules","privileged_teacher")
        with self.assertRaises(ValueError):
            train_pooled_reference(META_SEEDS[0],episodes=60001)


if __name__=="__main__":
    unittest.main()
