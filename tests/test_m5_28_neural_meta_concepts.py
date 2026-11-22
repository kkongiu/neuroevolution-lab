"""M5.28: independent neural meta-learning and no oracle leakage."""
import unittest
import numpy as np

from neuroevolution.m5_28_neural_meta_concepts import (
    META_SEEDS,KNOWN_EVAL_SEEDS,NOVEL_EVAL_SEEDS,NONAFFINE_EVAL_SEEDS,
    HELD_MASKS,TRAIN_MASKS,TRAIN_RULES,HELD_RULES,
    CODE_DIM,DECODE_HIDDEN,PARAM_COUNT,SUPPORT_N,QUERY_N,
    NeuralConceptEncoder,actual_support,check_eval_seed,
    parity_rule,observed_reward,support_split,support_encoding,
    query_labels_from_real_actions,meta_train,task_triplet,evaluate_one,
)


class M528Tests(unittest.TestCase):
    def test_strict_meta_train_mask_partition_and_cohort_firewall(self):
        self.assertEqual(len(TRAIN_RULES),24)
        self.assertEqual(len(HELD_RULES),8)
        self.assertFalse(set(TRAIN_MASKS)&set(HELD_MASKS))
        self.assertFalse(set(KNOWN_EVAL_SEEDS)&set(NOVEL_EVAL_SEEDS))
        self.assertFalse(set(NOVEL_EVAL_SEEDS)&set(NONAFFINE_EVAL_SEEDS))
        self.assertEqual(PARAM_COUNT,5057)
        self.assertEqual(CODE_DIM,32)
        for cohort,seeds in (
            ("familiar_rule_new_split",KNOWN_EVAL_SEEDS),
            ("completely_heldout_rule",NOVEL_EVAL_SEEDS),
            ("nonaffine_counterexample",NONAFFINE_EVAL_SEEDS)):
            self.assertGreater(len(seeds),0)
            for seed in seeds[:3]:
                check_eval_seed(seed,cohort)
                rules=task_triplet(seed,cohort)
                if cohort=="completely_heldout_rule":
                    self.assertTrue(all(mask in HELD_MASKS
                                        for mask,bias in rules))
                else:
                    self.assertTrue(all(mask in TRAIN_MASKS
                                        for mask,bias in rules))
                self.assertEqual(rules[0][0],rules[1][0])
                self.assertNotEqual(rules[0][1],rules[1][1])
        for bad in (5100,5200,5800.0,6012):
            with self.assertRaises(ValueError):
                check_eval_seed(bad,"completely_heldout_rule")
        with self.assertRaises(ValueError):
            check_eval_seed(5900,"familiar_rule_new_split")

    def test_support_and_query_inputs_disjoint_and_full_rank(self):
        for seed in (11,13,179,211):
            rng=np.random.default_rng(seed)
            for _ in range(30):
                seen,unseen=support_split(rng)
                self.assertEqual(len(seen),8)
                self.assertEqual(len(unseen),8)
                self.assertFalse(set(seen)&set(unseen))
                self.assertEqual(set(seen)|set(unseen),set(range(16)))
                pivots={}
                for cue in seen:
                    row=1|(cue<<1)
                    while row:
                        p=row.bit_length()-1
                        if p not in pivots:
                            pivots[p]=row;break
                        row^=pivots[p]
                self.assertEqual(len(pivots),5)

    def test_only_chosen_action_reward_enters_neural_support(self):
        rng=np.random.default_rng(1234)
        seen,_=support_split(rng)
        example=actual_support(3,1,seen,rng,False)
        self.assertEqual(len(example),SUPPORT_N)
        coded=support_encoding(example)
        self.assertEqual(coded.shape,(8,5))
        for i,(cue,chosen,reward) in enumerate(example):
            self.assertEqual(reward,observed_reward(3,1,cue,chosen))
            self.assertEqual(int((coded[i,4]+1)/2),
                             chosen^(1-reward))
        with self.assertRaises(ValueError):
            observed_reward(1,0,4,2)
        self.assertNotEqual(
            parity_rule(3,1,0),parity_rule(3,0,0))

    def test_actual_gradient_train_and_frozen_eval_no_updates(self):
        model=NeuralConceptEncoder(META_SEEDS[0])
        initial=model.digest()
        rng=np.random.default_rng(131)
        seen,unseen=support_split(rng)
        records=actual_support(5,1,seen,rng)
        observed=query_labels_from_real_actions(5,1,unseen,rng)
        self.assertEqual(len(observed),QUERY_N)
        self.assertEqual(set(observed.tolist()),{0.,1.})
        one=model.train_episode(records,unseen,observed)
        self.assertGreaterEqual(one,0)
        self.assertLessEqual(one,1)
        self.assertEqual(model.steps,1)
        self.assertNotEqual(model.digest(),initial)
        code=model.encode(records)
        self.assertEqual(code.shape,(CODE_DIM,))
        self.assertEqual(code.dtype,np.float32)
        pred=model.predict(code,unseen)
        self.assertEqual(pred.shape,(QUERY_N,))
        self.assertTrue(np.isfinite(pred).all())
        model.meta_seed=META_SEEDS[0]
        digest=model.digest()
        for cohort,seed in (
            ("familiar_rule_new_split",KNOWN_EVAL_SEEDS[0]),
            ("completely_heldout_rule",NOVEL_EVAL_SEEDS[0]),
            ("nonaffine_counterexample",NONAFFINE_EVAL_SEEDS[0])):
            result=evaluate_one(model,seed,cohort,"trained_neural")
            self.assertEqual(model.digest(),digest)
            self.assertTrue(result["no_query_outcomes_observed_in_evaluation"])
            self.assertTrue(result["no_neural_weight_updates_in_evaluation"])
            self.assertEqual(result["code_float32_storage_bits"],3072)
            self.assertEqual(result["actual_support_experiences_per_context"],8)
            self.assertEqual(result["never_seen_cue_patterns_in_exam"],8)
            self.assertGreaterEqual(result["unseen_mean"],0)
            self.assertLessEqual(result["unseen_mean"],1)

    def test_same_meta_train_task_rng_when_rewards_scrambled(self):
        true=meta_train(META_SEEDS[0],False,episodes=40)
        false=meta_train(META_SEEDS[0],True,episodes=40)
        initial=NeuralConceptEncoder(META_SEEDS[0])
        self.assertEqual(true.steps,false.steps)
        self.assertNotEqual(true.digest(),false.digest())
        self.assertNotEqual(true.digest(),initial.digest())
        self.assertNotEqual(false.digest(),initial.digest())
        for bad in (0,5124):
            with self.assertRaises(ValueError):meta_train(bad,False,episodes=1)
        with self.assertRaises(ValueError):
            meta_train(META_SEEDS[0],False,episodes=60001)

    def test_separate_symbolic_reference_not_embedded_in_neural(self):
        model=NeuralConceptEncoder(META_SEEDS[0])
        model.meta_seed=META_SEEDS[0]
        for arm in ("literal_episodic","symbolic_gf2_reference",
                    "untrained_neural"):
            result=evaluate_one(
                model,KNOWN_EVAL_SEEDS[0],
                "familiar_rule_new_split",arm)
            self.assertTrue(0<=result["unseen_mean"]<=1)
            if arm=="symbolic_gf2_reference":
                self.assertEqual(result["unseen_mean"],1)
                self.assertEqual(result["code_float32_storage_bits"],15)
                self.assertTrue(all(c is not None for c in
                                    result["symbolic_codes"]))
            if arm=="literal_episodic":
                self.assertEqual(result["seen_mean"],1)
        with self.assertRaises(ValueError):
            evaluate_one(model,5900,
                         "completely_heldout_rule","oracle_leak")


if __name__=="__main__":
    unittest.main()
