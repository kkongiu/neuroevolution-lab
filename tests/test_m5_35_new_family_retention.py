"""M5.35: new-family, no-leak, equal adaptation budgets and forgetting audit."""
import unittest

import numpy as np
import neuroevolution.m5_34_learned_representation as old
import neuroevolution.m5_35_new_family_retention as new


class TestM535(unittest.TestCase):
    def test_true_new_family_exhaustive_no_overlap(self):
        catalog=new.enumerate_new_functions()
        self.assertEqual([len(x) for x in catalog.values()], [96,48])
        oldset={"".join(map(str,row)) for row in new.CATALOG.tolist()}
        self.assertEqual(len(oldset),112)
        self.assertFalse(set(catalog["conditional_mux"]) & oldset)
        self.assertFalse(set(catalog["signed_exact_k"]) & oldset)
        self.assertFalse(set(catalog["signed_exact_k"]) &
                         set(catalog["conditional_mux"]))
        self.assertEqual(new.NOVEL_SPLIT_SEED,3535)
        self.assertEqual(new.ADAPТ_STEPS if hasattr(new,"ADAPТ_STEPS") else
                         new.ADAPT_STEPS,1200)
        self.assertEqual(new.BATCH,32)
        self.assertEqual(new.EPISODES,2)

    def test_five_partition_whole_truth_disjoint_and_frozen(self):
        a,b,c,d,e=new.new_family_split()
        self.assertEqual([len(x) for x in (a,b,c,d,e)],
                         [85,27,60,28,56])
        groups=[{x["signature"] for x in v} for v in (a,b,c,d,e)]
        self.assertEqual(len(set().union(*groups)),256)
        self.assertEqual(sum(map(len,groups)),256)
        self.assertEqual(len({x["signature"] for x in d
                              if x["cohort"]=="conditional_mux"}),16)
        self.assertEqual(len({x["signature"] for x in d
                              if x["cohort"]=="signed_exact_k"}),12)
        self.assertEqual(len({x["signature"] for x in c
                              if x["cohort"]=="conditional_mux"}),40)
        self.assertEqual(len({x["signature"] for x in c
                              if x["cohort"]=="signed_exact_k"}),20)
        self.assertEqual(
            [x["signature"] for x in d],
            [x["signature"] for x in new.new_family_split()[3]],
        )

    def test_eight_chosen_action_reward_observations_disjoint_exam(self):
        _,oldheld,_,newheld,_=new.new_family_split()
        for f in (oldheld[0],newheld[0],newheld[-1]):
            for i in range(new.EPISODES):
                records,exam=new.chosen_support(f,i)
                self.assertEqual(len(records),8)
                self.assertEqual(len(set(v[0] for v in records)),8)
                self.assertEqual(len(exam),8)
                self.assertFalse(set(exam)&{v[0] for v in records})
                self.assertEqual(set(exam)|{v[0] for v in records},
                                 set(range(16)))
                for cue,action,success in records:
                    self.assertIn(action,(0,1))
                    self.assertEqual(success,int(action==f["truth"][cue]))
        with self.assertRaises(ValueError):
            new.chosen_support(newheld[0],99)

    def test_identical_initial_checkpoints_equal_training_budgets(self):
        a,b,c,d,_=new.new_family_split()
        # A fast mechanical smoke test, NOT scientific evaluation:
        # all full hyperparameters restored before CI's main campaign.
        original=new.ADAPT_STEPS
        oldpretrain=old.PRETRAIN_STEPS
        new.ADAPT_STEPS=3
        old.PRETRAIN_STEPS=3
        try:
            base=old.SharedRepresentation(1301).pretrain(a)
            hash_before=base.checkpoint()["sha256"]
            n=new.continue_supervised(base,a,c,1301,"N")
            h=new.continue_supervised(base,a,c,1301,"H")
            r=new.continue_supervised(base,a,c,1301,"R")
            self.assertEqual(base.checkpoint()["sha256"],hash_before)
            self.assertEqual(h.training_phase2["novel_train_episodes"],48)
            self.assertEqual(r.training_phase2["novel_train_episodes"],48)
            self.assertEqual(n.training_phase2["novel_train_episodes"],96)
            self.assertEqual(r.training_phase2["old_train_episodes"],48)
            self.assertEqual(n.training_phase2["old_train_episodes"],0)
            self.assertEqual(h.training_phase2["old_train_episodes"],0)
            self.assertEqual([n.steps,h.steps,r.steps],[6,6,6])
            for model in (base,n,h,r):
                self.assertEqual(
                    sum(v.nbytes for v in model.params.values()),12608
                )
                self.assertTrue(
                    all(v.dtype==np.float32 and np.isfinite(v).all()
                        for v in model.params.values())
                )
            self.assertNotEqual(n.checkpoint()["sha256"],hash_before)
            self.assertNotEqual(h.checkpoint()["sha256"],hash_before)
            self.assertNotEqual(r.checkpoint()["sha256"],hash_before)
            self.assertNotEqual(n.checkpoint()["sha256"],r.checkpoint()["sha256"])
            again=new.continue_supervised(base,a,c,1301,"R")
            self.assertEqual(r.checkpoint()["sha256"],again.checkpoint()["sha256"])
            snapshots={
                "before_new_families":base,"new_only_all_new":n,
                "half_novel_unique_no_replay":h,
                "replay_half_old_half_new":r,
            }
            result=new.trial(d[0],1301,0,snapshots)
            self.assertEqual(set(result["arms"]),set(new.ARM_NAMES))
            self.assertTrue(result["unseen_cues_disjoint"])
            self.assertEqual(len(result["frozen_eight_unobserved_cue_ids"]),8)
            self.assertEqual(len(result["actual_chosen_action_reward_support"]),8)
            for arm in result["arms"].values():
                self.assertEqual(len(arm["frozen_all16_predictions"]),16)
                self.assertGreaterEqual(arm["scores"]["unseen8_accuracy"],0)
                self.assertLessEqual(arm["scores"]["unseen8_accuracy"],1)
        finally:
            new.ADAPT_STEPS=original
            old.PRETRAIN_STEPS=oldpretrain

    def test_only_registered_training_set_used_by_sequential_fit(self):
        a,b,c,d,e=new.new_family_split()
        oldpretrain=old.PRETRAIN_STEPS
        steps=new.ADAPT_STEPS
        old.PRETRAIN_STEPS=1
        new.ADAPT_STEPS=1
        try:
            base=old.SharedRepresentation(1303).pretrain(a)
            with self.assertRaises(ValueError):
                new.continue_supervised(base,a,c,1303,"oracle")
            with self.assertRaises(ValueError):
                new.continue_supervised(base,a,c+d[:1],1303,"R")
            with self.assertRaises(ValueError):
                new.continue_supervised(base,a+b[:1],c,1303,"R")
        finally:
            new.ADAPT_STEPS=steps
            old.PRETRAIN_STEPS=oldpretrain


if __name__=="__main__":
    unittest.main()
