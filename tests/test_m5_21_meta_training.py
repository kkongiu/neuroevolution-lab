"""M5.21 meta-training lifecycle, isolation, and frozen new-life transfer."""
import unittest

from neuroevolution.m5_21_meta_training import (
    TRAIN_SEEDS,EVAL_SEEDS,TRAIN_ARMS,EVAL_ARMS,STRENGTHS,
    GROUPS,BLOCK,TRAIN_STEPS,MEMORY_LIMIT,empty_q,copy_q,q_digest,
    valid,trials,Brain,Controller,run_life,train_controller,
    frozen_exam,environment_reward
)


class M521Tests(unittest.TestCase):
    def test_seed_firewall_and_task_cue_balance(self):
        self.assertEqual(len(TRAIN_SEEDS),64)
        self.assertEqual(len(EVAL_SEEDS),24)
        self.assertFalse(set(TRAIN_SEEDS)&set(EVAL_SEEDS))
        for invalid in (4200,4300,4464,4499,4524,161000,4400.0):
            with self.assertRaises(ValueError):valid(invalid)
        for phase,n in enumerate((400,400,400,160,160)):
            sequence=trials(4400,phase,n)
            self.assertEqual(len(sequence),n)
            for start in range(0,n,BLOCK):
                block=sequence[start:start+BLOCK]
                self.assertEqual(block.count((0,0)),10)
                self.assertEqual(set(block),{(0,0),(0,1),(1,0),(1,1)})

    def test_meta_policy_is_only_thing_transferred_across_lives(self):
        q,training=train_controller("train_real",(4400,4401))
        self.assertEqual(len(training),2)
        self.assertNotEqual(q_digest(q),q_digest(empty_q()))
        for life in training:
            self.assertEqual(life["fresh_brain_updates"],TRAIN_STEPS)
            self.assertEqual(life["q_updates"],28)
            self.assertEqual(life["effective_meta_blocks"],28)
            self.assertFalse(life["frozen_exams_run"])
            self.assertFalse(life["exam_feedback_used_in_training"])
        self.assertNotEqual(training[0]["fresh_brain_founder_sha256"],
                            training[1]["fresh_brain_founder_sha256"])
        q_hash=q_digest(q)
        result=run_life(4500,"learned_transfer",q=q,train_meta=False)
        self.assertEqual(q_digest(q),q_hash)
        self.assertEqual(result["q_updates"],0)
        self.assertTrue(result["frozen_exams_run"])
        self.assertEqual(result["fresh_brain_updates"],TRAIN_STEPS)
        self.assertEqual(result["memory_capacity"],MEMORY_LIMIT)
        self.assertEqual(result["q_parameter_sha256"],q_hash)
        self.assertNotEqual(result["fresh_brain_founder_sha256"],
                            training[0]["fresh_brain_founder_sha256"])
        with self.assertRaises(ValueError):
            run_life(4500,"train_real",q=q,train_meta=True)
        with self.assertRaises(ValueError):
            run_life(4400,"learned_transfer",q=q,train_meta=False)

    def test_negative_Q_reward_control_has_equal_training_budget(self):
        good,good_rows=train_controller("train_real",(4400,4401))
        bad,bad_rows=train_controller("train_corrupted",(4400,4401))
        self.assertNotEqual(q_digest(good),q_digest(bad))
        self.assertEqual(sum(r["q_updates"] for r in good_rows),56)
        self.assertEqual(sum(r["q_updates"] for r in bad_rows),56)
        for g,b in zip(good_rows,bad_rows):
            self.assertEqual(g["fresh_brain_founder_sha256"],
                             b["fresh_brain_founder_sha256"])
            self.assertEqual(g["fresh_brain_updates"],
                             b["fresh_brain_updates"])

    def test_frozen_exam_and_policy_purity_and_constant_control(self):
        brain=Brain(4500)
        before=brain.snapshot()
        score=frozen_exam(brain,4500,0,20)
        self.assertTrue(0<=score<=1)
        self.assertEqual(before,brain.snapshot())
        q=empty_q()
        copy=copy_q(q)
        self.assertIsNot(q,copy)
        c=Controller(4500,"learned_transfer",q=q,train_meta=False)
        strengths,_=c.choose(1,True)
        self.assertEqual(len(strengths),GROUPS)
        self.assertTrue(all(x in STRENGTHS for x in strengths))
        for arm in ("fixed_weak","fixed_strong"):
            obj=Controller(4500,arm,train_meta=False)
            weights,_=obj.choose(1,True)
            self.assertEqual(weights,(
                (.3,)*4 if arm=="fixed_weak" else (1.,)*4))
        with self.assertRaises(ValueError):
            Controller(4500,"train_real",train_meta=True)
        with self.assertRaises(ValueError):
            Controller(4400,"learned_transfer",train_meta=False)

    def test_evaluation_same_fresh_brains_computation_no_Q_updates(self):
        rows=[run_life(4500,a,train_meta=False) for a in EVAL_ARMS]
        self.assertEqual(len({
            r["fresh_brain_founder_sha256"] for r in rows}),1)
        self.assertEqual(len({
            r["reference_forward_evaluations"] for r in rows}),1)
        self.assertTrue(all(r["q_updates"]==0 for r in rows))
        self.assertTrue(all(r["fresh_brain_updates"]==TRAIN_STEPS for r in rows))
        self.assertTrue(all(r["memory_capacity"]==MEMORY_LIMIT for r in rows))
        self.assertTrue(all(r["effective_meta_blocks"]==28 for r in rows))
        self.assertEqual(rows,[
            run_life(4500,a,train_meta=False) for a in EVAL_ARMS])


if __name__=="__main__":
    unittest.main()
