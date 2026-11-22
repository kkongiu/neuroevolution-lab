"""M5.15 bounded school: learning is from observed feedback, not labels."""
import unittest

from neuroevolution.m5_15_school import (
    SEEDS, TASKS, ARMS, TRAIN_STEPS, EXAM_STEPS, MEMORY_SIZE,
    PredictiveLearner, ticket_stream, features, target, run, experiment,
)


class PredictiveClassroomTests(unittest.TestCase):
    def test_balanced_lived_world_and_three_action_negative_feedback(self):
        for task in TASKS:
            s=ticket_stream(3800,task,TRAIN_STEPS+EXAM_STEPS)
            self.assertEqual(len(s),TRAIN_STEPS+EXAM_STEPS)
            for p in range(0,len(s),40):
                self.assertEqual(len(set(s[p:p+40])),4)
                self.assertEqual(len([v for v in s[p:p+40] if v==(0,0)]),10)
        self.assertEqual(target("xor",1,1),0)
        self.assertEqual(target("ternary",1,1),2)
        self.assertEqual(len(features(1,0,2,3)),6)
        self.assertNotEqual(target("ternary",1,1),0)
        self.assertNotEqual(target("ternary",1,1),1)

    def test_policy_receives_only_own_experienced_feedback(self):
        for arm in ARMS:
            if arm=="native_reinforce":continue
            p=PredictiveLearner(3800,"xor",arm)
            a=p.choose(0,1,0,True)
            self.assertIn(a,(0,1))
            p.experience(0,1,a,1)
            self.assertEqual(p.env_experiences,1)
            self.assertEqual(len(p.buffer),1)
            self.assertEqual(p.buffer[0][:3],(0,1,a))
            self.assertEqual(p.updates,1 if arm=="predict_online" else 5)
        # Actual stored record has chosen action and experienced feedback,
        # never a privileged desired alternative.
        for invalid in (3700,3824,161000,162000,163000):
            with self.assertRaises(ValueError):
                PredictiveLearner(invalid,"xor","predict_replay")

    def test_same_model_founders_and_bounded_memory(self):
        a=PredictiveLearner(3800,"ternary","predict_repeat")
        b=PredictiveLearner(3800,"ternary","predict_replay")
        self.assertEqual(a.w,b.w)
        self.assertEqual(a.v,b.v)
        self.assertEqual(a.bias,b.bias)
        self.assertEqual(a.dim,6)
        for i in range(MEMORY_SIZE+15):
            b.experience(i%2,(i//2)%2,i%3,int(i%4==0))
        self.assertEqual(len(b.buffer),MEMORY_SIZE)
        self.assertEqual(b.env_experiences,MEMORY_SIZE+15)
        self.assertEqual(b.updates,(MEMORY_SIZE+15)*5)

    def test_frozen_examinations_reproducible_and_gated(self):
        for task in TASKS:
            for arm in ARMS:
                r=run(3800,task,arm)
                self.assertEqual(r["training_environment_experiences"],
                                 TRAIN_STEPS)
                self.assertTrue(r["frozen_exam"])
                self.assertTrue(r["individual_memory_not_inherited"])
                self.assertTrue(r["no_correct_action_labels"])
                self.assertEqual(len(r["train_curve"]),TRAIN_STEPS//120)
                self.assertEqual(r,run(3800,task,arm))
                self.assertTrue(0<=r["exam_accuracy"]<=1)
                self.assertEqual(r["exam_accuracy"],
                                 r["exam_successes"]/EXAM_STEPS)
        with self.assertRaises(ValueError):
            experiment((3800,3800))
        summary=experiment((3800,3801))
        self.assertTrue(summary["replay_and_repeat_equal_update_budget"])
        self.assertFalse(summary["reproduction_or_evolution"])
        self.assertEqual(summary["advance_to_evolution"],
                         all(summary["screening_gate"].values()))


if __name__=="__main__":
    unittest.main()
