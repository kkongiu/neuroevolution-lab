"""M5.16 fairness and knowledge-retention controls on experienced feedback."""
import unittest
from neuroevolution.m5_16_continual_school import (
    ARMS,SEEDS,MEMORY_CAP,PROBE,TRAIN_A,TRAIN_B,RETRAIN_A,UPDATES,
    SchoolBrain,cues,sensory,environment_success,exam,run,experiment,
)

class ContinualSchoolTests(unittest.TestCase):
    def test_curriculum_is_matched_balanced_and_conflicting(self):
        for phase,n in ((0,TRAIN_A),(1,TRAIN_B),(2,RETRAIN_A),
                        (10,PROBE),(15,PROBE)):
            pairs=cues(3900,phase,n)
            self.assertEqual(len(pairs),n)
            for k in range(0,n,40):
                self.assertEqual(set(pairs[k:k+40]),
                                 {(0,0),(0,1),(1,0),(1,1)})
                self.assertEqual(pairs[k:k+40].count((1,1)),10)
        for x in (0,1):
            for y in (0,1):
                self.assertNotEqual(
                    environment_success(0,x,y,0),
                    environment_success(1,x,y,0))
                self.assertEqual(len(sensory(x,y,0,1)),7)

    def test_same_initial_weights_and_actual_rewards_only(self):
        brains=[SchoolBrain(3900,arm) for arm in ARMS]
        for b in brains[1:]:
            self.assertEqual(b.w,brains[0].w)
            self.assertEqual(b.v,brains[0].v)
            self.assertEqual(b.bias,brains[0].bias)
        b=SchoolBrain(3900,"context_replay")
        choice=b.choose(0,1,0,0,True)
        received=environment_success(0,0,1,choice)
        b.live(0,1,0,choice,received)
        self.assertEqual(tuple(b.context_buffers[0][0]),
                         (0,1,0,choice,received))
        self.assertEqual(b.lived_experiences,1)
        self.assertEqual(b.parameter_updates,UPDATES)
        self.assertEqual(len(b.context_buffers[0]),1)
        self.assertEqual(len(b.fifo),0)

    def test_stratification_preserves_old_A_while_fifo_discards(self):
        for arm in ("fifo_replay","context_replay","recent_repeat"):
            b=SchoolBrain(3900,arm)
            for tick in range(220):
                c=0 if tick<80 else 1
                x,y=(tick%2,(tick//2)%2)
                a=b.choose(x,y,c,tick,True)
                b.live(x,y,c,a,environment_success(c,x,y,a))
            self.assertEqual(b.lived_experiences,220)
            self.assertEqual(b.parameter_updates,220*UPDATES)
            self.assertLessEqual(
                len(b.fifo)+sum(len(v) for v in b.context_buffers.values()),
                MEMORY_CAP)
            if arm=="fifo_replay":
                self.assertFalse(any(record[2]==0 for record in b.fifo))
                self.assertEqual(len(b.fifo),128)
            if arm=="context_replay":
                self.assertEqual(len(b.context_buffers[0]),64)
                self.assertEqual(len(b.context_buffers[1]),64)
                self.assertGreater(b.replayed_old_context_during_B,0)
            if arm=="recent_repeat":
                self.assertEqual(len(b.fifo),0)
                self.assertEqual(len(b.context_buffers[0]),0)

    def test_exams_are_frozen_and_gate_does_not_assume_success(self):
        for arm in ARMS:
            obj=SchoolBrain(3900,arm)
            state=obj.digest_state()
            accuracy=exam(obj,3900,0,10)
            self.assertTrue(0<=accuracy<=1)
            self.assertEqual(state,obj.digest_state())
            record=run(3900,arm)
            self.assertTrue(record["exam_parameters_frozen"])
            self.assertEqual(record["lived_training_steps"],
                             TRAIN_A+TRAIN_B+RETRAIN_A)
            self.assertEqual(record["parameter_updates"],
                             (TRAIN_A+TRAIN_B+RETRAIN_A)*
                             (1 if arm=="online" else UPDATES))
            self.assertEqual(record,run(3900,arm))
        small=experiment((3900,3901))
        self.assertEqual(small["development_screen_pass"],
                         all(small["screening_gate"].values()))
        self.assertFalse(small["independent_replication_complete"])
        self.assertFalse(small["evolution_authorized"])
        self.assertFalse(small["advance_to_evolution"])
        self.assertEqual(small["summary"]["fifo_replay"][
                             "A_record_count_after_B"],0)
        self.assertEqual(small["summary"]["context_replay"][
                             "A_record_count_after_B"],64)

    def test_seed_firewall(self):
        for bad in (3800,3924,161000,162000,163000,3900.0):
            with self.assertRaises(ValueError):SchoolBrain(bad,"online")
        with self.assertRaises(ValueError):experiment((3900,3900))
        with self.assertRaises(ValueError):SchoolBrain(3900,"oracle")

if __name__=="__main__":
    unittest.main()
