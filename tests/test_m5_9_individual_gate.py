"""M5.9c positive learning prerequisite has genuinely separate controls."""
import unittest
from neuroevolution.m5_9_individual_gate import (
    ARMS, PROBE_SEEDS, TRIALS_PER_PHASE, WINDOW, sensory, schedule, run, experiment
)


class IndividualGateTests(unittest.TestCase):
    def test_sensory_never_leaks_hidden_rule(self):
        self.assertEqual(len(sensory(1,0.)),19)
        self.assertEqual(sensory(0,0.),sensory(0,0.))
        self.assertEqual(sum(sensory(1,1.)),2.)

    def test_schedule_is_stable_and_balanced(self):
        for seed in PROBE_SEEDS:
            cues=schedule(seed)
            self.assertEqual(len(cues),2*TRIALS_PER_PHASE)
            self.assertEqual(cues,schedule(seed))
            for i in range(0,len(cues),WINDOW):
                self.assertEqual(sum(cues[i:i+WINDOW]),WINDOW//2)

    def test_frozen_plastic_and_shuffled_reward(self):
        s=3200
        records={a:run(s,a) for a in ARMS}
        self.assertEqual(records["frozen"]["private_weight_drift_l2"],0.)
        self.assertEqual(records["frozen"]["nonzero_policy_updates"],0)
        self.assertGreater(records["plastic"]["nonzero_policy_updates"],0)
        self.assertGreater(records["shuffled_feedback"]["nonzero_policy_updates"],0)
        self.assertEqual({v["memory_predictor_updates"] for v in records.values()},
                         {0})
        self.assertEqual(records, {a:run(s,a) for a in ARMS})

    def test_gate_rejects_sealed_and_has_explicit_failure_behavior(self):
        for seed in (1000,3000,3224,161000,162000,163000):
            with self.assertRaises(ValueError):run(seed,"plastic")
        with self.assertRaises(ValueError):run(3200,"unknown")
        with self.assertRaises(ValueError):experiment((3200,3200))
        r=experiment((3200,3201))
        self.assertEqual(len(r["raw_rows"]),6)
        self.assertEqual(r["advance_to_delayed_feedback"],
                         all(r["preregistered_screening_criteria"].values()))
        self.assertFalse(r["selection_or_reproduction"])


if __name__=="__main__":
    unittest.main()
