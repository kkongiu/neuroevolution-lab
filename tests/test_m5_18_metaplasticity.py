"""M5.18 gate and selective-plasticity scientific controls."""
import unittest
from neuroevolution.m5_18_metaplasticity import (
    ARMS, SEEDS, PHASES, TRAIN_STEPS, PROBE_STEPS, UPDATES,
    Learner, require_seed, curriculum, environmental_reward,
    frozen_exam, simulate, experiment, has_replay, has_plasticity,
    is_adaptive, LOOSE_PROTECTION_FRACTION
)

class MetaplasticityTests(unittest.TestCase):
    def test_equal_task_order_and_unknown_correct_action(self):
        self.assertEqual(len(SEEDS),24)
        self.assertEqual(PHASES,(
            ("A",400),("B",400),("C",400),("A",160),("B",160)))
        for phase,(name,n) in enumerate(PHASES):
            pairs=curriculum(4100,phase,n)
            self.assertEqual(len(pairs),n)
            for j in range(0,n,40):
                self.assertEqual(set(pairs[j:j+40]),
                                 {(0,0),(0,1),(1,0),(1,1)})
                self.assertEqual(pairs[j:j+40].count((0,0)),10)
        for x in (0,1):
            for y in (0,1):
                self.assertNotEqual(
                    environmental_reward(0,x,y,0),
                    environmental_reward(1,x,y,0))

    def test_founders_and_equal_stimuli_and_feedback_only(self):
        brains=[Learner(4100,arm) for arm in ARMS]
        for learner in brains:
            self.assertEqual(learner.original_weights,
                             brains[0].original_weights)
            self.assertTrue(len(learner.active)==145)
            self.assertFalse(learner.anchors)
        for brain in brains:
            x,y,ctx=0,1,0
            a=brain.choose(x,y,ctx,0,False)
            actual=environmental_reward(ctx,x,y,a)
            brain.observe(x,y,ctx,a,actual,0)
            self.assertEqual(brain.lived,1)
            self.assertEqual(brain.updates,UPDATES if has_replay(brain.arm) else 1)
            if is_adaptive(brain.arm):
                self.assertTrue(.1<=brain.protection_fraction<=1)
                self.assertEqual(len(brain.protection_trace),1)
            elif brain.arm=="loose_plasticity_only":
                self.assertEqual(brain.protection_fraction,
                                 LOOSE_PROTECTION_FRACTION)
            else:
                self.assertEqual(brain.protection_fraction,1)

    def test_gate_responds_without_curriculum_boundary_or_replay(self):
        brain=Learner(4100,"adaptive_plasticity_only")
        for tick in range(40):
            x,y=curriculum(4100,0,40)[tick]
            a=brain.choose(x,y,0,tick,True)
            actual=environmental_reward(0,x,y,a)
            brain.observe(x,y,0,a,actual,tick)
        self.assertEqual(brain.lived,40)
        self.assertEqual(brain.updates,40)
        self.assertTrue(any(g<1 for g in brain.protection_trace))
        before=brain.snapshot()
        gate=brain.protection_fraction
        num_trace=len(brain.protection_trace)
        frozen_exam(brain,4100,0,20)
        self.assertEqual(before,brain.snapshot())
        self.assertEqual(brain.protection_fraction,gate)
        self.assertEqual(len(brain.protection_trace),num_trace)
        brain.consolidate(0)
        self.assertEqual(len(brain.anchors),1)
        self.assertGreater(max(brain.total_importance),0)

    def test_matched_parameter_updates_and_negative_outcomes(self):
        for arm in ARMS:
            record=simulate(4100,arm)
            expected=TRAIN_STEPS*(UPDATES if has_replay(arm) else 1)
            self.assertEqual(record["lived_experiences"],TRAIN_STEPS)
            self.assertEqual(record["parameter_updates"],expected)
            self.assertTrue(record["frozen_probes"])
            self.assertEqual(
                record,simulate(4100,arm))
            self.assertTrue(0.<=record["average_protection_fraction"]<=1.)
        report=experiment((4100,4101))
        self.assertEqual(report["development_screen_pass"],
                         all(report["screening_gate"].values()))
        self.assertFalse(report["independent_replication_complete"])
        self.assertFalse(report["advance_to_evolution"])
        self.assertFalse(report["evolution_authorized"])
        self.assertTrue(report["no_structural_growth_or_pruning"])

    def test_seed_firewall(self):
        for invalid in (4000,4124,161000,162000,163000,4100.0):
            with self.assertRaises(ValueError):require_seed(invalid)
        with self.assertRaises(ValueError):
            experiment((4100,4100))
        with self.assertRaises(ValueError):
            Learner(4100,"oracle_labels")

if __name__=="__main__":
    unittest.main()
