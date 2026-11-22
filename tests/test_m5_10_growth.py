"""M5.10: structural growth, no rule oracle, matched preallocation and return."""
import unittest
from neuroevolution.m5_10_growth import (
    DEV_SEEDS,ARMS,CONTEXT_ORDER,TRIALS_PER_PHASE,WINDOW,
    observation,context_rules,cue_schedule,run,experiment
)

class BrainGrowthTests(unittest.TestCase):
    def test_visible_context_not_hidden_correct_response(self):
        for context in range(3):
            a=observation(context,0,0.)
            self.assertEqual(len(a),19)
            self.assertEqual(sum(a),2.)
            self.assertEqual(a,observation(context,0,0.))
            self.assertEqual(a[2+context],1.)
        self.assertTrue(all(x in (0,1) for x in context_rules(3300)))

    def test_exogenous_balanced_cues(self):
        cues=cue_schedule(3300)
        self.assertEqual(len(cues),len(CONTEXT_ORDER)*TRIALS_PER_PHASE)
        for i in range(0,len(cues),WINDOW):
            self.assertEqual(sum(cues[i:i+WINDOW]),WINDOW//2)

    def test_growth_is_real_and_no_preallocation_performance_artifact(self):
        for seed in (3300,3301):
            rows={a:run(seed,a) for a in ARMS}
            self.assertEqual(rows["small_shared"]["modules_at_end"],1)
            self.assertEqual(rows["large_preallocated"]["modules_at_birth"],3)
            self.assertEqual(rows["growing_modules"]["modules_at_birth"],1)
            self.assertEqual(rows["growing_modules"]["modules_at_end"],3)
            self.assertEqual([v["tick"] for v in rows["growing_modules"]["new_module_events"]],
                             [TRIALS_PER_PHASE,3*TRIALS_PER_PHASE])
            self.assertEqual(rows["large_preallocated"]["answer_trace_sha256"],
                             rows["growing_modules"]["answer_trace_sha256"])
            self.assertEqual(rows["large_preallocated"]["active_expert_trace_sha256"],
                             rows["growing_modules"]["active_expert_trace_sha256"])
            self.assertLess(rows["growing_modules"]["allocated_hidden_unit_ticks"],
                            rows["large_preallocated"]["allocated_hidden_unit_ticks"])
            self.assertEqual(rows["growing_modules"],run(seed,"growing_modules"))
            self.assertFalse(rows["growing_modules"]["fitness_selection"])

    def test_guard_and_no_unregistered_holdouts(self):
        for seed in (3224,3324,161000,162000,163000):
            with self.assertRaises(ValueError):run(seed,"growing_modules")
        with self.assertRaises(ValueError):experiment((3300,3300))
        p=experiment((3300,3301))
        self.assertTrue(p["equivalence_control"]["all_seed_answer_traces_equal"])
        self.assertTrue(p["equivalence_control"]["all_seed_expert_routing_equal"])
        self.assertEqual(p["advance_to_evolution"],all(p["screening_gate"].values()))
        self.assertTrue(p["context_routing_is_engineered"])
        self.assertFalse(p["hidden_rule_is_visible_to_organism"])

if __name__=="__main__":
    unittest.main()
