"""M5.9d adaptive rate detector correctness and paired experimental fences."""
import unittest

from neuroevolution.m5_6_online import Learner, RATE
from neuroevolution.m5_9_adaptive_plasticity import (
    ARMS,SEEDS,PHASES,PHASE_LENGTH,WINDOW,ErrorMonitor,
    RATE_MIN,RATE_MAX,genotype,cues_for,run,experiment,
)


class AdaptivePlasticityTests(unittest.TestCase):
    def test_surprise_requires_error_shift_and_no_future_observations(self):
        ctrl=ErrorMonitor()
        for _ in range(70):
            ctrl.update(1)
        base=ctrl.fast_error-ctrl.slow_error
        elevated=ctrl.update(0)
        self.assertGreater(elevated,RATE_MIN)
        self.assertGreater(ctrl.fast_error-ctrl.slow_error,base)
        self.assertGreaterEqual(elevated,RATE_MIN)
        self.assertLessEqual(elevated,RATE_MAX)
        self.assertTrue(0<=ctrl.max_surprise<=1)

    def test_learning_rate_default_and_bounds(self):
        g=genotype(3224)
        learner=Learner.fresh(g,(5,5))
        observation=tuple([1.]+[0.]*18)
        a,cache=learner.action(observation,(0,1),__import__('random').Random(2))
        self.assertIsNotNone(a)
        for invalid in (-.1,.081,float("nan"),float("inf")):
            with self.assertRaises(ValueError):
                learner.learn(cache,1.,learning_rate=invalid)
        w=tuple(learner.weights)
        self.assertEqual(learner.learn(cache,1.,learning_rate=0.),0.)
        self.assertEqual(tuple(learner.weights),w)
        self.assertGreater(learner.learn(cache,1.),0.)
        self.assertEqual(RATE,.02)

    def test_balanced_cues_repeated_reversals_and_rate_variation(self):
        for seed in (3224,3247):
            cues=cues_for(seed)
            self.assertEqual(len(cues),PHASES*PHASE_LENGTH)
            for i in range(0,len(cues),WINDOW):
                self.assertEqual(sum(cues[i:i+WINDOW]),WINDOW//2)
            adaptive=run(seed,"adaptive")
            rates=adaptive["rate_history"][:]
            __import__("random").Random(seed*770_147+1241).shuffle(rates)
            records={arm:(adaptive if arm=="adaptive" else
                          run(seed,arm,yoked_rates=tuple(rates)
                              if arm=="yoked_shuffled_rate" else None))
                     for arm in ARMS}
            self.assertEqual(sorted(records["adaptive"]["rate_history"]),
                             sorted(records["yoked_shuffled_rate"]["rate_history"]))
            self.assertAlmostEqual(records["adaptive"]["mean_learning_rate"],
                                   records["yoked_shuffled_rate"]["mean_learning_rate"])
            with self.assertRaises(ValueError):
                run(seed,"yoked_shuffled_rate")
            for row in records.values():
                self.assertEqual(len(row["phase_windows"]),PHASES)
                self.assertEqual(row["policy_updates"],PHASES*PHASE_LENGTH)
                self.assertTrue(RATE_MIN<=row["min_learning_rate"]<=RATE_MAX)
                self.assertTrue(RATE_MIN<=row["max_learning_rate"]<=RATE_MAX)
            self.assertAlmostEqual(records["constant"]["mean_learning_rate"],.02)
            self.assertNotEqual(records["adaptive"]["min_learning_rate"],
                                records["adaptive"]["max_learning_rate"])

    def test_no_sealed_cohorts_and_gate_stays_exploratory(self):
        for seed in (3200,3250,162000,163000):
            with self.assertRaises(ValueError): genotype(seed)
            with self.assertRaises(ValueError): run(seed,"adaptive")
        with self.assertRaises(ValueError): experiment((3224,3224))
        result=experiment((3224,3225))
        self.assertEqual(len(result["raw_rows"]),10)
        self.assertEqual(result["advance_to_evolution"],
                         all(result["screening_gate"].values()))
        self.assertFalse(result["decision_oracle"])
        self.assertFalse(result["fitness_selection"])


if __name__=="__main__":
    unittest.main()
