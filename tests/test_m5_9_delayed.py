"""M5.9b true delayed reward and strict offspring genome separation."""
import math
import unittest

from neuroevolution.m5_6_online import Learner
from neuroevolution.m5_9_delayed import (
    ARMS, ATTEMPTS, CHANCE_SOLVE, EPISODES, GATES, POP, Problem,
    SOLVED_THRESHOLD, evolve, experiment, founders, solve, tasks,
)


class DelayedTaskTests(unittest.TestCase):
    def test_hidden_answers_not_available_to_agent(self):
        a=Problem(0,0,tuple([(0,1,1)]*ATTEMPTS))
        b=Problem(1,0,tuple([(0,1,1)]*ATTEMPTS))
        for gate in range(GATES):
            self.assertEqual(a.observed(0,gate,0.),
                             b.observed(0,gate,0.))
            self.assertNotEqual(a.correct(0,gate,0),
                                b.correct(0,gate,0))
        self.assertEqual(CHANCE_SOLVE,.125)
        self.assertEqual(SOLVED_THRESHOLD,4)

    def test_genetic_only_not_lifelong_memory(self):
        g=founders(3100)[0]
        self.assertEqual(len(g),260)
        a=solve(g,tasks(3100,0,EPISODES),9182,plastic=True)
        b=solve(g,tasks(3100,0,EPISODES),9182,plastic=False)
        self.assertGreater(a["policy_weight_updates"],0)
        self.assertGreater(a["weight_drift"],0)
        self.assertEqual(b["weight_drift"],0)
        self.assertEqual(a["predictor_memory_retained"],0)
        self.assertEqual(solve(g,tasks(3100,0,EPISODES),9182,
                               plastic=True),a)
        fresh=Learner.fresh(g,(5,5))
        self.assertEqual(fresh.weights,list(g))
        self.assertEqual(fresh.hidden,[0.]*8)
        self.assertEqual(fresh.predictor.learned_transitions,0)

    def test_equal_offspring_slots_and_novel_task_scope(self):
        outcomes={arm:evolve(3100,arm,generations=2) for arm in ARMS}
        for arm,result in outcomes.items():
            self.assertEqual(len(result["history"]),2)
            self.assertTrue(all(g["offspring_slots"]==POP
                                for g in result["history"]))
            self.assertTrue(all(g["inherited_only_genetic_initial_weights"]
                                for g in result["history"]))
            self.assertEqual(result["transferred_episodic_memory"],0)
            self.assertTrue(0<=result["novel_attempt_success"]<=1)

    def test_holdout_and_separate_cohort_guards(self):
        for seed in (3000,3150,161000,162000,163000):
            with self.assertRaises(ValueError):
                founders(seed)
        for seed in (3100,3000,161000,162000):
            with self.assertRaises(ValueError):
                tasks(seed,777,5,novel=True)
        with self.assertRaises(ValueError):
            experiment((3100,3100))
        with self.assertRaises(ValueError):
            evolve(3100,"solver_plastic",generations=50)


if __name__=="__main__":
    unittest.main()
