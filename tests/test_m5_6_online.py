"""M5.6 development neural intrinsic reward x physical ecology."""
import random
import unittest

from neuroevolution import m4_2, m4_4a, m5_5b
from neuroevolution.m5_6_online import Learner, simulate_development


class CuriosityOnlineTests(unittest.TestCase):
    def setUp(self):
        self.g = tuple(m4_4a._initial_genome(
            124000, "rnn", m4_2.Settings()))

    def test_food_or_novelty_changes_real_synapses(self):
        agent = Learner.fresh(self.g, (1, 1))
        obs = (0.,) * 19
        action, cache = agent.action(obs, (0, 1, 2, 3), random.Random(3))
        self.assertIn(action, range(4))
        self.assertGreater(agent.learn(cache, 0.1), 0.)
        self.assertNotEqual(tuple(agent.weights), self.g)
        self.assertGreater(agent.update_count, 0)
        self.assertEqual(agent.curiosity.visits, {(1, 1): 1})

    def test_zero_reward_and_frozen_retain_initial_synapses(self):
        agent = Learner.fresh(self.g, (1, 1))
        _, cache = agent.action((0.,) * 19, (0, 1), random.Random(9))
        self.assertEqual(agent.learn(cache, 0), 0)
        self.assertEqual(agent.learn(cache, 0.1, enabled=False), 0)
        self.assertEqual(tuple(agent.weights), self.g)

    def test_local_recurrent_weights_get_credit(self):
        agent = Learner.fresh(self.g, (0, 0))
        agent.hidden = [0.2] * 8
        obs = tuple(0.5 for _ in range(19))
        _, cache = agent.action(obs, (0, 1, 2, 3), random.Random(2))
        before = agent.weights[:]
        agent.learn(cache, 0.5)
        self.assertGreater(sum(abs(agent.weights[i] - before[i])
                               for i in range(19, 27)), 0)
        self.assertGreater(sum(abs(agent.weights[i] - before[i])
                               for i in range(0, 19)), 0)
        self.assertGreater(sum(abs(agent.weights[i] - before[i])
                               for i in range(224, 260)), 0)

    def test_development_ecology_reproducible_energy_conserved(self):
        settings = m5_5b.Settings(
            seed_start=300, seeds=1, shifted_start=400, shifted_seeds=1,
            ticks=12, size=11, initial_food=9, food_cap=14,
            initial_population=8, max_population=18,
            checkpoint_every=4, no_food_controls=0)
        results = {}
        for arm in ("novelty", "no_novelty", "frozen",
                    "shuffled_novelty"):
            result = simulate_development(1001, settings, arm,
                                          predator_count=1)
            self.assertAlmostEqual(result["energy_residual"], 0., places=6)
            self.assertEqual(result, simulate_development(
                1001, settings, arm, predator_count=1))
            self.assertEqual(result["claim"],
                             "development only; no held-out curiosity benefit demonstrated")
            results[arm] = result
        self.assertEqual(results["no_novelty"]["novelty_reward_total"], 0.)
        self.assertEqual(results["frozen"]["novelty_reward_total"], 0.)
        self.assertEqual(results["frozen"]["weights_changed"], 0)
        self.assertGreater(results["novelty"]["novelty_reward_total"], 0.)

    def test_heldout_worlds_are_blocked(self):
        with self.assertRaises(ValueError):
            simulate_development(122000, m5_5b.Settings(), "novelty")


if __name__ == "__main__":
    unittest.main()
