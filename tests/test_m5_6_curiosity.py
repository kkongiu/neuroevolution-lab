"""Unit tests for isolated M5.6 curiosity reward signals."""
import unittest

from neuroevolution.m5_6_curiosity import IntrinsicReward


class CuriosityRewardTests(unittest.TestCase):
    def test_novelty_is_individual_and_decays_on_revisit(self):
        a, b = IntrinsicReward(), IntrinsicReward()
        first = a.observe_arrival((2, 3))
        second = a.observe_arrival((2, 3))
        self.assertGreater(first, second)
        self.assertEqual(first, b.observe_arrival((2, 3)))
        self.assertEqual(a.visits[(2, 3)], 2)

    def test_new_location_is_new_and_reward_does_not_add_energy(self):
        reward = IntrinsicReward()
        energy = 5
        self.assertEqual(reward.observe_arrival((0, 0)),
                         reward.observe_arrival((1, 0)))
        self.assertEqual(energy, 5)

    def test_learning_progress_rejects_unlearnable_surprise(self):
        reward = IntrinsicReward()
        self.assertEqual(reward.learning_progress(3, 3), 0)
        self.assertEqual(reward.learning_progress(3, 4), 0)
        self.assertGreater(reward.learning_progress(3, 2), 0)

    def test_rewards_are_capped(self):
        reward = IntrinsicReward(novelty_scale=100, progress_scale=100, maximum=.2)
        self.assertEqual(reward.observe_arrival((0, 0)), .2)
        self.assertEqual(reward.learning_progress(10, 0), .2)

    def test_invalid_values_rejected(self):
        for x in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                IntrinsicReward(maximum=x)
            with self.assertRaises(ValueError):
                IntrinsicReward().learning_progress(x, 0)
        with self.assertRaises(ValueError):
            IntrinsicReward().observe_arrival((1.0, 2))
