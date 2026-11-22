"""M5.6 predictive progress: causal fixed-probe regression and integration."""
import unittest
from neuroevolution import m5_5b
from neuroevolution.m5_6_prediction import LocalPredictor, PROBE_SIZE
from neuroevolution.m5_6_online import simulate_development


class ProgressTests(unittest.TestCase):
    def test_held_aside_probe_not_optimized_as_gradient_data(self):
        p = LocalPredictor()
        before = tuple(tuple(row) for row in p.weights)
        obs = (0.,)*19
        after = (1.,)*19
        for i in range(PROBE_SIZE):
            self.assertEqual(p.transition(obs, 0, after), 0.)
        self.assertEqual(p.learned_transitions, 0)
        self.assertEqual(tuple(tuple(row) for row in p.weights), before)
        original_probe = list(p.probes)
        loss_before = p.probe_loss()
        gain = p.transition(obs, 0, after)
        self.assertGreater(gain, 0.)
        self.assertLessEqual(gain, .1)
        self.assertEqual(p.probes, original_probe)
        self.assertEqual(p.learned_transitions, 1)
        self.assertLess(p.probe_loss(), loss_before)

    def test_malformed_observations_do_not_create_progress(self):
        p = LocalPredictor()
        with self.assertRaises(ValueError):
            p.transition((0.,)*18, 0, (0.,)*19)
        with self.assertRaises(ValueError):
            p.transition((0.,)*19, 4, (0.,)*19)
        self.assertEqual(p.learned_transitions, 0)

    def test_real_ecology_progress_reproducible_and_energy_conserved(self):
        s = m5_5b.Settings(
            seed_start=300, seeds=1, shifted_start=400, shifted_seeds=1,
            ticks=12, size=11, initial_food=9, food_cap=14,
            initial_population=8, max_population=18,
            checkpoint_every=4, no_food_controls=0)
        first = simulate_development(1080, s, "learning_progress",
                                     curiosity_gain=4, predator_count=1)
        self.assertEqual(first, simulate_development(
            1080, s, "learning_progress", curiosity_gain=4, predator_count=1))
        self.assertLess(abs(first["energy_residual"]), 1e-6)
        self.assertGreaterEqual(first["predictor_training_transitions"], 0)
        self.assertGreaterEqual(first["predictor_reward_events"], 0)
        with self.assertRaises(ValueError):
            simulate_development(162000, s, "learning_progress")


if __name__ == "__main__":
    unittest.main()
