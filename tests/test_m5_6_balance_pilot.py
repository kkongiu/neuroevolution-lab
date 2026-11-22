"""Paired development-only balance experiment invariants."""
import unittest
from neuroevolution import m5_5b
from neuroevolution.m5_6_balance_pilot import evaluate, POLICIES
from neuroevolution.m5_6_online import simulate_development


class BalancePilotTests(unittest.TestCase):
    def test_matched_grid_and_zero_curiosity_baseline(self):
        ans = evaluate(seeds=(1001, 1002), pressures=(0, 1))
        self.assertEqual(ans["status"], "development_only_not_confirmatory")
        self.assertEqual(ans["worlds"], 2 * 2 * len(POLICIES))
        self.assertEqual(len(ans["rows_sha256"]), 64)
        for pressure in ("0", "1"):
            stats = ans["by_pressure"][pressure]
            self.assertEqual(stats["no_curiosity"]["paired_survival_diff_vs_none"], 0.)
            for label, *_ in POLICIES:
                self.assertEqual(stats[label]["worlds_total"], 2)
                self.assertLessEqual(abs(
                    stats[label]["paired_survival_diff_vs_none"]), 1.)
        self.assertTrue(all(abs(r["energy_residual"]) < 1e-6
                            for r in ans["rows"]))

    def test_level_control_changes_only_reward_not_world_rules(self):
        s = m5_5b.Settings(
            seed_start=300, seeds=1, shifted_start=400, shifted_seeds=1,
            ticks=12, size=11, initial_food=9, food_cap=14,
            initial_population=8, max_population=18, checkpoint_every=4,
            no_food_controls=0)
        zero = simulate_development(1001, s, "novelty",
                                    curiosity_gain=0, predator_count=1)
        ordinary = simulate_development(1001, s, "no_novelty",
                                        predator_count=1)
        self.assertEqual(zero["survived"], ordinary["survived"])
        self.assertEqual(zero["food_eaten"], ordinary["food_eaten"])
        self.assertEqual(zero["predation_kills"], ordinary["predation_kills"])
        self.assertEqual(zero["population"], ordinary["population"])
        self.assertEqual(zero["novelty_reward_total"], 0.)
        with self.assertRaises(ValueError):
            simulate_development(1001, s, "novelty", curiosity_gain=5)
        with self.assertRaises(ValueError):
            evaluate(seeds=(122000,))


if __name__ == "__main__":
    unittest.main()
