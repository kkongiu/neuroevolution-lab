"""Development-only M5.6 calibration fences and basic behavior."""
import unittest
from neuroevolution.m5_6_difficulty import (
    calibrate, CANDIDATES, SEEDS, PREDATORS,
)


class DifficultyTests(unittest.TestCase):
    def test_calibrates_only_no_curiosity(self):
        result = calibrate(seeds=(1020, 1021))
        self.assertEqual(result["num_runs"], 2 * len(CANDIDATES))
        self.assertEqual(result["status"], "development_calibration_only")
        self.assertEqual(result["predators"], 1)
        self.assertEqual(result["selection_policy"],
                         "no_curiosity_only_survival_nearest_0.5")
        self.assertTrue(all(row["arm"] == "no_novelty"
                            for row in result["raw_rows"]))
        self.assertTrue(all(abs(row["energy_residual"]) < 1e-6
                            for row in result["raw_rows"]))
        self.assertEqual(result, calibrate(seeds=(1020, 1021)))

    def test_final_cohorts_blocked(self):
        with self.assertRaises(ValueError):
            calibrate(seeds=(162000,))
        with self.assertRaises(ValueError):
            calibrate(seeds=(1020, 1020))


if __name__ == "__main__":
    unittest.main()
