"""Calibrated comparison: no outcome leakage, complete paired controls."""
import unittest

from neuroevolution.m5_6_calibrated_curiosity import ARMS, execute


class CalibratedCuriosityTests(unittest.TestCase):
    def test_development_only_complete_paired_grid_and_energy(self):
        report = execute(seeds=(1032, 1033))
        self.assertEqual(report["physical_trials"], 2 * len(ARMS))
        self.assertEqual(report["reserved_holdout_worlds_used"], [])
        self.assertEqual(report["calibration_worlds_used"], [])
        self.assertEqual(report["status"], "development_exploratory_not_heldout")
        self.assertEqual(len(report["summary"]), len(ARMS))
        self.assertEqual(report["summary"]["no_curiosity"]
                         ["paired_world_survival_difference_vs_no_curiosity"], 0)
        self.assertTrue(all(abs(r["energy_residual"]) < 1e-6
                            for r in report["raw_rows"]))
        self.assertEqual(report, execute(seeds=(1032, 1033)))

    def test_does_not_allow_calibration_or_sealed_world_seeds(self):
        for seed in (1001, 1020, 161000, 162000, 163000):
            with self.assertRaises(ValueError):
                execute((seed,))
        with self.assertRaises(ValueError):
            execute((1032, 1032))


if __name__ == "__main__":
    unittest.main()
