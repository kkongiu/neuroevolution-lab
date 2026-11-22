"""M5.8: deterministic baseline calibration and untouched-world robustness."""
import unittest

from neuroevolution.m5_8_robustness import (
    PRESSURES, calibration, evaluate, physical_run,
)
from neuroevolution.m5_8_worlds import make_world


class RobustnessTests(unittest.TestCase):
    def test_calibration_baseline_only_and_reproducible(self):
        a=calibration((2000,2001,2002))
        self.assertEqual(a,calibration((2000,2001,2002)))
        self.assertIn(a["chosen_pressure"],PRESSURES)
        self.assertEqual(len(a["raw_rows"]),6)
        self.assertEqual({r["model_decision_mode"] for r in a["raw_rows"]},
                         {"disabled"})
        self.assertEqual({r["seed"] for r in a["raw_rows"]},{2000,2001,2002})

    def test_separate_novel_development_seeds(self):
        for seed in (2064,2095):
            w=make_world(seed)
            self.assertEqual(w["seed"],seed)
        for seed in (1000,2096,161000,162000,163000):
            with self.assertRaises(ValueError): make_world(seed)
        with self.assertRaises(ValueError):calibration((2064,))
        with self.assertRaises(ValueError):evaluate(
            calibration_seeds=(2000,),
            check_seeds=(2000,))

    def test_pressure_really_modulates_predator_physics(self):
        full=physical_run(2000,"progressive","disabled","full")
        moderate=physical_run(2000,"progressive","disabled","moderate")
        off=physical_run(2000,"progressive","disabled","none")
        self.assertGreaterEqual(full["predator_extra_steps"],
                                moderate["predator_extra_steps"])
        self.assertEqual(off["predator_extra_steps"],0)
        for row in (full,moderate,off):
            self.assertLess(abs(row["energy_residual"]),1e-6)

    def test_three_arm_novel_world_pairing(self):
        report=evaluate(calibration_seeds=(2000,2001),
                        check_seeds=(2064,))
        self.assertEqual(report["physical_trials"],9)
        self.assertTrue(report["no_cross_world_synaptic_transfer"])
        self.assertEqual(report["new_world_seeds"],[2064])
        self.assertEqual(set(report["world_level_results"]),
                         {"progressive","uniform","reversed"})
        self.assertEqual({r["seed"] for r in report["raw_rows"]},{2064})
        self.assertEqual(len(report["raw_rows"]),9)
        self.assertTrue(all(r["post_cutoff_weight_updates"]==0
                            for r in report["raw_rows"]))


if __name__=="__main__":
    unittest.main()
