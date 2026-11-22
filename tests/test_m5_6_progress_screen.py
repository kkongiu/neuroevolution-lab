"""M5.6 progress-screen safety and repeatability."""
import unittest
from neuroevolution.m5_6_progress_screen import ARMS, execute


class ProgressScreenTests(unittest.TestCase):
    def test_one_pair_retention_protocol_is_identical_until_cutoff(self):
        r = execute((1080,))
        self.assertEqual(r["physical_trials"],len(ARMS))
        self.assertEqual(r["reward_during_evaluation"],0)
        self.assertEqual(r["policy_updates_during_evaluation"],0)
        by={x["condition"]:x for x in r["raw_rows"]}
        a=by["progress_retained"]
        z=by["progress_reset_synapses"]
        for key in ("survived_at_cutoff","predictor_reward_total",
                    "predictor_training_transitions","predictor_reward_events",
                    "novelty_reward_total"):
            self.assertEqual(a[key],z[key])
        self.assertEqual(a["post_cutoff_weight_updates"],0)
        self.assertEqual(z["post_cutoff_weight_updates"],0)
        self.assertLess(abs(a["energy_residual"]),1e-6)

    def test_fails_closed_for_all_other_worlds(self):
        for seed in (1001,1032,1056,161000,162000,163000):
            with self.assertRaises(ValueError): execute((seed,))
        with self.assertRaises(ValueError): execute((1080,1080))

    def test_reproducibility(self):
        self.assertEqual(execute((1080,)),execute((1080,)))


if __name__=="__main__":
    unittest.main()
