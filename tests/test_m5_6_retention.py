"""Development-only M5.6 memory-retention interventions."""
import unittest

from neuroevolution.m5_6_retention import ARMS, execute


class RetentionTests(unittest.TestCase):
    def test_retained_and_reset_pre_cutoff_matching(self):
        report = execute((1056, 1057))
        self.assertEqual(report["physical_trials"], 2*len(ARMS))
        self.assertEqual(report["novelty_reward_during_evaluation"], 0)
        self.assertTrue(report["hidden_memory_reset_for_both"])
        self.assertTrue(all(r["post_cutoff_weight_updates"] == 0
                            for r in report["raw_rows"]))
        index = {(r["seed"], r["condition"]): r for r in report["raw_rows"]}
        for seed in (1056, 1057):
            retained = index[seed, "trained_curiosity_retained"]
            reset = index[seed, "trained_curiosity_reset_synapses"]
            self.assertEqual(retained["survived_at_cutoff"],
                             reset["survived_at_cutoff"])
            self.assertEqual(retained["post_cutoff_weight_updates"], 0)
            self.assertTrue(abs(retained["energy_residual"]) < 1e-6)
            self.assertTrue(abs(reset["energy_residual"]) < 1e-6)

    def test_only_fresh_development_seeds(self):
        for seed in (1000, 1032, 161000, 162000, 163000):
            with self.assertRaises(ValueError):
                execute((seed,))
        with self.assertRaises(ValueError):
            execute((1056, 1056))

    def test_reproducible(self):
        self.assertEqual(execute((1056,)), execute((1056,)))


if __name__ == "__main__":
    unittest.main()
