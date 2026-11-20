import csv
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.plasticity import (
    ARMS, Network, Settings, curve_auc, examples, make_task,
    paired_bootstrap, run, sigmoid,
)


class PlasticityTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(seed_start=5, seeds=2, tasks=3, steps=12,
                                 batch_size=2, width=6, eval_size=24,
                                 report_every=4, reset_fraction=0.25)

    def test_settings_reject_invalid_config(self):
        with self.assertRaises(ValueError):
            Settings(tasks=1).validate()
        with self.assertRaises(ValueError):
            Settings(steps=11, report_every=4).validate()
        with self.assertRaises(ValueError):
            Settings(width=1).validate()

    def test_task_generation_is_reproducible(self):
        self.assertEqual(make_task(5, 3), make_task(5, 3))
        task = make_task(7, 2)
        self.assertEqual(examples(task, 2, 50), examples(task, 2, 50))

    def test_stable_sigmoid(self):
        self.assertGreater(sigmoid(1000), 0.99)
        self.assertLess(sigmoid(-1000), 0.01)

    def test_gradient_learning_changes_predictions(self):
        network = Network.create(random.Random(12), 8)
        samples = [((1.0, 1.0), 1)] * 12
        before = network.predict((1.0, 1.0))
        for _ in range(100):
            network.train_batch(samples, 0.04)
        self.assertGreater(network.predict((1.0, 1.0)), before)

    def test_low_activity_and_random_renewal(self):
        baseline = Network.create(random.Random(2), 8)
        probes = [(0.2, -0.8), (0.4, 0.7)] * 4
        net1 = Network.create(random.Random(2), 8)
        net2 = Network.create(random.Random(2), 8)
        self.assertEqual(net1, baseline)
        ids1 = net1.renew(2, random.Random(10), probes, True)
        ids2 = net2.renew(2, random.Random(10), probes, True)
        self.assertEqual(ids1, ids2)
        self.assertEqual(net1, net2)
        self.assertEqual(len(ids1), 2)
        self.assertNotEqual(net1, baseline)
        ids_random = baseline.renew(2, random.Random(8), probes, False)
        self.assertEqual(len(ids_random), 2)
        self.assertEqual(len(set(ids_random)), 2)

    def test_auc_calculation(self):
        self.assertAlmostEqual(curve_auc([(0, 0), (5, 1), (10, 1)], 10), 0.75)

    def test_bootstrap_is_deterministic(self):
        self.assertEqual(paired_bootstrap([0.01, -0.02, 0.03], 18),
                         paired_bootstrap([0.01, -0.02, 0.03], 18))

    def test_campaign_is_deterministic_and_initially_equal(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a"
            b = Path(tmp) / "b"
            first = run(self.settings, a)
            second = run(self.settings, b)
            self.assertEqual(first, second)
            self.assertEqual((a / "measurements.csv").read_bytes(),
                             (b / "measurements.csv").read_bytes())
            self.assertTrue((a / "curves.svg").exists())
            with (a / "measurements.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), self.settings.seeds * self.settings.tasks * 4)
            for seed in range(self.settings.seed_start,
                              self.settings.seed_start + self.settings.seeds):
                first_task = [r for r in rows if int(r["seed"]) == seed
                              and int(r["task_index"]) == 0]
                self.assertEqual({r["arm"] for r in first_task}, set(ARMS))
                self.assertEqual(len({r["auc"] for r in first_task}), 1)
                self.assertEqual(len({r["end_accuracy"] for r in first_task}), 1)
                self.assertEqual({r["reset_units"] for r in first_task}, {"0"})
            self.assertTrue(all(0 <= row["mean_difference"] + 1 <= 2 for row in
                                first["comparisons"].values()))


if __name__ == "__main__":
    unittest.main()
