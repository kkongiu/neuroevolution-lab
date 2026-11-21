import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.plasticity import Network
from neuroevolution.meta import (
    ARMS, MetaSettings, Policy, features, fitness, prepare, run,
    simulate, summarize,
)


class MetaTests(unittest.TestCase):
    def setUp(self):
        self.config = MetaSettings(
            train_start=14, train_seeds=2,
            validation_start=100, validation_seeds=1,
            test_start=200, test_seeds=2,
            tasks=3, steps=12, batch_size=2, width=6,
            eval_size=16, report_every=4,
            generations=2, population=4,
        )

    def test_invalid_settings_and_seed_leakage(self):
        with self.assertRaises(ValueError):
            MetaSettings(train_start=2000, train_seeds=5,
                         validation_start=2000).validate()
        with self.assertRaises(ValueError):
            MetaSettings(steps=13).validate()
        with self.assertRaises(ValueError):
            MetaSettings(tasks=1).validate()

    def test_actions_are_bounded_and_depend_on_observations(self):
        no = Policy((-5.0, 0.0, 0.0, 0.0, 0.0))
        yes = Policy((5.0, 0.0, 0.0, 0.0, 0.0))
        neutral = Policy((0.0, 0.0, 0.0, 0.0, 0.0))
        x = (1.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(no.action(x, 3), 0)
        self.assertEqual(yes.action(x, 3), 3)
        self.assertIn(neutral.action(x, 3), (1, 2))
        sensitivity = Policy((0.0, 2.0, 0.0, 0.0, 0.0))
        self.assertNotEqual(sensitivity.action((1., -1., 0., 0., 0.), 3),
                            sensitivity.action((1., 1., 0., 0., 0.), 3))
        with self.assertRaises(ValueError):
            no.action((1.,), 3)

    def test_features_only_require_training_examples(self):
        data = prepare(2, self.config)
        net = Network.create(random.Random(2), self.config.width)
        x = features(net, data[1]["train"], data[0]["train"], 1, self.config)
        self.assertEqual(len(x), 5)
        self.assertTrue(all(-1 <= v <= 1 for v in x))
        self.assertEqual(x, features(net, data[1]["train"],
                                     data[0]["train"], 1, self.config))

    def test_equal_renewal_count_for_matched_arms(self):
        seed = 200
        data = prepare(seed, self.config)
        learned = simulate(seed, self.config, data, "learned",
                           Policy((3.0, 0, 0, 0, 0)))
        same = simulate(seed, self.config, data, "random_matched",
                        plan=learned["actions"])
        perm = list(reversed(learned["actions"]))
        shuffled = simulate(seed, self.config, data, "shuffled_matched",
                            plan=perm)
        self.assertEqual(learned["total_resets"], same["total_resets"])
        self.assertEqual(learned["total_resets"], shuffled["total_resets"])
        self.assertEqual(learned["actions"], [self.config.max_units] * 2)
        self.assertEqual(learned["rows"][0]["resets"], 0)
        with self.assertRaises(ValueError):
            simulate(seed, self.config, data, "random_matched", plan=None)

    def test_zero_renewal_equals_continual(self):
        seed = 201
        data = prepare(seed, self.config)
        learned = simulate(seed, self.config, data, "learned",
                           Policy((-3.0, 0, 0, 0, 0)))
        continual = simulate(seed, self.config, data, "continual")
        self.assertEqual(learned["auc"], continual["auc"])
        self.assertEqual(learned["end_accuracy"], continual["end_accuracy"])
        self.assertEqual(learned["total_resets"], 0)
        self.assertEqual(learned["actions"], [0, 0])

    def test_full_smoke_is_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            out_a, out_b = Path(folder) / "a", Path(folder) / "b"
            result_a = run(self.config, out_a)
            result_b = run(self.config, out_b)
            self.assertEqual(result_a, result_b)
            self.assertEqual((out_a / "task_metrics.csv").read_bytes(),
                             (out_b / "task_metrics.csv").read_bytes())
            self.assertTrue((out_a / "auc.svg").exists())
            self.assertTrue((out_a / "meta_training.jsonl").exists())
            with (out_a / "test_seed_metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), self.config.test_seeds * len(ARMS))
            self.assertEqual(set(result_a["test_arm_means"]), set(ARMS))
            for seed in range(self.config.test_start,
                              self.config.test_start + self.config.test_seeds):
                for arm in ("random_matched", "shuffled_matched"):
                    row = next(r for r in rows if int(r["seed"]) == seed
                               and r["arm"] == arm)
                    ref = next(r for r in rows if int(r["seed"]) == seed
                               and r["arm"] == "learned")
                    self.assertEqual(row["total_resets"], ref["total_resets"])
            self.assertIn("learned_minus_shuffled_matched_auc",
                          result_a["test_paired_auc_comparisons"])


if __name__ == "__main__":
    unittest.main()
