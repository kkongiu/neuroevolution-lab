import copy
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.m3_2 import (
    ARMS, MATCHED, Policy, Settings, evenly_spaced, make_tasks, observe,
    output_gradient, run, simulate,
)
from neuroevolution.plasticity import Network


class NaturalInterferenceTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            train_start=20, train_seeds=2,
            validation_start=100, validation_seeds=1,
            test_start=200, test_seeds=2,
            shifted_start=300, shifted_seeds=2,
            tasks=6, steps=12, batch_size=2, width=6,
            eval_size=16, report_every=4,
            renew_units=1, quota=2, generations=2, population=4,
        )

    def test_disjoint_seed_splits_and_budget_constraints(self):
        self.settings.validate()
        with self.assertRaises(ValueError):
            Settings(train_start=100, train_seeds=4,
                     validation_start=102).validate()
        with self.assertRaises(ValueError):
            Settings(test_start=300, shifted_start=300).validate()
        with self.assertRaises(ValueError):
            Settings(tasks=4, quota=3).validate()
        with self.assertRaises(ValueError):
            Settings(width=4, renew_units=4).validate()

    def test_task_recurs_without_injected_damage(self):
        found = 0
        for seed in range(50, 60):
            tasks = make_tasks(seed, self.settings, shifted=False)
            self.assertEqual(len(tasks), self.settings.tasks)
            ids = set()
            for t in tasks:
                if t["status"] == "return":
                    found += 1
                    self.assertIn(t["task_id"], ids)
                else:
                    self.assertNotIn(t["task_id"], ids)
                ids.add(t["task_id"])
        self.assertGreater(found, 0)

    def test_shifted_has_new_families(self):
        kinds = {t["kind"] for seed in range(300, 315)
                 for t in make_tasks(seed, self.settings, shifted=True)}
        self.assertIn("checker", kinds)
        self.assertIn("wedge", kinds)

    def test_gradient_and_feature_leakage(self):
        data = make_tasks(201, self.settings, shifted=False)
        net = Network.create(random.Random(91), self.settings.width)
        train_grad = output_gradient(net, data[0]["train"][:16])
        self.assertEqual(len(train_grad), self.settings.width + 1)
        obs = observe(net, data[1]["train"], data[0]["train"], 1, self.settings)
        altered = copy.deepcopy(data)
        altered[0]["eval"] = []
        altered[1]["eval"] = []
        self.assertEqual(obs, observe(net, altered[1]["train"],
                                      altered[0]["train"], 1, self.settings))
        self.assertEqual(len(obs), 6)
        self.assertTrue(-1 <= obs[3] <= 1)

    def test_policy_masks_are_real_controls(self):
        p = Policy((-1., 3., -4., 4., 0., 0.))
        a = (1., .9, .1, 1., 0., .4)
        b = (1., .9, .9, -1., 0., .4)
        self.assertEqual(p.decide(a, "accuracy_only"),
                         p.decide(b, "accuracy_only"))
        self.assertEqual(p.decide(a, "time_only"),
                         p.decide(b, "time_only"))
        self.assertNotEqual(p.decide(a, "adaptive"), p.decide(b, "adaptive"))

    def test_equal_dose_and_counterfactual_consistency(self):
        seed = 200
        data = make_tasks(seed, self.settings, False)
        p = Policy((-2., 0., 0., 0., 0., 0.))
        learned = simulate(seed, self.settings, data, "adaptive", p, counterfactual=True)
        self.assertEqual(learned["renewal_actions"], self.settings.quota)
        self.assertGreaterEqual(learned["forced"], 1)
        self.assertEqual(len(learned["counterfactuals"]), self.settings.tasks-1)
        self.assertEqual(sum(r["forced"] for r in learned["counterfactuals"]),
                         learned["forced"])
        schedule = evenly_spaced(self.settings)
        self.assertEqual(sum(schedule), self.settings.quota)
        shuffled = list(learned["actions"])
        random.Random(seed * 417 + 17).shuffle(shuffled)
        for arm, plan in (("uniform", schedule), ("shuffled", shuffled),
                          ("random_units", learned["actions"])):
            control = simulate(seed, self.settings, data, arm, plan=plan)
            self.assertEqual(control["renewal_actions"], self.settings.quota)
            self.assertEqual(control["renewed_units"],
                             self.settings.quota * self.settings.renew_units)
            self.assertAlmostEqual(learned["rows"][0]["auc"],
                                   control["rows"][0]["auc"], places=12)
        with self.assertRaises(ValueError):
            simulate(seed, self.settings, data, "uniform", plan=[True, False])

    def test_full_replay_is_deterministic_except_wall_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a", Path(folder) / "b"
            x, y = run(self.settings, a), run(self.settings, b)
            x.pop("wall_seconds_including_instrumentation")
            y.pop("wall_seconds_including_instrumentation")
            self.assertEqual(x, y)
            for name in ("seed_metrics.csv", "task_metrics.csv",
                         "local_counterfactuals.csv"):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
            with (a / "seed_metrics.csv").open(newline="") as f:
                seed_rows = list(csv.DictReader(f))
            self.assertEqual(len(seed_rows),
                             (self.settings.test_seeds + self.settings.shifted_seeds) * len(ARMS))
            with (a / "local_counterfactuals.csv").open(newline="") as f:
                local = list(csv.DictReader(f))
            self.assertEqual(len(local),
                             (self.settings.test_seeds + self.settings.shifted_seeds) *
                             (self.settings.tasks - 1))
            self.assertIn("adaptive_minus_time_only", x["main"]["comparisons"])


if __name__ == "__main__":
    unittest.main()
