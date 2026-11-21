import csv
import tempfile
import unittest
from pathlib import Path

from neuroevolution.meta import Policy, prepare
from neuroevolution.meta_budget import (
    ARMS, BudgetSettings, restricted, run, simulate_budget, uniform_plan,
)


class BudgetedTimingTests(unittest.TestCase):
    def setUp(self):
        self.settings = BudgetSettings(
            train_start=40, train_seeds=2,
            validation_start=90, validation_seeds=1,
            test_start=140, test_seeds=2,
            tasks=3, steps=12, batch_size=2, width=6,
            eval_size=16, report_every=4,
            generations=2, population=4,
        )

    def test_budget_validation(self):
        self.settings.validate()
        self.assertEqual(self.settings.budget, 1)
        self.assertEqual(self.settings.max_units, 1)
        with self.assertRaises(ValueError):
            BudgetSettings(budget_fraction=.001).validate()
        with self.assertRaises(ValueError):
            BudgetSettings(train_start=10, train_seeds=3,
                           validation_start=12).validate()

    def test_uniform_schedule_is_valid(self):
        schedule = uniform_plan(self.settings)
        self.assertEqual(len(schedule), self.settings.tasks - 1)
        self.assertEqual(sum(schedule), self.settings.budget)
        self.assertTrue(all(0 <= a <= self.settings.max_units for a in schedule))

    def test_budget_forcing_not_mistaken_for_decision(self):
        seed = self.settings.test_start
        data = prepare(seed, self.settings)
        wants_none = Policy((-3., 0., 0., 0., 0.))
        result = simulate_budget(seed, self.settings, data, "adaptive", wants_none)
        self.assertEqual(result["actions"], [0, 1])
        self.assertEqual(result["requested"], [0, 0])
        self.assertEqual(result["forced_actions"], 1)
        self.assertEqual(result["total_resets"], self.settings.budget)

    def test_state_blind_control_uses_no_hidden_state_weights(self):
        policy = Policy((.1, 2., 3., -1., .2))
        restricted_policy = restricted(policy, time_only=True)
        self.assertEqual(restricted_policy.weights, (.1, 0., 0., 0., .2))
        self.assertEqual(restricted(policy, time_only=False), policy)

    def test_invalid_matched_plan_is_rejected(self):
        data = prepare(self.settings.test_start, self.settings)
        with self.assertRaises(ValueError):
            simulate_budget(self.settings.test_start, self.settings, data,
                            "permuted", plan=[1, 1])
        with self.assertRaises(ValueError):
            simulate_budget(self.settings.test_start, self.settings, data,
                            "random_units")

    def test_dose_equality_and_determinism(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = run(self.settings, root / "a")
            b = run(self.settings, root / "b")
            self.assertEqual(a, b)
            self.assertEqual((root / "a" / "test_seed_metrics.csv").read_bytes(),
                             (root / "b" / "test_seed_metrics.csv").read_bytes())
            self.assertEqual(a["budget_units_per_sequence"], self.settings.budget)
            with (root / "a" / "test_seed_metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), self.settings.test_seeds * len(ARMS))
            for row in rows:
                if row["arm"] in ("adaptive", "time_only", "uniform",
                                  "permuted", "random_units"):
                    self.assertEqual(int(row["total_resets"]), self.settings.budget)
            self.assertIn("adaptive_minus_time_only",
                          a["paired_test_auc_differences"])


if __name__ == "__main__":
    unittest.main()
