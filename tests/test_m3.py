import csv
import tempfile
import unittest
from pathlib import Path

from neuroevolution.meta import Policy
from neuroevolution.m3 import ARMS, MATCHED, M3Settings, evaluate_test, prepare_world, restricted, run


class M3Tests(unittest.TestCase):
    def setUp(self):
        self.settings = M3Settings(
            train_start=21, train_seeds=2,
            validation_start=100, validation_seeds=1,
            test_start=200, test_seeds=2,
            shifted_start=300, shifted_seeds=2,
            tasks=8, steps=12, batch_size=2, width=6,
            eval_size=16, report_every=4,
            generations=2, population=4,
        )

    def test_seed_partitions_cannot_overlap(self):
        self.settings.validate()
        with self.assertRaises(ValueError):
            M3Settings(train_start=90, train_seeds=20, validation_start=100).validate()
        with self.assertRaises(ValueError):
            M3Settings(shifted_start=100, validation_start=100).validate()
        with self.assertRaises(ValueError):
            M3Settings(tasks=7).validate()

    def test_task_recurrence_has_same_latent_world(self):
        found_return = found_novel = False
        for seed in range(20, 40):
            data = prepare_world(seed, self.settings)
            earlier_ids = set()
            for task in data:
                if task["status"] == "return":
                    found_return = True
                    self.assertIn(task["task_id"], earlier_ids)
                else:
                    found_novel = True
                    self.assertNotIn(task["task_id"], earlier_ids)
                earlier_ids.add(task["task_id"])
        self.assertTrue(found_return)
        self.assertTrue(found_novel)

    def test_new_shifted_task_families_exist(self):
        encountered = set()
        for seed in range(300, 330):
            seen = prepare_world(seed, self.settings, shifted=True)
            encountered.update(task["kind"] for task in seen)
        self.assertIn("checker", encountered)
        self.assertIn("wedge", encountered)

    def test_world_reproduction_and_heldout_separation(self):
        a = prepare_world(204, self.settings)
        b = prepare_world(204, self.settings)
        self.assertEqual(a, b)
        for task in a:
            self.assertNotEqual(task["train"], task["eval"])

    def test_time_only_blinds_state(self):
        policy = Policy((.1, 1., -2., 3., .4))
        blind = restricted(policy, True)
        self.assertEqual(blind.weights, (.1, 0., 0., 0., .4))

    def test_matched_total_budget_and_causal_blinding(self):
        results = evaluate_test(
            self.settings.test_start, self.settings,
            Policy((.1, 1., -2., 3., .4)),
            Policy((.1, 0., 0., 0., .4)), shifted=False,
        )
        self.assertEqual({r["arm"] for r in results}, set(ARMS))
        self.assertTrue(all(r["total_resets"] == self.settings.budget
                            for r in results if r["arm"] in MATCHED))
        self.assertTrue(all(row["status"] in ("novel", "return")
                            for result in results for row in result["rows"]))
        self.assertEqual(results[0]["rows"][0]["auc"], results[1]["rows"][0]["auc"])

    def test_complete_smoke_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "a"
            b = Path(directory) / "b"
            summary_a = run(self.settings, a)
            summary_b = run(self.settings, b)
            self.assertEqual(summary_a, summary_b)
            self.assertEqual((a / "summary.json").read_bytes(),
                             (b / "summary.json").read_bytes())
            self.assertEqual((a / "task_metrics.csv").read_bytes(),
                             (b / "task_metrics.csv").read_bytes())
            with (a / "test_seed_metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                len(rows),
                (self.settings.test_seeds + self.settings.shifted_seeds) * len(ARMS))
            self.assertIn("adaptive_minus_time_only",
                          summary_a["shifted"]["comparisons"])
            self.assertEqual(summary_a["regular"]["opportunities"],
                             self.settings.test_seeds * (self.settings.tasks - 1))


if __name__ == "__main__":
    unittest.main()
