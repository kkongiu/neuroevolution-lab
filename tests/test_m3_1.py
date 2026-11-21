import csv
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.m3_1 import (
    Settings, Rule, make_seed_episodes, influence, influential_ids,
    shock, renew, choose_policy, policy_score, predicted_actions,
    evaluate_split, run,
)
from neuroevolution.plasticity import Network, Task, examples


class CausalRepairTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            train_start=20, train_seeds=2,
            validation_start=100, validation_seeds=1,
            test_start=200, test_seeds=2,
            shifted_start=300, shifted_seeds=2,
            trials=2, width=6, reset_units=2,
            pretrain_steps=12, adaptation_steps=12, batch_size=2,
            probe_size=16, eval_size=20, report_every=4,
        )

    def test_no_overlapping_seeds_or_invalid_settings(self):
        self.settings.validate()
        with self.assertRaises(ValueError):
            Settings(train_start=100, train_seeds=4,
                     validation_start=102).validate()
        with self.assertRaises(ValueError):
            Settings(test_start=300, shifted_start=300).validate()
        with self.assertRaises(ValueError):
            Settings(width=4, reset_units=4).validate()
        with self.assertRaises(ValueError):
            Settings(adaptation_steps=13, report_every=4).validate()

    def test_two_contexts_per_pretrained_model(self):
        data = make_seed_episodes(201, self.settings)
        self.assertEqual(len(data), 2 * self.settings.trials)
        for trial in range(self.settings.trials):
            pair = [p for p in data if p["trial"] == trial]
            self.assertEqual({p["context"] for p in pair},
                             {"intact", "disrupted"})
            self.assertEqual({p["index"] for p in pair},
                             {trial * 2, trial * 2 + 1})
            for episode in pair:
                self.assertEqual(set(("preserve", "renew")),
                                 {"preserve", "renew"} & episode.keys())
                self.assertGreaterEqual(episode["probe_accuracy"], 0)
                self.assertLessEqual(episode["probe_accuracy"], 1)

    def test_influence_and_interventions_are_deterministic(self):
        task = Task("line", .2, .1, .1)
        probe = examples(task, 17, 20)
        net = Network.create(random.Random(11), 6)
        baseline = influential_ids(net, probe, 2)
        copy = Network.create(random.Random(11), 6)
        self.assertEqual(baseline, influential_ids(copy, probe, 2))
        bad = shock(net, probe, self.settings)
        self.assertEqual(bad, baseline)
        chosen = renew(net, probe, self.settings, seed=15)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(len(set(chosen)), 2)

    def test_policy_decisions_have_no_context_label_dependency(self):
        probe_item = {"probe_accuracy": 0.4, "index": 0, "context": "intact"}
        changed = {**probe_item, "context": "disrupted"}
        policy = Rule("threshold_accuracy", .7)
        self.assertEqual(policy.decide(probe_item, 4),
                         policy.decide(changed, 4))
        self.assertTrue(policy.decide(probe_item, 4))
        self.assertFalse(Rule("always_preserve", -1).decide(probe_item, 4))
        self.assertTrue(Rule("always_renew", 2).decide(probe_item, 4))
        self.assertFalse(Rule("threshold_clock", .0).decide(probe_item, 4))

    def test_paired_potential_outcome_and_oracle_ceiling(self):
        episode = make_seed_episodes(205, self.settings)
        assert episode
        options = [Rule("always_preserve", 0), Rule("always_renew", 0)]
        scores = [policy_score(episode, predicted_actions(p, episode))
                  for p in options]
        oracle_actions = [e["potential_difference"] > 0 for e in episode]
        oracle = policy_score(episode, oracle_actions)
        self.assertGreaterEqual(oracle + 1e-12, max(scores))

    def test_dose_matched_shuffle_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            result1 = run(self.settings, a)
            result2 = run(self.settings, b)
            self.assertEqual(result1, result2)
            self.assertEqual((a / "counterfactuals.csv").read_bytes(),
                             (b / "counterfactuals.csv").read_bytes())
            self.assertEqual((a / "seed_metrics.csv").read_bytes(),
                             (b / "seed_metrics.csv").read_bytes())
            with (a / "seed_metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows),
                             self.settings.test_seeds + self.settings.shifted_seeds)
            for row in rows:
                self.assertEqual(row["learned_renewals"],
                                 row["shuffled_renewals"])
            with (a / "counterfactuals.csv").open(newline="") as handle:
                detail = list(csv.DictReader(handle))
            self.assertEqual(len(detail), len(rows) * 2 * self.settings.trials)
            self.assertIn("learned_minus_dose_matched_shuffled",
                          result1["main"]["contrasts"])
            self.assertIn("calibration_train_only", result1)


if __name__ == "__main__":
    unittest.main()
