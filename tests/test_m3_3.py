import copy
import csv
import math
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.m3_3 import (
    ARMS, MATCHED, ContextNetwork, Policy, Settings,
    acc, features, mask, run, simulate, uniform_schedule, world,
)


class ContextualMemoryTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            train_start=20, train_seeds=2,
            validation_start=100, validation_seeds=1,
            test_start=200, test_seeds=2,
            shifted_start=300, shifted_seeds=2,
            tasks=6, steps=8, batch_size=2, width=6,
            eval_size=12, report_every=2,
            buffer_size=4, replay_every=2,
            renew_units=1, quota=2, generations=2, population=4,
        )

    def test_disjoint_seed_partitions(self):
        self.settings.validate()
        with self.assertRaises(ValueError):
            Settings(train_start=100, train_seeds=3,
                     validation_start=101).validate()
        with self.assertRaises(ValueError):
            Settings(test_start=300, shifted_start=300).validate()
        with self.assertRaises(ValueError):
            Settings(tasks=6, quota=5).validate()
        with self.assertRaises(ValueError):
            Settings(width=4, renew_units=4).validate()

    def test_exact_inverse_labels_with_identical_xy(self):
        for seed in (201, 202, 203):
            environment = world(seed, self.settings)
            self.assertEqual(set(environment["panel"]), {0, 1, 2, 3})
            for a, b in ((0, 1), (2, 3)):
                pa, pb = environment["panel"][a], environment["panel"][b]
                self.assertEqual(len(pa), self.settings.eval_size)
                for (xy1, y1), (xy2, y2) in zip(pa, pb):
                    self.assertEqual(xy1[:2], xy2[:2])
                    self.assertNotEqual(xy1[2:], xy2[2:])
                    self.assertEqual(y1, 1 - y2)

    def test_mathematical_blind_ceiling_exactly_half(self):
        env = world(201, self.settings)
        net = ContextNetwork.create(201, self.settings)
        masked = [mask(env["panel"][ctx], True) for ctx in range(4)]
        self.assertAlmostEqual(
            sum(acc(net, panel) for panel in masked) / 4, .5, places=13
        )
        # Crucially, test AFTER actual sequential training as well.
        result = simulate(201, self.settings, env, "blind_replay_no_reset")
        self.assertAlmostEqual(result["final_balanced_accuracy"], .5, places=13)

    def test_first_four_contexts_all_present_and_shifted_family(self):
        a = world(204, self.settings)
        self.assertEqual(len({r["context"] for r in a["episodes"][:4]}), 4)
        self.assertEqual([r["context"] for r in a["episodes"]],
                         [r["context"] for r in world(204, self.settings)["episodes"]])
        shifted = world(305, self.settings, shifted=True)
        self.assertEqual(len(shifted["panel"]), 4)
        for ctx in (0, 1, 2, 3):
            self.assertEqual(len(shifted["panel"][ctx]), self.settings.eval_size)

    def test_training_only_state_features(self):
        data = world(201, self.settings)
        net = ContextNetwork.create(201, self.settings)
        current = data["episodes"][1]["train"]
        previous = data["episodes"][0]["train"]
        buffer = {data["episodes"][0]["context"]: previous[:4]}
        first = features(net, current, buffer, previous, 1, self.settings)
        changed = copy.deepcopy(data)
        changed["panel"] = {k: [] for k in changed["panel"]}
        again = features(net, changed["episodes"][1]["train"], buffer,
                         changed["episodes"][0]["train"], 1, self.settings)
        self.assertEqual(first, again)
        self.assertEqual(len(first), 5)

    def test_policy_state_ablations(self):
        p = Policy((0., 1., -2., 3., 0.))
        a = (1., .2, .2, .9, .2)
        b = (1., .9, .9, .1, .2)
        self.assertEqual(p.decide(a, "time_only"), p.decide(b, "time_only"))
        self.assertEqual(p.decide(a, "accuracy_only"), p.decide(b, "accuracy_only"))
        self.assertNotEqual(p.decide(a, "adaptive"), p.decide(b, "adaptive"))

    def test_budget_and_replay_changes_training_mix(self):
        seed = 201
        data = world(seed, self.settings)
        policy = Policy((-3., 0., 0., 0., 0.))
        adaptive = simulate(seed, self.settings, data, "adaptive", policy)
        self.assertEqual(adaptive["renewal_actions"], self.settings.quota)
        self.assertEqual(adaptive["renewed_units"], self.settings.quota)
        self.assertGreaterEqual(adaptive["forced"], 1)
        plan = uniform_schedule(self.settings)
        self.assertEqual(sum(plan), self.settings.quota)
        same = simulate(seed, self.settings, data, "uniform", plan=plan)
        shuffled = list(adaptive["actions"])
        random.Random(seed * 417 + 17).shuffle(shuffled)
        permuted = simulate(seed, self.settings, data, "shuffled", plan=shuffled)
        random_units = simulate(seed, self.settings, data,
                                "random_units", plan=adaptive["actions"])
        for item in (same, permuted, random_units):
            self.assertEqual(item["renewal_actions"], self.settings.quota)
            self.assertAlmostEqual(item["rows"][0]["learning_auc"],
                                   adaptive["rows"][0]["learning_auc"])
        with_replay = simulate(seed, self.settings, data, "cue_replay_no_reset")
        without_replay = simulate(seed, self.settings, data, "cue_no_replay_no_reset")
        self.assertGreater(with_replay["replay_batches"], 0)
        self.assertEqual(without_replay["replay_batches"], 0)

    def test_full_campaign_reproducible_except_clock(self):
        with tempfile.TemporaryDirectory() as path:
            a, b = Path(path) / "a", Path(path) / "b"
            first, second = run(self.settings, a), run(self.settings, b)
            first.pop("wall_seconds_including_meta_search")
            second.pop("wall_seconds_including_meta_search")
            self.assertEqual(first, second)
            for filename in ("seed_metrics.csv", "episode_metrics.csv"):
                self.assertEqual((a / filename).read_bytes(), (b / filename).read_bytes())
            with (a / "seed_metrics.csv").open(newline="") as file:
                records = list(csv.DictReader(file))
            self.assertEqual(len(records),
                             (self.settings.test_seeds + self.settings.shifted_seeds) *
                             len(ARMS))
            self.assertTrue(first["main"]["blind_accuracy_50_percent_for_all_seeds"])
            for row in records:
                if row["arm"] in MATCHED:
                    self.assertEqual(int(row["renewal_actions"]), self.settings.quota)
            self.assertIn("adaptive_vs_time_only",
                          first["main"]["comparisons"])


if __name__ == "__main__":
    unittest.main()
