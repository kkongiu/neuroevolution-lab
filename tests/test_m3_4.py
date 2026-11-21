import copy
import csv
import math
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.m3_4 import (
    ARMS, ReplayMemory, Resolver, Settings, THRESHOLDS,
    balanced_pairwise_accuracy, run, select_threshold, task_world,
    trace_resolver, train_arm,
)


class LatentContextInferenceTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            train_start=30, train_seeds=2,
            validation_start=100, validation_seeds=1,
            test_start=200, test_seeds=2,
            few_start=300, few_seeds=2,
            shifted_start=400, shifted_seeds=2,
            tasks=8, steps=8, batch_size=2, width=6,
            eval_size=12, report_every=2,
            main_support=8, hard_support=4, hard_flips=1,
            prototype_size=12, buffer_size=4,
            replay_every=2, knn_k=3,
        )

    def test_seed_splits_and_settings(self):
        self.settings.validate()
        with self.assertRaises(ValueError):
            Settings(train_start=100, train_seeds=4,
                     validation_start=101).validate()
        with self.assertRaises(ValueError):
            Settings(test_start=300, few_start=300).validate()
        with self.assertRaises(ValueError):
            Settings(hard_flips=6, hard_support=6).validate()
        with self.assertRaises(ValueError):
            Settings(tasks=5).validate()

    def test_opposite_context_labels_and_paired_holdouts(self):
        for seed in (201, 202, 402):
            data = task_world(seed, self.settings,
                              "shifted" if seed == 402 else "main")
            self.assertEqual(len(data["episodes"]), self.settings.tasks)
            first_four = [e["latent"] for e in data["episodes"][:4]]
            self.assertEqual(set(first_four), {0, 1, 2, 3})
            for a, b in ((0, 1), (2, 3)):
                for (xa, ya), (xb, yb) in zip(data["panel"][a], data["panel"][b]):
                    self.assertEqual(xa, xb)
                    self.assertEqual(ya, 1 - yb)

    def test_support_training_holdout_disjoint_and_noise(self):
        main = task_world(201, self.settings, "main")
        hard = task_world(301, self.settings, "few")
        base = task_world(301, self.settings, "main")
        self.assertEqual(len(main["episodes"][0]["support"]), 8)
        self.assertEqual(len(hard["episodes"][0]["support"]), 4)
        # Independent sample streams; no final-test labels used in resolver.
        for world in (main, hard):
            for episode in world["episodes"]:
                train_xy = {x for x, _ in episode["training"]}
                self.assertTrue(all(x not in train_xy for x, _ in episode["support"]))
                self.assertTrue(all(x not in train_xy for x, _ in world["panel"][episode["latent"]]))
        self.assertNotEqual(
            hard["episodes"][0]["support"],
            base["episodes"][0]["support"],
        )

    def test_resolver_runtime_independent_of_hidden_truth_and_holdouts(self):
        data = task_world(202, self.settings, "main")
        threshold = THRESHOLDS[1]
        source = trace_resolver(data, self.settings, threshold)
        changed = copy.deepcopy(data)
        for item in changed["episodes"]:
            item["latent"] = (item["latent"] + 1) % 4
            item["novel"] = not item["novel"]
        changed["panel"] = {c: [] for c in range(4)}
        result = trace_resolver(changed, self.settings, threshold)
        self.assertEqual(source["episode_slots"], result["episode_slots"])
        self.assertEqual(source["final_slots"], result["final_slots"])
        self.assertEqual(source["shuffled_final_slots"], result["shuffled_final_slots"])

    def test_first_episode_slot_creation_and_bounded_memory(self):
        data = task_world(205, self.settings, "main")
        resolver = Resolver(self.settings, THRESHOLDS[0])
        self.assertEqual(resolver.infer(data["episodes"][0]["support"], allow_new=True), 0)
        resolver.observe_training(0, data["episodes"][0]["training"])
        self.assertEqual(len(resolver.prototypes[0]), self.settings.prototype_size)
        with self.assertRaises(ValueError):
            resolver.infer([], allow_new=False)
        with self.assertRaises(ValueError):
            Resolver(self.settings, .6).infer(
                data["episodes"][0]["support"], allow_new=False)
        memory = ReplayMemory(self.settings, seed=205, kind="reservoir")
        for i in range(4):
            memory.update(i, data["episodes"][i]["training"])
        self.assertEqual(memory.size, self.settings.buffer_size * 4)
        self.assertIsNone(memory.replay(0, 205, 1, 4, 2)
                          if len(memory.stored) == 1 else None)

    def test_inference_trace_deterministic_and_threshold_selected_without_test(self):
        a, b = select_threshold(self.settings)
        self.assertEqual(a, b)
        self.assertIn(a, THRESHOLDS)
        data = task_world(205, self.settings, "main")
        one = trace_resolver(data, self.settings, a)
        two = trace_resolver(data, self.settings, a)
        self.assertEqual(one, two)
        self.assertTrue(1 <= one["first_unique"] <= 4)
        self.assertTrue(0 <= one["pairwise_accuracy"] <= 1)
        altered = Settings(**{**self.settings.__dict__, "test_start": 210})
        self.assertEqual(select_threshold(altered)[0], a)

    def test_replay_and_same_network_support_intervention(self):
        seed = 204
        data = task_world(seed, self.settings, "main")
        trace = trace_resolver(data, self.settings, .68)
        results = {
            arm: train_arm(seed, self.settings, data, trace, arm)
            for arm in ARMS
        }
        for r in results.values():
            self.assertEqual(len(r["episodes"]), self.settings.tasks)
            self.assertLessEqual(r["memory_stored"], self.settings.contexts *
                                 self.settings.buffer_size)
        self.assertGreater(results["inferred_fifo"]["replay_updates"], 0)
        self.assertEqual(results["inferred_no_replay"]["replay_updates"], 0)
        self.assertAlmostEqual(results["no_context_replay"]["final_balanced_accuracy"],
                               .5, places=12)
        self.assertAlmostEqual(results["inferred_fifo"]["no_support_accuracy"],
                               .5, places=12)
        self.assertIsNotNone(results["inferred_fifo"]["shuffled_support_accuracy"])

    def test_smoke_deterministic_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            first, second = run(self.settings, a), run(self.settings, b)
            first.pop("wall_seconds_including_meta_search")
            second.pop("wall_seconds_including_meta_search")
            self.assertEqual(first, second)
            for filename in ("seed_metrics.csv", "episode_metrics.csv",
                             "inference_metrics.csv"):
                self.assertEqual((a / filename).read_bytes(), (b / filename).read_bytes())
            with (a / "seed_metrics.csv").open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 6 * len(ARMS))
            self.assertTrue(first["main"]["blind_50_percent_exact"])
            self.assertIn("support_vs_shuffled", first["main"]["comparisons"])


if __name__ == "__main__":
    unittest.main()
