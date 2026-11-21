import copy
import csv
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.m3_5 import (
    ARMS, MATCHED, ENTROPY_GRID, MARGIN_GRID, Settings,
    decide_query, query_pool, resolve_seed, run, scheduled_queries,
    select_threshold, uncertainty_scores, _latent_rules,
)
from neuroevolution.m3_4 import Resolver, task_world


class ActiveInformationTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings(
            train_start=20, train_seeds=2,
            validation_start=100, validation_seeds=1,
            test_start=200, test_seeds=2,
            shifted_start=300, shifted_seeds=2,
            few_start=400, few_seeds=2,
            tasks=8, steps=8, batch_size=2, width=6,
            eval_size=12, report_every=2,
            main_support=6, hard_support=6, hard_flips=1,
            prototype_size=12, buffer_size=4, knn_k=3,
            replay_every=2, extra_per_request=2, request_quota=2,
        )

    def test_disjoint_seed_splits_and_budget_bounds(self):
        self.s.validate()
        with self.assertRaises(ValueError):
            Settings(train_start=100, train_seeds=3,
                     validation_start=102).validate()
        with self.assertRaises(ValueError):
            Settings(test_start=300, shifted_start=300).validate()
        with self.assertRaises(ValueError):
            Settings(tasks=8, request_quota=8).validate()
        with self.assertRaises(ValueError):
            Settings(extra_per_request=0).validate()

    def test_exact_query_budget_for_all_competitive_schedulers(self):
        seed = 202
        data = task_world(seed, self.s, "few")
        extra = query_pool(seed, self.s, data,
                           shifted=False, corrupt_extra=False)
        assert len(extra) == self.s.tasks
        for arm in MATCHED:
            threshold = .4 if arm == "margin_uncertainty" else .5
            trace = resolve_seed(seed, self.s, data, extra, arm, threshold)
            self.assertEqual(trace["query_events"], self.s.request_quota)
            self.assertEqual(trace["extra_labels"],
                             self.s.request_quota * self.s.extra_per_request)
            self.assertEqual(len(trace["log"]), self.s.tasks)
            self.assertEqual(sum(l["query"] for l in trace["log"]),
                             self.s.request_quota)
        no_query = resolve_seed(seed, self.s, data, extra, "no_query", 0.)
        all_query = resolve_seed(seed, self.s, data, extra, "query_all", 0.)
        self.assertEqual(no_query["extra_labels"], 0)
        self.assertEqual(all_query["extra_labels"],
                         self.s.tasks * self.s.extra_per_request)

    def test_schedules_are_reproducible_and_blind(self):
        period = scheduled_queries(204, self.s, "periodic")
        self.assertEqual(len(period), self.s.request_quota)
        self.assertEqual(len(scheduled_queries(204, self.s, "random")),
                         self.s.request_quota)
        self.assertEqual(scheduled_queries(204, self.s, "random"),
                         scheduled_queries(204, self.s, "random"))
        observed = {"margin": .9, "entropy": .8}
        a = decide_query(observed, "margin_uncertainty", .4,
                         index=2, remaining=1, settings=self.s, planned=set())
        b = decide_query(observed, "margin_uncertainty", .4,
                         index=2, remaining=1, settings=self.s, planned=set())
        self.assertEqual(a, b)
        no_state = {"margin": .0, "entropy": .0}
        x = decide_query(no_state, "margin_uncertainty", .4,
                         index=2, remaining=1, settings=self.s, planned=set())
        self.assertNotEqual(a, x)
        # One remaining opportunity must force spending the last query.
        forced = decide_query(no_state, "margin_uncertainty", .4,
                              index=self.s.tasks - 1, remaining=1,
                              settings=self.s, planned=set())
        self.assertEqual(forced, (True, False, True))

    def test_extra_samples_are_disjoint_and_labels_match_world(self):
        seed = 203
        for kind, shifted in (("few", False), ("shifted", True)):
            data = task_world(seed, self.s, kind)
            pool = query_pool(seed, self.s, data, shifted=shifted,
                              corrupt_extra=False)
            rules = _latent_rules(seed, shifted)
            for ep, samples in zip(data["episodes"], pool):
                c = ep["latent"]
                train_xy = {xy for xy, _ in ep["training"]}
                support_xy = {xy for xy, _ in ep["support"]}
                holdout_xy = {xy for xy, _ in data["panel"][c]}
                for point, label in samples:
                    self.assertNotIn(point, train_xy | support_xy | holdout_xy)
                    self.assertEqual(label, rules[c // 2].label(point) ^ (c % 2))
            corrupted = query_pool(seed, self.s, data, shifted=shifted,
                                   corrupt_extra=True)
            for clean, noisy in zip(pool, corrupted):
                self.assertEqual([x for x, _ in clean],
                                 [x for x, _ in noisy])
                self.assertEqual(sum(a[1] != b[1] for a, b in zip(clean, noisy)), 1)

    def test_query_action_cannot_see_future_extra_labels(self):
        data = task_world(204, self.s, "few")
        pool = query_pool(204, self.s, data, shifted=False,
                          corrupt_extra=False)
        resolver = Resolver(self.s, self.s.reservoir_threshold)
        resolver.prototypes[0] = data["episodes"][0]["training"][:12]
        original = uncertainty_scores(resolver, data["episodes"][1]["support"])
        asked = decide_query(original, "margin_uncertainty", .4,
                             index=1, remaining=2, settings=self.s,
                             planned=set())
        altered_pool = copy.deepcopy(pool)
        altered_pool[1] = [((.99, -.99), 1), ((-.98, .98), 0)]
        self.assertNotEqual(pool[1], altered_pool[1])
        # The decision sees support and existing prototypes, NOT altered_pool.
        self.assertEqual(asked, decide_query(
            uncertainty_scores(resolver, data["episodes"][1]["support"]),
            "margin_uncertainty", .4, index=1, remaining=2,
            settings=self.s, planned=set()))
        # Latent IDs and heldout panels are not used by the resolver.
        first = resolve_seed(204, self.s, data, pool,
                             "margin_uncertainty", .4)
        changed = copy.deepcopy(data)
        for e in changed["episodes"]:
            e["latent"] = (e["latent"] + 2) % 4
            e["novel"] = not e["novel"]
        changed["panel"] = {i: [] for i in range(4)}
        second = resolve_seed(204, self.s, changed, pool,
                              "margin_uncertainty", .4)
        self.assertEqual(first["episode_slots"], second["episode_slots"])
        self.assertEqual([l["query"] for l in first["log"]],
                         [l["query"] for l in second["log"]])

    def test_meta_selection_does_not_depend_on_sealed_test_seed_ranges(self):
        for arm, grid in (("margin_uncertainty", MARGIN_GRID),
                          ("entropy_uncertainty", ENTROPY_GRID)):
            selected, trace = select_threshold(self.s, arm)
            self.assertIn(selected, grid)
            same_s = Settings(**{**self.s.__dict__, "test_start": 205,
                                 "shifted_start": 310, "few_start": 420})
            selected2, trace2 = select_threshold(same_s, arm)
            self.assertEqual(selected, selected2)
            self.assertEqual(trace, trace2)

    def test_full_campaign_deterministic_except_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "a"
            b = Path(directory) / "b"
            one = run(self.s, a)
            two = run(self.s, b)
            one.pop("wall_seconds_including_meta_selection")
            two.pop("wall_seconds_including_meta_selection")
            self.assertEqual(one, two)
            for name in ("summary.json", "seed_metrics.csv", "request_metrics.csv"):
                if name == "summary.json":
                    continue  # contains measured wall time
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
            with (a / "seed_metrics.csv").open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 6 * len(ARMS))
            for row in rows:
                if row["arm"] in MATCHED:
                    self.assertEqual(int(row["query_events"]), self.s.request_quota)
                    self.assertEqual(int(row["extra_labels"]),
                                     self.s.request_quota * self.s.extra_per_request)
            self.assertIn("margin_vs_random", one["main"]["comparisons"])


if __name__ == "__main__":
    unittest.main()
