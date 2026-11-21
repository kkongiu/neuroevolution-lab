import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.m4_0 import DIRECTIONS, fresh_genome, observation, initial_state
from neuroevolution.m4_1 import (
    ACTIONS, ARMS, MaskAwareRNN, Settings, hierarchical_bootstrap,
    make_diverse_world, run, simulate, evolve_one,
)


class MaskedNavigationTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings(
            train_start=10, train_worlds=2,
            validation_start=100, validation_worlds=1,
            test_start=200, test_worlds=3,
            shifted_start=300, shifted_worlds=2,
            genetic_seed_start=800, genetic_runs=2,
            size=11, hidden=5, horizon=35,
            food_count=6, shifted_food_count=4,
            odor_radius=4, shifted_odor_radius=2,
            population=6, generations=2, elites=2, immigrants=1,
        )

    def test_seed_splits_include_all_generation_training_worlds(self):
        self.s.validate()
        with self.assertRaises(ValueError):
            Settings(train_start=100, train_worlds=4, generations=3,
                     validation_start=110).validate()
        with self.assertRaises(ValueError):
            Settings(test_start=300, shifted_start=300).validate()
        with self.assertRaises(ValueError):
            Settings(genetic_runs=1).validate()

    def test_worlds_have_food_reachable_from_randomized_starts(self):
        starts = set()
        for shifted in (False, True):
            for seed in range(200, 225):
                world = make_diverse_world(seed, self.s, shifted)
                self.assertEqual(world, make_diverse_world(seed, self.s, shifted))
                starts.add(world.start)
                self.assertNotIn(world.start, world.walls)
                self.assertNotIn(world.start, world.initial_food)
                self.assertTrue(all(p in world.reachable() and p not in world.walls
                                    for p in world.initial_food))
                self.assertGreaterEqual(sum(
                    (world.start[0] + dx, world.start[1] + dy) not in world.walls
                    for dx, dy in DIRECTIONS
                ), 2)
        self.assertGreater(len(starts), 5)

    def test_mask_changes_only_illegal_action_decoder(self):
        h = self.s.hidden
        genome = [0.] * (h * (15 + h + 1) + 4 * (h + 1))
        head = h * (15 + h + 1)
        genome[head + h] = 10.  # North output bias highest
        genome[head + (h+1) + h] = 7.  # East second
        obs = (1., 0., 1., 0.) + (0.,) * 11
        original = MaskAwareRNN(tuple(genome), h, masked=False)
        guarded = MaskAwareRNN(tuple(genome), h, masked=True)
        self.assertEqual(original.choose(obs), 0)
        self.assertEqual(guarded.choose(obs), 1)
        self.assertEqual(original.hidden, guarded.hidden)
        self.assertEqual(guarded.illegal_preferences, 1)
        self.assertEqual(guarded.corrected_preferences, 1)
        self.assertEqual(original.corrected_preferences, 0)
        self.assertEqual(guarded.choose((0.,)*15), 0)
        self.assertEqual(guarded.corrected_preferences, 1)

    def test_cross_over_physics_valid_under_same_genome(self):
        world = make_diverse_world(210, self.s)
        g = fresh_genome(random.Random(15), self.s)
        a = simulate(world, self.s, "evolved_masked", g, trace=True)
        b = simulate(world, self.s, "evolved_masked", g, trace=True)
        c = simulate(world, self.s, "evolved_masked_unmasked", g)
        self.assertEqual(a, b)
        self.assertEqual(a["seed"], c["seed"])
        self.assertEqual(a["collisions"], 0)
        self.assertGreaterEqual(c["collisions"], 0)
        self.assertEqual(len(a["path"]), a["steps"] + 1)
        self.assertEqual(len(a["events"]), a["steps"])
        self.assertEqual(a["corrected_preferences"],
                         a["illegal_preferences"])
        for event in a["events"]:
            self.assertNotIn(tuple(event["position"]), world.walls)

    def test_selection_does_not_touch_final_worlds(self):
        champion, history, selected = evolve_one(self.s, 804, True)
        modified = Settings(**{
            **self.s.__dict__,
            "test_start": 210,
            "shifted_start": 320,
        })
        champ2, hist2, sel2 = evolve_one(modified, 804, True)
        self.assertEqual(champion, champ2)
        self.assertEqual(history, hist2)
        self.assertEqual(selected, sel2)
        self.assertEqual(selected["candidate_world_evaluations"],
                         self.s.population * self.s.generations *
                         self.s.train_worlds)

    def test_bootstrap_is_by_genetic_run_and_world(self):
        self.assertEqual(hierarchical_bootstrap([[2., 2.], [2., 2.]], 7),
                         (2., 2.))
        a = hierarchical_bootstrap([[-2., 5.], [2., 3.]], 22)
        b = hierarchical_bootstrap([[-2., 5.], [2., 3.]], 22)
        self.assertEqual(a, b)
        self.assertLess(a[0], a[1])
        with self.assertRaises(ValueError):
            hierarchical_bootstrap([[1.], [1., 2.]], 1)

    def test_null_world_does_not_create_food(self):
        world = make_diverse_world(209, self.s)
        empty = world.__class__(world.seed, world.size, world.start,
                                world.walls, (), world.odor_radius, world.shifted)
        g = fresh_genome(random.Random(11), self.s)
        r = simulate(empty, self.s, "evolved_masked", g)
        self.assertEqual(r["food"], 0)
        self.assertEqual(r["collisions"], 0)
        self.assertLessEqual(r["steps"], int(self.s.initial_energy /
                                            self.s.movement_cost) + 1)

    def test_end_to_end_deterministic_and_equal_evolution_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "one", Path(tmp) / "two"
            result1 = run(self.s, a)
            result2 = run(self.s, b)
            self.assertEqual(result1, result2)
            for name in (
                "summary.json", "champions.json", "train_validation.csv",
                "test_world_metrics.csv", "null_food.csv", "example_main.svg",
                "example_shifted_trace.json",
            ):
                self.assertEqual((a/name).read_bytes(), (b/name).read_bytes())
            with (a / "test_world_metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), self.s.genetic_runs *
                             (self.s.test_worlds + self.s.shifted_worlds) *
                             len(ARMS))
            self.assertEqual(len(result1["evolution_selection"]),
                             self.s.genetic_runs * 2)
            self.assertTrue(all(
                x["candidate_world_evaluations"] ==
                self.s.generations * self.s.population * self.s.train_worlds
                for x in result1["evolution_selection"]
            ))
            self.assertTrue(result1["null_food"]["all_zero_food"])
            self.assertIn("primary_vs_walker",
                          result1["main"]["contrasts"])
            self.assertIn("mask_effect_on_mask_trained",
                          result1["main"]["contrasts"])
            self.assertTrue(all(float(r["collisions"]) == 0 for r in rows
                                if r["arm"] == "evolved_masked"))


if __name__ == "__main__":
    unittest.main()
