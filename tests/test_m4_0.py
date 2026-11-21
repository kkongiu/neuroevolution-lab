import csv
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.m4_0 import (
    ACTIONS, ARMS, DIRECTIONS, INPUTS, RecurrentController, Settings, World,
    apply_action, evolve, fresh_genome, genome_size, initial_state,
    make_world, observation, render_world_svg, run, simulate,
)


class SurvivalWorldTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings(
            train_start=10, train_worlds=2,
            validation_start=100, validation_worlds=1,
            test_start=200, test_worlds=3,
            shifted_start=300, shifted_worlds=2,
            size=11, food_count=6, shifted_food_count=4,
            odor_radius=4, shifted_odor_radius=2,
            horizon=35, hidden=5, population=6, generations=2,
            elites=2, immigrants=1,
        )

    def test_seed_partitions_and_settings(self):
        self.s.validate()
        with self.assertRaises(ValueError):
            Settings(train_start=100, train_worlds=4,
                     validation_start=102).validate()
        with self.assertRaises(ValueError):
            Settings(test_start=300, shifted_start=300).validate()
        with self.assertRaises(ValueError):
            Settings(size=10).validate()
        with self.assertRaises(ValueError):
            Settings(population=3, elites=2, immigrants=1).validate()

    def test_world_reproducibility_food_reachable_and_outside_walls(self):
        for shifted in (False, True):
            for seed in range(200, 216):
                a = make_world(seed, self.s, shifted)
                b = make_world(seed, self.s, shifted)
                self.assertEqual(a, b)
                self.assertNotIn(a.start, a.walls)
                self.assertEqual(
                    len(a.initial_food),
                    self.s.shifted_food_count if shifted else self.s.food_count,
                )
                self.assertEqual(len(a.initial_food), len(set(a.initial_food)))
                self.assertTrue(all(x in a.reachable() and x not in a.walls
                                    for x in a.initial_food))
                self.assertTrue(all((a.start[0] + dx, a.start[1] + dy) not in a.walls
                                    for dx, dy in DIRECTIONS))
                self.assertGreaterEqual(len(a.reachable()), 41)

    def test_energy_collision_consumption_and_death(self):
        size = 9
        start = (4, 4)
        walls = frozenset({
            (x, y) for x in range(size) for y in range(size)
            if x in (0, size-1) or y in (0, size-1)
        } | {(4, 3)})
        food = ((5, 4),)
        world = World(999, size, start, walls, food, odor_radius=5, shifted=False)
        state = initial_state(world, self.s)
        e = state.energy
        apply_action(world, state, 0, self.s)  # hit wall north
        self.assertEqual(state.position, start)
        self.assertEqual(state.collisions, 1)
        self.assertAlmostEqual(state.energy, e - self.s.movement_cost -
                               self.s.collision_extra_cost)
        apply_action(world, state, 1, self.s)  # east: eat once
        self.assertEqual(state.eaten, 1)
        self.assertNotIn(food[0], state.food)
        self.assertGreater(state.energy, e)
        apply_action(world, state, 3, self.s)
        apply_action(world, state, 1, self.s)
        self.assertEqual(state.eaten, 1)
        with self.assertRaises(ValueError):
            apply_action(world, state, 7, self.s)
        state.energy = .2
        apply_action(world, state, 0, self.s)
        self.assertEqual(state.energy, 0)
        with self.assertRaises(ValueError):
            apply_action(world, state, 1, self.s)

    def test_sensors_are_local_and_food_odor_has_no_absolute_map(self):
        world = make_world(204, self.s)
        state = initial_state(world, self.s)
        obs = observation(world, state, self.s)
        self.assertEqual(len(obs), INPUTS)
        self.assertTrue(all(0 <= x <= 1 for x in obs))
        self.assertEqual(obs[12], self.s.initial_energy / self.s.max_energy)
        state.food.clear()
        no_food = observation(world, state, self.s)
        self.assertEqual(no_food[4:12], (0.,) * 8)
        self.assertEqual(no_food[:4], obs[:4])

    def test_recurrent_genome_replays_deterministically_and_dose(self):
        world = make_world(201, self.s)
        rng = random.Random(717)
        genome = fresh_genome(rng, self.s)
        self.assertEqual(len(genome), genome_size(self.s.hidden))
        net = RecurrentController(genome, self.s.hidden)
        state = initial_state(world, self.s)
        a = net.choose(observation(world, state, self.s))
        self.assertIn(a, range(ACTIONS))
        self.assertTrue(any(abs(x) > 1e-8 for x in net.hidden))
        a1 = simulate(world, self.s, "evolved", genome, trace=True)
        a2 = simulate(world, self.s, "evolved", genome, trace=True)
        self.assertEqual(a1, a2)
        self.assertTrue(0 <= a1["food_collected"] <= len(world.initial_food))
        self.assertLessEqual(a1["steps_alive"], self.s.horizon)
        self.assertEqual(len(a1["path"]), a1["steps_alive"] + 1)
        self.assertEqual(len(a1["events"]), a1["steps_alive"])
        for event in a1["events"]:
            self.assertNotIn(tuple(event["position"]), world.walls)
        with self.assertRaises(ValueError):
            RecurrentController(genome[:-1], self.s.hidden)

    def test_all_agents_get_same_start_and_zero_odor_ablation(self):
        world = make_world(203, self.s)
        genome = fresh_genome(random.Random(51), self.s)
        for arm in ARMS:
            g = genome if arm in ("evolved", "random_neural",
                                  "evolved_no_odor") else None
            result = simulate(world, self.s, arm, g, trace=True)
            self.assertEqual(result["path"][0], world.start)
            self.assertEqual(result["seed"], world.seed)
        state = initial_state(world, self.s)
        obs = observation(world, state, self.s)
        net = RecurrentController(genome, self.s.hidden, no_odor=True)
        net.choose(obs)
        blind_hidden = list(net.hidden)
        altered = obs[:4] + (0.,) * 4 + obs[8:]
        reference = RecurrentController(genome, self.s.hidden)
        reference.choose(altered)
        self.assertEqual(blind_hidden, reference.hidden)

    def test_sealed_test_seed_changes_cannot_change_champion(self):
        a, history_a, selection_a = evolve(self.s)
        bsettings = Settings(**{
            **self.s.__dict__, "test_start": 215, "shifted_start": 350
        })
        b, history_b, selection_b = evolve(bsettings)
        self.assertEqual(a, b)
        self.assertEqual(history_a, history_b)
        self.assertEqual(selection_a, selection_b)
        self.assertEqual(selection_a["train_candidate_world_evaluations"],
                         self.s.generations * self.s.population *
                         self.s.train_worlds)

    def test_replay_campaign_deterministic_csv_and_svg(self):
        with tempfile.TemporaryDirectory() as path:
            a = Path(path) / "a"
            b = Path(path) / "b"
            one, two = run(self.s, a), run(self.s, b)
            self.assertEqual(one, two)
            for file in ("history.csv", "test_seed_metrics.csv",
                         "champion.json", "summary.json",
                         "example_main.svg", "example_shifted.svg",
                         "example_main_trace.json"):
                self.assertEqual((a / file).read_bytes(),
                                 (b / file).read_bytes())
            with (a / "test_seed_metrics.csv").open(newline="") as f:
                data = list(csv.DictReader(f))
            self.assertEqual(len(data),
                             (self.s.test_worlds + self.s.shifted_worlds) *
                             len(ARMS))
            self.assertIn("evolved_minus_wall_avoiding_random",
                          one["main"]["comparisons"])
            self.assertIn("<svg", (a / "example_main.svg").read_text())
            self.assertIn("events", json.loads(
                (a / "example_main_trace.json").read_text()))


if __name__ == "__main__":
    unittest.main()
