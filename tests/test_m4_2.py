import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.m4_0 import DIRECTIONS
from neuroevolution.m4_2 import (
    ACTIONS, CONTRASTS, DELAYED_ARMS, INPUTS, Controller, Settings,
    direction, evolve_one, fresh_genome, genome_size,
    hierarchical_bootstrap, make_pair, observation, reactive_trial,
    run, simulate,
)
from neuroevolution.m4_1 import make_diverse_world


class DelayedCueMemoryTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings(
            train_start=10, train_worlds=2,
            validation_start=100, validation_worlds=1,
            test_start=200, test_worlds=3,
            shifted_start=300, shifted_worlds=2,
            reactive_start=400, reactive_worlds=2,
            genetic_seed_start=800, genetic_runs=2,
            size=13, hidden=4, ff_hidden=5,
            stem_min=3, stem_max=4,
            shifted_stem_min=4, shifted_stem_max=5,
            cue_ticks=2, horizon=14,
            population=6, generations=2,
            elites=2, immigrants=1,
        )

    def test_seed_splits_and_geometry_constraints(self):
        self.s.validate()
        with self.assertRaises(ValueError):
            Settings(train_start=10, train_worlds=4, generations=4,
                     validation_start=20).validate()
        with self.assertRaises(ValueError):
            Settings(test_start=300, shifted_start=300).validate()
        with self.assertRaises(ValueError):
            Settings(reactive_start=57000).validate()
        with self.assertRaises(ValueError):
            Settings(size=11).validate()
        with self.assertRaises(ValueError):
            Settings(cue_ticks=3, stem_min=3).validate()

    def test_paired_mazes_identical_geometry_opposite_goal_and_no_leakage(self):
        transformations = set()
        for shifted in (False, True):
            for seed in range(200, 240):
                w0, w1 = make_pair(seed, self.s, shifted)
                transformations.add((w0.quarter_turns, w0.reflected))
                self.assertEqual(w0.walls, w1.walls)
                self.assertEqual(w0.start, w1.start)
                self.assertEqual(w0.fork, w1.fork)
                self.assertEqual(w0.gates, w1.gates)
                self.assertNotEqual(w0.target, w1.target)
                self.assertNotEqual(w0.target_direction,
                                    w1.target_direction)
                self.assertIn(w0.target_direction, w0.open_at(w0.fork))
                self.assertNotIn(w0.start, w0.walls)
                self.assertTrue(all(i not in w0.walls for i in w0.gates))
                initial0 = observation(w0, w0.start, 0, 24, 0, 0, self.s)
                initial1 = observation(w1, w1.start, 0, 24, 0, 0, self.s)
                self.assertEqual(initial0[:15], initial1[:15])
                self.assertNotEqual(initial0[15:], initial1[15:])
                self.assertEqual(initial0[4:12], (0.,) * 8)
                fork0 = observation(w0, w0.fork, 7, 17, 0, 0, self.s)
                fork1 = observation(w1, w1.fork, 11, 9, 0, 0, self.s)
                # Different elapsed times and energy must NOT leak past cue.
                self.assertEqual(fork0, fork1)
                self.assertEqual(fork0[15:], (0.,) * 4)
                self.assertEqual(fork0[12], 1.)
        self.assertGreater(len(transformations), 5)

    def test_scripted_memory_correct_without_oracle_goal_access(self):
        for seed in range(200, 240):
            for w in make_pair(seed, self.s):
                result = simulate(w, self.s, "scripted_memory", trace=True)
                self.assertEqual(result["success"], 1)
                self.assertEqual(result["fork_reached"], 1)
                self.assertEqual(result["committed"], 1)
                self.assertEqual(result["collisions"], 0)
                self.assertEqual(len(result["path"]), result["steps"] + 1)
                self.assertEqual(result["path"][-1], w.target)
                self.assertTrue(all(tuple(e["position"]) not in w.walls
                                    for e in result["events"]))

    def test_gate_is_one_way_and_rejects_wrong_choice(self):
        w0, w1 = make_pair(201, self.s)
        # Side targets are opposite; on a given world branch entry ends immediately.
        same_script = simulate(w0, self.s, "scripted_memory")
        self.assertEqual(same_script["success"], 1)
        blank = observation(w0, w0.fork, self.s.cue_ticks+1, 16., 0, 0, self.s)
        self.assertEqual(len(blank), INPUTS)
        self.assertEqual(blank[4:12], (0.,) * 8)

    def test_recurrence_can_carry_information_ff_cannot_at_identical_fork_obs(self):
        w0, w1 = make_pair(214, self.s)
        n = self.s.hidden
        stride = INPUTS + n + 1
        genes = [0.] * genome_size("rnn", self.s)
        # Four hidden units respond to respective cardinal cue and hold it.
        for i in range(4):
            genes[i*stride + 15+i] = 4.
            genes[i*stride + INPUTS+i] = 5.
            genes[i*stride + stride-1] = -0.5
            genes[n*stride + i*(n+1)+i] = 7.
        a = Controller(tuple(genes), "rnn", self.s)
        b = Controller(tuple(genes), "rnn", self.s)
        for world, agent in ((w0, a), (w1, b)):
            agent.choose(observation(world, world.start, 0, 24., 0, 0, self.s))
            for step in range(2, 5):
                agent.choose(observation(world, world.start, step, 22., 0, 0, self.s))
        fork = observation(w0, w0.fork, 9, 10., 0, 0, self.s)
        action0 = a.choose(fork)
        action1 = b.choose(fork)
        self.assertEqual(action0, w0.target_direction)
        self.assertEqual(action1, w1.target_direction)
        self.assertNotEqual(action0, action1)
        # The same learned weights with cleared hidden state cannot distinguish.
        a.reset_memory()
        b.reset_memory()
        self.assertEqual(a.choose(fork), b.choose(fork))
        # Stateless model also cannot respond differently to same current fork data.
        ff = Controller(fresh_genome(random.Random(1), self.s, "ff"),
                        "ff", self.s)
        self.assertEqual(ff.choose(fork), ff.choose(fork))

    def test_evolution_budget_selection_disjoint_from_final_worlds(self):
        genome, history, selected = evolve_one(self.s, 800, "rnn")
        changed = Settings(**{
            **self.s.__dict__,
            "test_start": 210,
            "shifted_start": 320,
            "reactive_start": 430,
        })
        g2, h2, s2 = evolve_one(changed, 800, "rnn")
        self.assertEqual(genome, g2)
        self.assertEqual(history, h2)
        self.assertEqual(selected, s2)
        self.assertEqual(selected["candidate_trials"],
                         self.s.generations * self.s.population *
                         self.s.train_worlds * 2)
        self.assertEqual(len(genome), genome_size("rnn", self.s))
        self.assertLess(abs(genome_size("rnn", Settings()) -
                            genome_size("ff", Settings())) /
                        genome_size("rnn", Settings()), .04)

    def test_hierarchical_bootstrap_replicates_dimensions(self):
        self.assertEqual(hierarchical_bootstrap([[1., 1.], [1., 1.]], 9),
                         (1., 1.))
        with self.assertRaises(ValueError):
            hierarchical_bootstrap([[1.], [1., 2.]], 9)

    def test_reactive_transfer_has_frozen_sensors_and_mask(self):
        world = make_diverse_world(403, self.s)
        g = fresh_genome(random.Random(7), self.s, "rnn")
        for arm in ("rnn_reactive", "rnn_reactive_no_odor",
                    "greedy_odor", "legal_random"):
            r = reactive_trial(world, self.s, arm,
                               g if arm.startswith("rnn_") else None)
            self.assertEqual(r["collisions"], 0)
            self.assertGreaterEqual(r["steps"], 1)
            self.assertLessEqual(r["steps"], self.s.horizon)

    def test_end_to_end_deterministic_reproducible_and_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a", Path(folder) / "b"
            result_a, result_b = run(self.s, a), run(self.s, b)
            self.assertEqual(result_a, result_b)
            for name in ("summary.json", "genotypes.json",
                         "train_validation.csv", "delayed_trial_metrics.csv",
                         "reactive_transfer_metrics.csv",
                         "example_main.svg", "example_shifted_trace.json"):
                self.assertEqual((a/name).read_bytes(), (b/name).read_bytes())
            with (a/"delayed_trial_metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows),
                             self.s.genetic_runs * (self.s.test_worlds +
                             self.s.shifted_worlds) * 2 * len(DELAYED_ARMS))
            self.assertEqual(len(result_a["selection"]), self.s.genetic_runs * 2)
            self.assertTrue(all(r["candidate_trials"] ==
                                self.s.generations * self.s.population *
                                self.s.train_worlds * 2
                                for r in result_a["selection"]))
            self.assertEqual(result_a["main"]["means"]["scripted_memory"]["success"], 1.)
            self.assertEqual(result_a["main"]["means"]["legal_random"]["collisions"], 0.)
            self.assertIn("primary_rnn_vs_ff", result_a["main"]["contrasts"])
            self.assertIn("reset_memory_effect", result_a["main"]["contrasts"])


if __name__ == "__main__":
    unittest.main()
