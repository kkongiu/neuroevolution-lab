import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.m4_2 import Controller, make_pair, genome_size, fresh_genome
from neuroevolution.m4_3 import (
    ARMS, Settings, curriculum_delays, selected_delays,
    cue_observation, neutral_observation, fork_observation,
    inputs_for_trial, run_trial, evolve_one, run, guided_corridor_trial,
)


class PureRecallTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings(
            train_start=10, train_worlds=2,
            validation_start=100, validation_worlds=2,
            test_start=200, test_worlds=3,
            shifted_start=300, shifted_worlds=2,
            guided_start=500, guided_worlds=2,
            genetic_seed_start=810, genetic_runs=2,
            reactive_start=400, reactive_worlds=2,
            size=13, hidden=4, ff_hidden=5,
            population=6, generations=2, elites=2, immigrants=1,
            stem_min=3, stem_max=4, shifted_stem_min=4,
            shifted_stem_max=5, cue_ticks=2, horizon=14,
        )

    def test_frozen_partition_checks_and_curriculum(self):
        self.s.validate()
        self.assertEqual(curriculum_delays(0), (0, 1, 2))
        self.assertEqual(curriculum_delays(1), (0, 1, 2))
        self.assertEqual(curriculum_delays(2), (1, 2, 4))
        self.assertEqual(curriculum_delays(4), (2, 4, 6))
        self.assertEqual(curriculum_delays(7), (2, 4, 6))
        with self.assertRaises(ValueError):
            curriculum_delays(-1)
        with self.assertRaises(ValueError):
            Settings(train_start=100, train_worlds=5,
                     generations=2, validation_start=105).validate()
        with self.assertRaises(ValueError):
            Settings(guided_start=62000).validate()
        with self.assertRaises(ValueError):
            Settings(cue_ticks=3).validate()
        with self.assertRaises(ValueError):
            inputs_for_trial(make_pair(210, self.s)[0], -1)

    def test_stimulus_is_uninformative_after_cue_opposite_targets(self):
        for seed in range(200, 240):
            first, opposite = make_pair(seed, self.s)
            for delay in (1, 2, 4, 6, 10):
                a = inputs_for_trial(first, delay)
                b = inputs_for_trial(opposite, delay)
                self.assertEqual(len(a), self.s.cue_ticks + delay + 1)
                self.assertNotEqual(a[0][15:], b[0][15:])
                self.assertEqual(a[0][:15], b[0][:15])
                self.assertEqual(a[1], a[0])
                self.assertEqual(a[2:], b[2:])
                self.assertTrue(all(x == neutral_observation()
                                    for x in a[2:-1]))
                self.assertEqual(a[-1][15:], (0.,)*4)
                self.assertEqual(a[-1][12], 1.)
                self.assertEqual(len([x for x in a[-1][:4] if x == 0.]), 2)
                self.assertEqual(a[-1][:4], b[-1][:4])
                self.assertEqual(a[-1][4:12], (0.,)*8)

    def test_paired_controls_exact_50_percent_for_every_seed(self):
        rng = random.Random(13)
        rnn = fresh_genome(rng, self.s, "rnn")
        ff = fresh_genome(rng, self.s, "ff")
        for seed in range(200, 240):
            pair = make_pair(seed, self.s)
            for delay in (1, 3, 8):
                for arm in ("ff", "rnn_reset", "rnn_no_cue",
                            "random_binary"):
                    genome = ff if arm == "ff" else (
                        None if arm == "random_binary" else rnn
                    )
                    score = sum(run_trial(w, self.s, arm, genome, delay)
                                ["success"] for w in pair)
                    self.assertEqual(score, 1, (seed, delay, arm))
                self.assertTrue(all(
                    run_trial(w, self.s, "scripted_one_bit", None, delay)
                    ["success"] == 1 for w in pair
                ))

    def test_one_bit_memory_is_realizable_by_recurrent_weights(self):
        # A manually constructed recurrent circuit holds the 4 cardinal cue
        # bits in its hidden state. It is a positive CONTROL, not evolved.
        cfg = self.s
        n = cfg.hidden
        stride = 19 + n + 1
        weights = [0.] * genome_size("rnn", cfg)
        for i in range(4):
            weights[i*stride+15+i] = 4.0
            weights[i*stride+19+i] = 5.0
            weights[i*stride+stride-1] = -.5
            weights[n*stride+i*(n+1)+i] = 7.0
        for seed in range(200, 220):
            for w in make_pair(seed, cfg):
                self.assertEqual(
                    run_trial(w, cfg, "rnn", tuple(weights), 4)["success"], 1
                )
                # Reset wipes the early cue information.
                reset = run_trial(w, cfg, "rnn_reset", tuple(weights), 4)
                self.assertIn(reset["success"], (0, 1))
        # A feedforward controller cannot hold the early information.
        self.assertEqual(genome_size("rnn", Settings()), 260)
        self.assertEqual(genome_size("ff", Settings()), 268)

    def test_cue_swap_inverts_initial_information_but_not_true_reward(self):
        w0, w1 = make_pair(218, self.s)
        a = inputs_for_trial(w0, 4)
        flipped = inputs_for_trial(w0, 4, flipped=True)
        b = inputs_for_trial(w1, 4)
        self.assertEqual(flipped, b)
        self.assertNotEqual(a[0], flipped[0])
        self.assertEqual(a[2:], flipped[2:])
        blank = inputs_for_trial(w0, 4, no_cue=True)
        self.assertEqual(blank[0][15:], (0.,)*4)

    def test_genetic_selection_not_changed_by_final_seed_range(self):
        first, hist, selected = evolve_one(self.s, 810, "rnn")
        changed = Settings(**{
            **self.s.__dict__, "test_start": 215,
            "shifted_start": 335, "guided_start": 520,
        })
        second, hist2, selected2 = evolve_one(changed, 810, "rnn")
        self.assertEqual(first, second)
        self.assertEqual(hist, hist2)
        self.assertEqual(selected, selected2)
        self.assertEqual(selected["candidate_training_trials"],
                         self.s.generations*self.s.population*
                         self.s.train_worlds*2*3)

    def test_all_models_reach_guided_fork(self):
        w = make_pair(504, self.s)[0]
        rnn = fresh_genome(random.Random(13), self.s, "rnn")
        ff = fresh_genome(random.Random(14), self.s, "ff")
        for arm, g in (("rnn", rnn), ("ff", ff)):
            result = guided_corridor_trial(w, self.s, arm, g)
            self.assertEqual(result["fork_reached"], 1)
            self.assertIn(result["success"], (0, 1))

    def test_end_to_end_small_campaign_reproduces_all_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/"a", Path(directory)/"b"
            a = run(self.s, first)
            b = run(self.s, second)
            self.assertEqual(a, b)
            for file in (
                "summary.json", "champions.json", "train_validation.csv",
                "recall_trial_metrics.csv", "corridor_transfer.csv",
                "example_main.svg", "example_shifted_trace.json",
            ):
                self.assertEqual((first/file).read_bytes(),
                                 (second/file).read_bytes())
            for split in ("main", "shifted"):
                d = a[split]
                self.assertTrue(all(d["analytical_controls_exact"].values()))
                self.assertIn("reset_effect", d["contrasts"])
            with (first/"recall_trial_metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(
                len(rows),
                self.s.genetic_runs*(self.s.test_worlds+self.s.shifted_worlds)
                *2*len(ARMS)
            )
            self.assertEqual(len(a["selection"]), self.s.genetic_runs*2)


if __name__ == "__main__":
    unittest.main()
