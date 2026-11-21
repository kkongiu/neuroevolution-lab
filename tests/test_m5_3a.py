"""M5.3A representational capacity test: no trained genotype called evolved."""
import csv
import random
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

from neuroevolution import m5_0 as base
from neuroevolution import m5_3a as cap


class RepresentationStudyTests(unittest.TestCase):
    def setUp(self):
        self.s = cap.Settings(
            seed_start=700, seeds=2, shifted_start=900, shifted_seeds=2,
            ticks=40, size=11, initial_food=9, food_cap=14,
            initial_population=8, max_population=18,
            checkpoint_every=10, hidden=12,
            teacher_seed_start=1100, teacher_seed_count=2,
            validation_teacher_start=1200, validation_teacher_count=1,
            teacher_ticks=35, train_cap=150, validation_cap=90,
            epochs=3, learning_rate=.03,
        )

    def test_protocol_splits_and_validation(self):
        self.s.validate()
        self.assertEqual(base.genome_size(12), 401)
        self.assertEqual(base.genome_size(2), 51)
        self.assertEqual(len(cap.ARMS), 6)
        for n in (2, 12):
            self.assertEqual(base.genome_size(n), len(cap.initial_student(n)))
        with self.assertRaises(ValueError):
            replace(self.s, seed_start=1100).validate()
        with self.assertRaises(ValueError):
            replace(self.s, shifted_start=700).validate()
        with self.assertRaises(ValueError):
            replace(self.s, ticks=400, epochs=3).validate()

    def test_engineered_numeric_logits_directions_and_energy(self):
        weights = cap.witness_12()
        self.assertEqual(len(weights), 401)
        n = 12
        inputs = [0.] * base.INPUTS
        inputs[12] = .5
        for direction in range(4):
            obs = list(inputs)
            obs[8 + direction] = 1.  # one local adjacent food
            chosen = cap.predict(weights, n, tuple(obs), tuple(range(4)))
            self.assertEqual(chosen, direction)
        # A high-energy legally eligible animal can autonomously choose birth.
        obs = list(inputs)
        obs[12] = 1.
        self.assertEqual(
            cap.predict(weights, n, tuple(obs), tuple(range(5))), 4
        )
        obs[12] = .4
        self.assertNotEqual(
            cap.predict(weights, n, tuple(obs), tuple(range(5))), 4
        )

    def test_neural_action_output_changes_only_when_numeric_genome_changes(self):
        weights = list(cap.witness_12())
        obs = [0.] * base.INPUTS
        obs[8] = 1.
        obs[12] = 1.
        self.assertEqual(cap.predict(weights, 12, tuple(obs), (0, 4)), 4)
        # Causal action-head ablation removes the birth preference.
        head = 12 * (base.INPUTS + 12 + 1)
        weights[head + 4 * (12 + 1) + 4] = 0.
        weights[head + 4 * (12 + 1) + 12] = -10.
        self.assertEqual(cap.predict(weights, 12, tuple(obs), (0, 4)), 0)

    def test_teacher_only_train_local_sensors_and_legal_actions(self):
        a = cap.teacher_examples(self.s, validation=False)
        b = cap.teacher_examples(self.s, validation=True)
        self.assertEqual(len(a), self.s.train_cap)
        self.assertEqual(len(b), self.s.validation_cap)
        self.assertTrue(all(
            len(sensors) == 15 and label in legal and
            all(0 <= v <= 1 for v in sensors) and
            all(0 <= action <= 4 for action in legal)
            for sensors, legal, label in a + b
        ))
        self.assertEqual(a, cap.teacher_examples(self.s, validation=False))

    def test_supervised_student_deterministic_and_no_recurrent_prior(self):
        training = cap.teacher_examples(self.s, validation=False)
        validation = cap.teacher_examples(self.s, validation=True)
        for n in (2, 12):
            wa, ma = cap.train_student(n, self.s, training, validation)
            wb, mb = cap.train_student(n, self.s, training, validation)
            self.assertEqual((wa, ma), (wb, mb))
            self.assertEqual(len(wa), base.genome_size(n))
            stride = base.INPUTS + n + 1
            for i in range(n):
                self.assertTrue(all(
                    wa[i*stride + base.INPUTS + k] == 0.
                    for k in range(n)
                ))
            self.assertGreaterEqual(ma["best_validation_accuracy"], 0.)
            self.assertLessEqual(ma["best_validation_accuracy"], 1.)
            self.assertTrue(1 <= ma["selected_epoch"] <= self.s.epochs)

    def test_genotype_bank_shared_exactly_by_worlds(self):
        bank, info = cap.genotypes(self.s)
        self.assertEqual(set(bank), set(cap.ARMS))
        self.assertEqual(len(bank["imitation_2"]), 51)
        self.assertEqual(len(bank["imitation_12"]), 401)
        self.assertEqual(
            sorted(bank["scrambled_12"]), sorted(bank["engineered_12"])
        )
        self.assertNotEqual(bank["scrambled_12"], bank["engineered_12"])
        self.assertNotEqual(bank["random_12"], bank["engineered_12"])
        self.assertIn("best_validation_accuracy", info["imitation_12"])

    def test_clonal_birth_and_ledger_no_action_script(self):
        bank, _ = cap.genotypes(self.s)
        for arm in cap.ARMS:
            a = cap.simulate(702, self.s, arm, bank[arm], detailed=True)
            b = cap.simulate(702, self.s, arm, bank[arm], detailed=True)
            self.assertEqual(a, b)
            row, checks, trace = a
            self.assertEqual(row["all_actions_original_neural"], 1)
            self.assertEqual(row["inheritance"], "clone_no_mutation")
            self.assertAlmostEqual(row["energy_ledger_residual"], 0.)
            self.assertTrue(all(
                abs(ck["energy_ledger_residual"]) < 1e-6
                for ck in checks
            ))
            self.assertTrue(all(
                event["genome"] == base.genome_digest(bank[arm])
                for event in trace["events"] if event["type"] == "birth"
            ))
            self.assertTrue(all(
                tuple(event["location"]) not in set(map(tuple, trace["walls"]))
                for event in trace["events"] if event["type"] == "birth"
            ))

    def test_main_world_paired_analysis_not_action_pseudoreplication(self):
        rows = []
        for seed, values in (
            (1, (1, 1, 0, 0, 0, 0)),
            (2, (1, 0, 0, 0, 0, 0)),
            (3, (0, 0, 0, 0, 0, 0)),
        ):
            rows.extend({
                "seed": seed, "split": "main", "arm": arm,
                "survived": value
            } for arm, value in zip(cap.ARMS, values))
        cmp = cap.contrasts(rows, "main", "survived", 30)
        self.assertAlmostEqual(
            cmp["engineered_12_minus_random_12"]["mean_delta"], 2/3
        )
        self.assertEqual(
            cmp["engineered_12_minus_random_12"]["positive_worlds"], 2
        )

    def test_reproducible_complete_smoke_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            ra, rb = cap.run(self.s, a), cap.run(self.s, b)
            self.assertEqual(ra, rb)
            for filename in (
                "summary.json", "world_metrics.csv",
                "population_checkpoints.csv", "genotypes.json",
                "population_curves.svg",
                "events_main_engineered_12.json",
                "events_shifted_imitation_2.json",
            ):
                self.assertEqual((a/filename).read_bytes(),
                                 (b/filename).read_bytes())
            with (a/"world_metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(
                len(rows), len(cap.ARMS)*(self.s.seeds+self.s.shifted_seeds)
            )
            self.assertIn("engineered_12_primary_pass",
                          ra["preregistered_gates"])


if __name__ == "__main__":
    unittest.main()
