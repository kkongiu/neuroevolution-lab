"""Regression and causal-intervention checks for M5.2A.

Engineering smoke seeds are deliberately distinct from sealed final seeds.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from neuroevolution import m5_0 as base
from neuroevolution import m5_1 as controls
from neuroevolution import m5_2a as trial


class FactorialCausalDiagnosis(unittest.TestCase):
    def setUp(self):
        self.s = trial.Settings(
            seed_start=700, seeds=2, shifted_start=900, shifted_seeds=2,
            negative_control_worlds=1, ticks=40, size=11,
            initial_food=9, food_cap=14, initial_population=8,
            max_population=18, checkpoint_every=10, hidden=4,
        )

    def test_arm_constants_and_sealed_partitions(self):
        self.s.validate()
        self.assertEqual(len(set(trial.ARMS)), 4)
        self.assertFalse(set(range(75000, 75020)).intersection(
            range(76000, 76010)
        ))
        with self.assertRaises(ValueError):
            trial.Settings(hidden=4).validate()

    def test_nn_is_exact_unmodified_m50_actor(self):
        for shifted in (False, True):
            for seed in (700, 701):
                before, cp0, _ = base.simulate(
                    seed, self.s, "heritable_mutation", shifted=shifted
                )
                after, cp1, _ = trial.simulate(
                    seed, self.s, "NN", shifted=shifted
                )
                for field in (
                    "births", "deaths", "alive_final", "food_eaten",
                    "food_remaining", "max_generation",
                    "motor_mask_corrections",
                    "ineligible_reproduction_preferences",
                ):
                    self.assertEqual(before[field], after[field],
                                     (seed, shifted, field))
                self.assertEqual(
                    [(p["tick"], p["population"], p["food_stock"])
                     for p in cp0],
                    [(p["tick"], p["population"], p["food_stock"])
                     for p in cp1],
                )

    def test_gt_is_exact_m51_positive_control(self):
        for shifted in (False, True):
            for seed in (700, 701):
                original, cp0, _ = controls.simulate(
                    seed, self.s, "greedy_odor", shifted=shifted
                )
                intervention, cp1, _ = trial.simulate(
                    seed, self.s, "GT", shifted=shifted
                )
                for field in (
                    "births", "deaths", "alive_final", "food_eaten",
                    "food_remaining", "max_generation",
                    "motor_mask_corrections",
                ):
                    self.assertEqual(original[field], intervention[field],
                                     (seed, shifted, field))
                self.assertEqual(
                    [(p["population"], p["food_stock"]) for p in cp0],
                    [(p["population"], p["food_stock"]) for p in cp1],
                )

    def test_threshold_reproduction_intervention_is_real(self):
        eco = base.create_ecology(700, self.s, "heritable_mutation")
        a = next(x for x in eco.agents.values()
                 if base.blocked_and_free(eco, x)[1])
        a.age, a.energy = 5, 50
        # Zero-output neural controller prefers direction 0, not REPRODUCE.
        a.genome = tuple(0. for _ in a.genome)
        action, _ = trial.choose(eco, a, 5, "NT", {})
        self.assertEqual(action, base.REPRODUCE)
        a.energy = 40
        action, _ = trial.choose(eco, a, 5, "NT", {})
        self.assertNotEqual(action, base.REPRODUCE)
        # NN/GN still use unmodified legal neural birth decision.
        weights = [0.] * base.genome_size(self.s.hidden)
        head = self.s.hidden * (base.INPUTS + self.s.hidden + 1)
        weights[head + 4 * (self.s.hidden + 1) + self.s.hidden] = 10.
        a.genome = tuple(weights)
        action, info = trial.choose(eco, a, 5, "GN", {})
        self.assertEqual(action, base.REPRODUCE)
        self.assertEqual(info["neural_birth_eligible_wish"], 1)

    def test_greedy_motor_intervention_ignores_genome_direction(self):
        eco = base.create_ecology(700, self.s, "heritable_mutation")
        a = next(x for x in eco.agents.values()
                 if base.blocked_and_free(eco, x)[1])
        blocked, free, _, motors, _ = controls._candidate_actions(eco, a)
        chosen_target = free[-1]
        expected_direction = base.DIRECTIONS.index(
            (chosen_target[0] - a.position[0],
             chosen_target[1] - a.position[1])
        )
        eco.food = {chosen_target}
        a.age, a.energy = 0, 25
        a.genome = tuple(0. for _ in a.genome)
        action, meta = trial.choose(eco, a, 2, "GN", {})
        self.assertEqual(action, expected_direction)
        self.assertEqual(meta["neural_birth_eligible_wish"], 0)
        # Giving the genotype a strong opposite motor bias cannot alter
        # greedy navigation (as long as it does not ask for reproduction).
        weights = [0.] * len(a.genome)
        head = self.s.hidden * (base.INPUTS + self.s.hidden + 1)
        weights[head + motors[0]*(self.s.hidden+1)+self.s.hidden] = 11.
        a.genome = tuple(weights)
        again, _ = trial.choose(eco, a, 2, "GN", {})
        self.assertEqual(again, expected_direction)

    def test_all_arms_balanced_deterministic_no_movement_through_walls(self):
        for arm in trial.ARMS:
            a = trial.simulate(702, self.s, arm, detailed=True)
            b = trial.simulate(702, self.s, arm, detailed=True)
            self.assertEqual(a, b)
            result, checkpoints, trace = a
            self.assertAlmostEqual(result["energy_ledger_residual"], 0.)
            self.assertTrue(all(
                abs(x["energy_ledger_residual"]) < 1e-6
                for x in checkpoints
            ))
            self.assertEqual(
                result["births"], result["action_births"]
            )
            self.assertEqual(
                result["food_eaten"], result["action_food_consumed"]
            )
            self.assertTrue(all(
                tuple(e["location"]) not in set(map(tuple, trace["walls"]))
                for e in trace["events"] if e["type"] == "birth"
            ))

    def test_world_level_factorial_contrasts_not_action_pseudoreplication(self):
        rows = []
        for seed, scores in (
            (1, dict(NN=0, NT=1, GN=0, GT=1)),
            (2, dict(NN=0, NT=0, GN=1, GT=1)),
            (3, dict(NN=0, NT=1, GN=1, GT=1)),
        ):
            rows.extend(
                dict(seed=seed, split="main", arm=arm, survived=value)
                for arm, value in scores.items()
            )
        cmp = trial.comparisons(rows, "main", "survived", 88)
        self.assertAlmostEqual(cmp["NT_minus_NN"]["mean_delta"], 2/3)
        self.assertAlmostEqual(cmp["GN_minus_NN"]["mean_delta"], 2/3)
        self.assertAlmostEqual(cmp["GT_minus_NN"]["mean_delta"], 1.)
        self.assertAlmostEqual(
            cmp["interaction_threshold_by_motor"]["mean_delta"], -1/3
        )
        self.assertEqual(cmp["GT_minus_NN"]["positive_worlds"], 3)

    def test_full_small_campaign_has_reproducible_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, q = Path(tmp) / "a", Path(tmp) / "b"
            first, second = trial.run(self.s, p), trial.run(self.s, q)
            self.assertEqual(first, second)
            for file in (
                "summary.json", "world_metrics.csv",
                "population_checkpoints.csv", "population_curves.svg",
                "events_main_NN.json", "events_shifted_GT.json",
            ):
                self.assertEqual((p / file).read_bytes(),
                                 (q / file).read_bytes())
            with (p / "world_metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4 * (self.s.seeds +
                                            self.s.shifted_seeds))
            self.assertEqual(
                set(first["main_primary_survival"]),
                {"NT_minus_NN", "GN_minus_NN", "GT_minus_NT",
                 "GT_minus_GN", "GT_minus_NN",
                 "interaction_threshold_by_motor"}
            )


if __name__ == "__main__":
    unittest.main()
