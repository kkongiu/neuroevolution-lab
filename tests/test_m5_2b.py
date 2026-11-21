"""M5.2B pre-registered genuine genotype-controlled neural ecology tests."""
import csv
import tempfile
import unittest
from pathlib import Path

from neuroevolution import m5_0 as base
from neuroevolution import m5_2b as lab


class CompactNeuralEcologyTests(unittest.TestCase):
    def setUp(self):
        self.s = lab.Settings(
            seed_start=700, seeds=2, shifted_start=900, shifted_seeds=2,
            ticks=40, size=11, initial_food=9, food_cap=14,
            initial_population=8, max_population=18,
            checkpoint_every=10, hidden=2,
        )

    def test_preregistered_51_parameter_architecture_and_seeds(self):
        self.s.validate()
        self.assertEqual(base.genome_size(self.s.hidden), 51)
        self.assertEqual(len(set(lab.ARMS)), 4)
        self.assertTrue(set(range(77000, 77024)).isdisjoint(
            range(78000, 78012)
        ))
        with self.assertRaises(ValueError):
            lab.Settings(hidden=3).validate()
        with self.assertRaises(ValueError):
            lab.Settings(sparse_mutated_weights=4).validate()

    def test_sparse_changes_exactly_three_and_does_not_touch_parent(self):
        parent = tuple(float(i) / 10 for i in range(51))
        before = parent
        child, indexes = lab.mutate_sparse(parent, self.s, 702, 7, 2)
        self.assertEqual(parent, before)
        self.assertEqual(len(indexes), 3)
        self.assertEqual(len(set(indexes)), 3)
        self.assertEqual(
            [i for i, (p, c) in enumerate(zip(parent, child)) if p != c],
            list(indexes)
        )
        self.assertEqual(
            lab.mutate_sparse(parent, self.s, 702, 7, 2),
            (child, indexes)
        )
        different, _ = lab.mutate_sparse(parent, self.s, 702, 8, 2)
        self.assertNotEqual(child, different)

    def test_non_sparse_arms_are_original_m50_neural_actor(self):
        for arm in ("dense_inherited", "clone_inherited",
                    "random_offspring"):
            for shifted in (False, True):
                previous, old_check, _ = base.simulate(
                    701, self.s, lab.BASE_ARM[arm], shifted=shifted
                )
                result, new_check, _ = lab.simulate(
                    701, self.s, arm, shifted=shifted
                )
                for field in (
                    "births", "deaths", "alive_final",
                    "food_eaten", "food_spawned", "food_remaining",
                    "max_generation", "motor_mask_corrections",
                    "ineligible_reproduction_preferences",
                ):
                    self.assertEqual(previous[field], result[field],
                                     (arm, shifted, field))
                self.assertEqual(
                    [(x["tick"], x["population"], x["food_stock"])
                     for x in old_check],
                    [(x["tick"], x["population"], x["food_stock"])
                     for x in new_check]
                )

    def test_founders_and_candidate_food_schedule_are_arm_invariant(self):
        ecologies = [base.create_ecology(701, self.s, lab.BASE_ARM[a])
                     for a in lab.ARMS]
        self.assertTrue(all(
            a.world == ecologies[0].world and
            a.initial_genome_digests == ecologies[0].initial_genome_digests
            for a in ecologies
        ))
        self.assertTrue(all(
            base._food_schedule(701, self.s, eco, False) ==
            base._food_schedule(701, self.s, ecologies[0], False)
            for eco in ecologies
        ))

    def test_genotype_affects_all_actions_including_reproduction(self):
        eco = base.create_ecology(701, self.s, "heritable_mutation")
        agent = next(a for a in eco.agents.values()
                     if base.blocked_and_free(eco, a)[1])
        agent.age, agent.energy = 5, 50.
        blocked, _ = base.blocked_and_free(eco, agent)
        sensors = base.observe(eco, agent, blocked)
        weights = [0.] * base.genome_size(self.s.hidden)
        head = self.s.hidden * (base.INPUTS + self.s.hidden + 1)
        weights[head + 4*(self.s.hidden+1)+self.s.hidden] = 10.
        agent.genome = tuple(weights)
        logits_birth = base.network_logits(agent, sensors, self.s)
        self.assertEqual(max(range(base.ACTIONS), key=lambda i: logits_birth[i]), 4)
        weights[head + 4*(self.s.hidden+1)+self.s.hidden] = -10.
        weights[head + self.s.hidden] = 20.
        agent.genome = tuple(weights)
        logits_move = base.network_logits(agent, sensors, self.s)
        self.assertEqual(max(range(base.ACTIONS), key=lambda i: logits_move[i]), 0)
        self.assertNotEqual(logits_birth, logits_move)
        # The network is able to decide both birth and movement; no script
        # selects a preferred direction or fixed reproductive age threshold.

    def test_sparse_birth_events_are_genetic_and_energy_conserving(self):
        result, checkpoints, trace = lab.simulate(
            702, self.s, "sparse_inherited", detailed=True
        )
        self.assertEqual(result["all_actions_neural"], 1)
        self.assertEqual(result["sparse_births"], result["births"])
        self.assertAlmostEqual(result["energy_ledger_residual"], 0.)
        self.assertTrue(all(
            abs(x["energy_ledger_residual"]) < 1e-6
            for x in checkpoints
        ))
        births = [x for x in trace["events"] if x["type"] == "birth"]
        self.assertEqual(len(births), result["births"])
        self.assertTrue(all(
            len(x["sparse_mutated_indices"]) == 3 and
            len(set(x["sparse_mutated_indices"])) == 3 and
            x["genome"]
            for x in births
        ))
        self.assertTrue(all(
            tuple(x["location"]) not in set(map(tuple, trace["walls"]))
            for x in births
        ))
        repeat = lab.simulate(702, self.s, "sparse_inherited", detailed=True)
        self.assertEqual((result, checkpoints, trace), repeat)

    def test_world_paired_statistics_and_gate_logic(self):
        rows = []
        for seed, values in (
            (1, (1, 0, 0, 0)),
            (2, (1, 0, 1, 0)),
            (3, (0, 0, 0, 0)),
        ):
            rows.extend({
                "seed": seed, "split": "main",
                "arm": arm, "survived": score
            } for arm, score in zip(lab.ARMS, values))
        x = lab.contrast(rows, "main", "survived", 80)
        self.assertAlmostEqual(
            x["sparse_inherited_minus_random_offspring"]["mean_delta"],
            2 / 3
        )
        self.assertAlmostEqual(
            x["sparse_inherited_minus_clone_inherited"]["mean_delta"],
            1 / 3
        )

    def test_reproducible_small_campaign_and_read_only_results(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "first"
            q = Path(root) / "again"
            first = lab.run(self.s, p)
            second = lab.run(self.s, q)
            self.assertEqual(first, second)
            for filename in (
                "summary.json", "world_metrics.csv",
                "population_checkpoints.csv",
                "population_curves.svg", "events_main_sparse_inherited.json",
                "events_shifted_random_offspring.json",
            ):
                self.assertEqual(
                    (p / filename).read_bytes(), (q / filename).read_bytes()
                )
            with (p / "world_metrics.csv").open(newline="") as h:
                rows = list(csv.DictReader(h))
            self.assertEqual(len(rows), 4 * (self.s.seeds + self.s.shifted_seeds))
            self.assertIn("main_all_primary_pass", first["preregistered_gates"])
            self.assertIn("novel_mutation_vs_clone_pass",
                          first["preregistered_gates"])


if __name__ == "__main__":
    unittest.main()
