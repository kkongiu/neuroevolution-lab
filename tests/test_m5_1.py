import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution import m5_0 as base
from neuroevolution.m5_1 import (
    ACTIVE_ARMS, ALL_ARMS, NEGATIVE_ARMS, Settings,
    _act_scripted, _compare, scripted_choice, run, simulate,
)


class EcologicalViabilityTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings(
            seed_start=700, seeds=2,
            shifted_start=900, shifted_seeds=2,
            negative_control_worlds=2, ticks=40,
            size=11, initial_food=9, food_cap=14,
            initial_population=8, max_population=18,
            checkpoint_every=10, hidden=4,
        )

    def test_validation_and_split_constraints(self):
        self.s.validate()
        with self.assertRaises(ValueError):
            Settings(scripted_reproduction_threshold=55).validate()
        with self.assertRaises(ValueError):
            Settings(scripted_reproduction_threshold=30).validate()
        with self.assertRaises(ValueError):
            Settings(seed_start=73000, seeds=5,
                     shifted_start=73002).validate()
        with self.assertRaises(ValueError):
            Settings(negative_control_worlds=20).validate()
        self.assertEqual(set(ACTIVE_ARMS).intersection(NEGATIVE_ARMS), set())

    def test_same_world_genomes_spawn_schedule_across_arms(self):
        ecologies = [
            base.create_ecology(700, self.s, "heritable_mutation")
            for _ in ALL_ARMS
        ]
        self.assertTrue(all(e.world == ecologies[0].world
                            for e in ecologies))
        self.assertTrue(all(e.initial_genome_digests ==
                            ecologies[0].initial_genome_digests
                            for e in ecologies))
        self.assertTrue(all(
            base._food_schedule(700, self.s, e, False) ==
            base._food_schedule(700, self.s, ecologies[0], False)
            for e in ecologies
        ))

    def test_scripts_observe_local_food_no_hidden_map_or_genome(self):
        eco = base.create_ecology(700, self.s, "heritable_mutation")
        target_agent = next(
            a for a in eco.agents.values()
            if base.blocked_and_free(eco, a)[1]
        )
        _, free = base.blocked_and_free(eco, target_agent)
        target = free[0]
        eco.food = {target}
        target_agent.age = 0
        target_agent.energy = 25
        visits = {target_agent.position: 1}
        choices = (
            scripted_choice(eco, target_agent, "greedy_odor", 1, visits),
            scripted_choice(eco, target_agent, "greedy_novelty", 1, visits),
        )
        actual = (
            target[0] - target_agent.position[0],
            target[1] - target_agent.position[1],
        )
        expected = base.DIRECTIONS.index(actual)
        self.assertEqual(choices, (expected, expected))
        # Replacing the numerical genome cannot change scripted action.
        target_agent.genome = tuple([100.] * len(target_agent.genome))
        self.assertEqual(
            scripted_choice(eco, target_agent, "greedy_odor", 1, visits),
            expected
        )
        self.assertEqual(
            scripted_choice(eco, target_agent, "greedy_novelty", 1, visits),
            expected
        )

    def test_birth_physics_and_energy_ledger_exact(self):
        eco = base.create_ecology(701, self.s, "heritable_mutation")
        target_agent = next(
            a for a in eco.agents.values()
            if base.blocked_and_free(eco, a)[1]
        )
        target_agent.age = 5
        target_agent.energy = 50.
        before_energy = target_agent.energy
        choice = scripted_choice(
            eco, target_agent, "greedy_odor", 5,
            {target_agent.position: 1}
        )
        self.assertEqual(choice, base.REPRODUCE)
        old_ids = set(eco.agents)
        _act_scripted(eco, target_agent.id, 5, choice)
        newborn_ids = set(eco.agents) - old_ids
        self.assertEqual(len(newborn_ids), 1)
        child = eco.agents[newborn_ids.pop()]
        self.assertEqual(child.parent, target_agent.id)
        self.assertEqual(child.generation, 1)
        self.assertEqual(child.age, 0)
        self.assertEqual(child.hidden, [0.] * self.s.hidden)
        self.assertEqual(child.energy, self.s.newborn_energy)
        self.assertAlmostEqual(
            target_agent.energy,
            before_energy - self.s.movement_cost -
            self.s.reproduction_energy_cost
        )
        self.assertEqual(eco.births, 1)
        # Energy accounting reflects the external test's manual energy
        # intervention (setting agent.energy 36->50): adjust initial ledger.
        expected_residual = self.s.initial_energy - 50.
        self.assertAlmostEqual(base.energy_balance(eco), expected_residual)

    def test_original_neural_physics_identical_to_unmodified_m50(self):
        for seed in (700, 701, 704):
            old, old_check, _ = base.simulate(
                seed, self.s, "heritable_mutation"
            )
            new, new_check, _ = simulate(
                seed, self.s, "neural_heritable"
            )
            for measure in ("births", "deaths", "alive_final",
                            "food_eaten", "food_spawned",
                            "food_remaining", "max_generation",
                            "motor_mask_corrections",
                            "ineligible_reproduction_preferences"):
                self.assertEqual(new[measure], old[measure],
                                 (seed, measure))
            self.assertEqual(
                [x["population"] for x in old_check],
                [x["population"] for x in new_check]
            )
            self.assertAlmostEqual(new["energy_ledger_residual"], 0, places=7)

    def test_no_birth_and_no_regrowth_controls(self):
        # A no-birth population must eventually age out. A short 40-tick
        # smoke is not required to be extinct when lifespan remains 100.
        no_birth, _, _ = simulate(
            700, self.s, "greedy_no_reproduction"
        )
        self.assertEqual(no_birth["births"], 0)
        no_regen, _, _ = simulate(700, self.s, "greedy_no_regrowth")
        self.assertEqual(no_regen["food_spawned"], self.s.initial_food)
        aged = Settings(**{**self.s.__dict__, "max_age": 15})
        aged_res, aged_check, _ = simulate(
            700, aged, "greedy_no_reproduction"
        )
        self.assertEqual(aged_res["births"], 0)
        self.assertEqual(aged_res["alive_final"], 0)
        self.assertLessEqual(aged_res["extinct_at"], 15)

    def test_scripted_physics_energy_and_no_teleport(self):
        for arm in ("greedy_odor", "greedy_novelty", "random_legal"):
            row, checkpoints, trace = simulate(
                700, self.s, arm, detailed=True
            )
            self.assertTrue(all(abs(r["energy_ledger_residual"]) < 1e-6
                                for r in checkpoints))
            self.assertEqual(row["energy_ledger_residual"], 0.)
            self.assertEqual(row["births"], row["action_births"])
            self.assertEqual(row["food_eaten"], row["action_food_consumed"])
            self.assertEqual(len(checkpoints), 5)
            self.assertEqual(row["policy_genome_used"], 0)
            self.assertTrue(all(
                tuple(e["location"]) not in set(map(tuple,trace["walls"]))
                for e in trace["events"] if e["type"] == "birth"
            ))
            self.assertLessEqual(
                max(x["population"] for x in checkpoints),
                self.s.max_population
            )

    def test_deterministic_with_mismatched_shift_no_final_seed_tuning(self):
        for shifted in (False, True):
            for arm in ACTIVE_ARMS:
                one = simulate(708, self.s, arm, shifted=shifted,
                               detailed=True)
                two = simulate(708, self.s, arm, shifted=shifted,
                               detailed=True)
                self.assertEqual(one, two)
                self.assertEqual(one[0]["seed"], 708)

    def test_pairwise_comparison_is_over_worlds_not_actions(self):
        rows = [
            {"split":"main","arm":"a","seed":1,"food_eaten":10},
            {"split":"main","arm":"b","seed":1,"food_eaten":8},
            {"split":"main","arm":"a","seed":2,"food_eaten":3},
            {"split":"main","arm":"b","seed":2,"food_eaten":4},
        ]
        d=_compare(rows,"main","a","b","food_eaten",1000)
        self.assertEqual(d["mean_delta"], .5)
        self.assertEqual(d["positive_worlds"],1)
        self.assertEqual(d["negative_worlds"],1)

    def test_full_small_campaign_and_artifacts_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/"a"
            b=Path(tmp)/"b"
            result_a=run(self.s,a)
            result_b=run(self.s,b)
            self.assertEqual(result_a,result_b)
            self.assertEqual(
                set(result_a["mean"]["main"]),set(ALL_ARMS)
            )
            self.assertEqual(
                set(result_a["mean"]["shifted"]),set(ACTIVE_ARMS)
            )
            self.assertTrue(result_a["negative_controls"][
                "no_regrowth_initial_only"
            ])
            for name in (
                "summary.json","world_metrics.csv",
                "population_checkpoints.csv",
                "population_curves.svg",
                "ecosystem_main_greedy_novelty.svg",
                "events_main_neural_heritable.json",
                "events_shifted_greedy_novelty.json",
            ):
                self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())
            with (a/"world_metrics.csv").open(newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(
                len(rows), (self.s.seeds+self.s.shifted_seeds)*len(ACTIVE_ARMS)
                +self.s.negative_control_worlds*len(NEGATIVE_ARMS)
            )
            self.assertIn("<svg",(a/"population_curves.svg").read_text())
            self.assertIn("main",result_a["mean"])
            self.assertIn("shifted",result_a["mean"])


if __name__=="__main__":
    unittest.main()
