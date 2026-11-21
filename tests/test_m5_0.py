import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.m5_0 import (
    ACTIONS, ARMS, INPUTS, REPRODUCE, Settings, _food_schedule, act,
    blocked_and_free, create_ecology, energy_balance, genome_cosine,
    genome_size, initial_genome, make_child_genome, observe, run, simulate,
)


class PersistentEcologyTests(unittest.TestCase):
    def setUp(self):
        self.s = Settings(
            seed_start=700, seeds=2,
            shifted_start=900, shifted_seeds=2,
            ticks=40, size=11,
            initial_food=9, food_cap=14,
            initial_population=8, max_population=18,
            checkpoint_every=10, no_food_controls=2,
            hidden=4,
        )

    def test_parameter_and_partition_validation(self):
        self.s.validate()
        with self.assertRaises(ValueError):
            Settings(seed_start=100, seeds=5,
                     shifted_start=102).validate()
        with self.assertRaises(ValueError):
            Settings(max_population=20,
                     initial_population=20).validate()
        with self.assertRaises(ValueError):
            Settings(reproduction_energy_cost=11,
                     newborn_energy=12).validate()
        with self.assertRaises(ValueError):
            Settings(no_food_controls=13).validate()
        self.assertEqual(genome_size(self.s.hidden),
                         self.s.hidden * (INPUTS + self.s.hidden + 1) +
                         ACTIONS * (self.s.hidden + 1))

    def test_initial_population_heredity_same_across_all_arms(self):
        worlds = [create_ecology(700, self.s, a) for a in ARMS]
        self.assertTrue(all(
            eco.world == worlds[0].world and
            eco.initial_genome_digests == worlds[0].initial_genome_digests
            for eco in worlds
        ))
        self.assertTrue(all(
            {x.position for x in eco.agents.values()}.isdisjoint(eco.food)
            for eco in worlds
        ))
        self.assertTrue(all(
            a.position in worlds[0].world.reachable()
            for a in worlds[0].agents.values()
        ))
        self.assertEqual(len(worlds[0].agents), self.s.initial_population)
        for eco in worlds:
            self.assertAlmostEqual(energy_balance(eco), 0, places=8)
            self.assertEqual(len(eco.food), self.s.initial_food)
        schedules = [_food_schedule(700, self.s, eco, False) for eco in worlds]
        self.assertTrue(all(x == schedules[0] for x in schedules))
        self.assertEqual(len(schedules[0]), self.s.ticks)

    def test_local_sensor_no_map_identity_or_future_state(self):
        eco = create_ecology(701, self.s, "heritable_mutation")
        agent = next(iter(eco.agents.values()))
        blocked, free = blocked_and_free(eco, agent)
        sensory = observe(eco, agent, blocked)
        self.assertEqual(len(sensory), INPUTS)
        self.assertTrue(all(0. <= x <= 1. for x in sensory))
        eco.food.clear()
        no_food = observe(eco, agent, blocked)
        self.assertEqual(no_food[4:12], (0.,) * 8)
        self.assertEqual(sensory[:4], no_food[:4])
        self.assertEqual(sensory[-3:], no_food[-3:])

    def test_birth_is_chosen_by_network_not_automatically(self):
        eco = create_ecology(702, self.s, "heritable_mutation")
        agent = next(iter(eco.agents.values()))
        agent.age = self.s.min_reproduction_age
        # Engineered logits are a *test fixture* only, not an organism given
        # to the scientific campaign. The genome prefers reproduce action.
        genome = [0.] * genome_size(self.s.hidden)
        output = self.s.hidden * (INPUTS + self.s.hidden + 1)
        genome[output + REPRODUCE*(self.s.hidden+1)+self.s.hidden] = 20.
        agent.genome = tuple(genome)
        before_ids = set(eco.agents)
        before_energy = agent.energy
        act(eco, agent.id, 1)
        born = set(eco.agents) - before_ids
        self.assertEqual(len(born), 1)
        self.assertEqual(eco.births, 1)
        child = eco.agents[born.pop()]
        self.assertEqual(child.parent, agent.id)
        self.assertEqual(child.founder, agent.founder)
        self.assertEqual(child.generation, 1)
        self.assertEqual(child.age, 0)
        self.assertEqual(child.energy, self.s.newborn_energy)
        self.assertEqual(child.hidden, [0.]*self.s.hidden)
        self.assertNotEqual(child.position, agent.position)
        self.assertAlmostEqual(
            agent.energy,
            before_energy - self.s.movement_cost -
            self.s.reproduction_energy_cost
        )
        self.assertAlmostEqual(energy_balance(eco), 0, places=8)
        self.assertEqual(eco.events[0]["type"], "birth")
        self.assertNotEqual(child.genome, agent.genome)

    def test_offspring_modes_and_energy_conservation(self):
        p = initial_genome(random.Random(9), self.s)
        cloned = make_child_genome(
            p, self.s, "heritable_no_mutation", random.Random(10)
        )
        mutated = make_child_genome(
            p, self.s, "heritable_mutation", random.Random(10)
        )
        redrawn = make_child_genome(
            p, self.s, "random_offspring", random.Random(10)
        )
        self.assertEqual(p, cloned)
        self.assertNotEqual(p, mutated)
        self.assertNotEqual(p, redrawn)
        self.assertEqual(len(mutated), len(redrawn))
        self.assertAlmostEqual(genome_cosine(p, cloned), 1., places=10)
        self.assertGreater(genome_cosine(p, mutated),
                           genome_cosine(p, redrawn))
        with self.assertRaises(ValueError):
            make_child_genome(p, self.s, "unknown", random.Random(10))

    def test_food_free_world_never_spawns_births_or_energy(self):
        for seed in (700, 701):
            r, cks, trace = simulate(
                seed, self.s, "heritable_mutation",
                no_food=True, detailed=True
            )
            self.assertEqual(r["births"], 0)
            self.assertEqual(r["food_eaten"], 0)
            self.assertEqual(r["food_spawned"], 0)
            self.assertEqual(r["alive_final"], 0)
            self.assertIsNotNone(r["extinct_at"])
            self.assertEqual(r["final_energy_balance_residual"], 0.)
            self.assertTrue(all(
                item["energy_ledger_residual"] == 0.
                for item in cks
            ))

    def test_no_collisions_renewable_resources_and_population_caps(self):
        settings = Settings(**{
            **self.s.__dict__, "ticks": 70,
            "initial_energy": 40,
            "initial_population": 8,
            "max_population": 14,
        })
        r, cks, trace = simulate(
            703, settings, "heritable_mutation", detailed=True
        )
        self.assertLessEqual(max(x["population"] for x in cks),
                             settings.max_population)
        self.assertGreaterEqual(r["food_spawned"], settings.initial_food)
        self.assertLessEqual(r["food_remaining"], settings.food_cap)
        self.assertEqual(r["birth_actions"], r["births"])
        self.assertEqual(r["deaths"],
                         r["deaths_starvation"] + r["deaths_age"])
        self.assertAlmostEqual(r["final_energy_balance_residual"], 0., places=8)
        self.assertEqual(cks[-1]["tick"], settings.ticks)
        self.assertGreaterEqual(r["food_eaten"], 0)
        self.assertTrue(all(
            p["energy_ledger_residual"] < 1e-6 for p in cks
        ))
        self.assertTrue(all(
            event["location"] not in trace["walls"]
            for event in trace["event_log"]
            if event["type"] == "birth"
        ))

    def test_seed_reproducibility_across_arms_and_shift(self):
        for shifted in (False, True):
            for arm in ARMS:
                a = simulate(708, self.s, arm,
                             shifted=shifted, detailed=True)
                b = simulate(708, self.s, arm,
                             shifted=shifted, detailed=True)
                self.assertEqual(a, b)
                self.assertEqual(a[0]["seed"], 708)
                self.assertEqual(len(a[1]), 5)

    def test_complete_campaign_reproducible_and_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            s1, s2 = run(self.s, a), run(self.s, b)
            self.assertEqual(s1, s2)
            self.assertTrue(s1["no_external_fitness_or_champion_selection"])
            self.assertTrue(s1["no_food"]["all_extinct"])
            self.assertEqual(
                s1["primary_heritable_minus_random_offspring"]["paired_worlds"],
                self.s.seeds
            )
            for name in (
                "summary.json", "seed_metrics.csv",
                "population_checkpoints.csv",
                "lineages_main_heritable_mutation.json",
                "ecosystem_main.svg", "ecosystem_shifted.svg",
            ):
                self.assertEqual((a/name).read_bytes(), (b/name).read_bytes())
            with (a/"seed_metrics.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3*(self.s.seeds+self.s.shifted_seeds) +
                             self.s.no_food_controls)
            with (a/"population_checkpoints.csv").open(newline="") as f:
                records = list(csv.DictReader(f))
            self.assertEqual(len(records), 3*(self.s.seeds+self.s.shifted_seeds)*5)
            self.assertTrue(all(float(x["energy_ledger_residual"]) < 1e-6
                                for x in records))
            self.assertIn("<svg", (a/"ecosystem_main.svg").read_text())


if __name__ == "__main__":
    unittest.main()
