"""M5.5B independent enemies: deterministic physics, genes and controls."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from neuroevolution import m5_0 as base
from neuroevolution import m5_5a as old
from neuroevolution import m5_5b as new


class PredatorFeasibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small = new.Settings(
            seed_start=300, seeds=2, shifted_start=400, shifted_seeds=1,
            ticks=30, size=11, initial_food=9, food_cap=14,
            initial_population=8, max_population=18,
            checkpoint_every=10, no_food_controls=0,
        )
        cls.small.validate()

    def test_preregistered_fences_and_original_protocol_untouched(self):
        full = new.Settings()
        full.validate()
        self.assertEqual(full.seed_start, 94000)
        self.assertEqual(full.shifted_start, 94100)
        self.assertEqual(new.PREDATOR_LEVELS, (0, 1, 3))
        self.assertEqual(new.N_GENES, 9)
        self.assertEqual(new.BOOT_REPS, 20000)
        for settings in (
            replace(full, seed_start=93000),
            replace(full, seed_start=94200),
            replace(full, shifted_start=94300),
            replace(full, seeds=9),
            replace(full, odor_radius=3),
            replace(full, innate_mutation_sd=.3),
        ):
            with self.assertRaises(ValueError):
                settings.validate()
        self.assertEqual(old.N_GENES, 8)
        self.assertEqual(old.Settings().seed_start, 93000)

    def test_exact_original_eight_source_genes_and_ninth_independent(self):
        s = self.small
        a = new.founder_genes(300, s)
        self.assertEqual(a, new.founder_genes(300, s))
        self.assertNotEqual(a, new.founder_genes(301, s))
        self.assertEqual(tuple(g[:8] for g in a), old.founder_genes(300, s))
        self.assertEqual(len(a), s.initial_population)
        self.assertTrue(all(len(g) == 9 for g in a))
        with self.assertRaises(ValueError):
            new.gene_hash((0.,) * 8)
        self.assertEqual(len({new.gene_hash(g) for g in a}), len(a))

    def test_only_real_birth_perturbs_exactly_registered_gene_indices(self):
        s = self.small
        g = tuple(i / 10 for i in range(9))
        for arm in new.ARMS:
            child, changed = new.child_genes(g, 300, 10, 4, arm, s)
            self.assertEqual((child, changed),
                             new.child_genes(g, 300, 10, 4, arm, s))
            if arm in (new.CLONE, new.BLIND):
                self.assertIs(child, g)
                self.assertEqual(changed, [])
            elif arm == new.SPARSE:
                self.assertEqual(len(changed), 2)
            else:
                self.assertEqual(len(changed), 9)
            if changed:
                self.assertTrue(all(child[i] != g[i] for i in changed))
        for arm in (new.SPARSE, new.DENSE):
            child, _ = new.child_genes((2.5,) * 9, 300, 11, 6, arm, s)
            self.assertTrue(all(-2.5 <= x <= 2.5 for x in child))

    def test_zero_enemy_exact_legacy_innate_choice_including_noise(self):
        s = self.small
        eco = base.create_ecology(300, s, "heritable_no_mutation")
        gene = new.founder_genes(300, s)[0]
        a = eco.agents[0]
        a.genome = gene
        old_agent = replace(a, genome=gene[:8])
        visits = {a.position: 1}
        for tick in range(1, 40):
            choice, obs = new.choose_action(
                eco, a, tick, visits, None, [], new.CLONE)
            ref, _ = old.choose_action(
                eco, old_agent, tick, visits, None, old.CLONE)
            self.assertEqual(choice, ref)
            self.assertEqual(obs["enemies_visible"], 0)
            blind, _ = new.choose_action(
                eco, a, tick, visits, None, [], new.BLIND)
            self.assertEqual(blind, choice)

    def test_local_fear_can_affect_selection_but_no_omniscience(self):
        s = self.small
        eco = base.create_ecology(300, s, "heritable_no_mutation")
        a = eco.agents[0]
        a.position = (5, 5)
        a.energy = 5.
        a.genome = (0.,) * 8 + (2.5,)
        with (patch.object(base, "blocked_and_free", return_value=(
                [False] * 4, [(5, 4), (6, 5), (5, 6), (4, 5)])),
              patch.object(base, "observe", return_value=(0.,) * 15),
              patch.object(base, "can_reproduce", return_value=False)):
            observed_diff = False
            for tick in range(1, 100):
                fearful, info = new.choose_action(
                    eco, a, tick, {(5, 5): 1}, None,
                    [(6, 5)], new.CLONE)
                blind, _ = new.choose_action(
                    eco, a, tick, {(5, 5): 1}, None,
                    [(6, 5)], new.BLIND)
                unseen, unseen_info = new.choose_action(
                    eco, a, tick, {(5, 5): 1}, None,
                    [(9, 9)], new.CLONE)
                empty, _ = new.choose_action(
                    eco, a, tick, {(5, 5): 1}, None,
                    [], new.CLONE)
                self.assertEqual(unseen, empty)
                self.assertEqual(unseen_info["enemies_visible"], 0)
                self.assertEqual(info["enemies_visible"], 1)
                self.assertNotEqual(fearful, 1)  # don't run into predator east
                observed_diff |= fearful != blind
            self.assertTrue(observed_diff)

    def test_physical_capture_removes_energy_exactly_once(self):
        s = self.small
        eco = base.create_ecology(300, s, "heritable_no_mutation")
        a = eco.agents[0]
        a.genome = new.founder_genes(300, s)[0]
        energy = a.energy
        enemies = [a.position]
        count = new.step_predators(eco, enemies, 1)
        self.assertEqual(count, 1)
        self.assertNotIn(a.id, eco.agents)
        self.assertEqual(eco.cumulative_death_discard, energy)
        self.assertEqual(eco.deaths, 1)
        self.assertEqual(eco.events[-1]["type"], "predation")
        self.assertEqual(new.step_predators(eco, [], 2), 0)

    def test_exact_source_world_and_physical_energy_for_all_controls(self):
        s = self.small
        for predators in new.PREDATOR_LEVELS:
            rows = []
            for arm in new.ARMS:
                row, ck, detail = new.simulate(
                    300, s, arm, predators, detailed=True)
                rows.append(row)
                self.assertEqual(row["births"],
                                 row["original_costed_birth_actions"])
                self.assertEqual(row["births"],
                                 len(detail["full_birth_genealogy"]))
                self.assertEqual(row["predation_deaths"],
                                 len(detail["predation_events"]))
                self.assertTrue(all(abs(c["energy_ledger_residual"]) < 1e-6
                                    for c in ck))
                self.assertAlmostEqual(row["energy_ledger_residual"], 0.)
                if predators == 0:
                    self.assertEqual(row["predation_deaths"], 0)
                if arm in (new.CLONE, new.BLIND):
                    self.assertEqual(row["mutated_births"], 0)
                if arm == new.SPARSE:
                    self.assertEqual(row["mutated_births"], row["births"])
                    self.assertTrue(all(
                        len(g["changed_indices"]) == 2
                        for g in detail["full_birth_genealogy"]))
                if arm == new.DENSE:
                    self.assertEqual(row["mutated_births"], row["births"])
            for key in ("world_sha", "founder_sha",
                        "predator_initial_positions", "original_world_sha"):
                self.assertEqual(len({r[key] for r in rows}), 1)

    def test_small_exploration_is_reproducible_and_has_complete_arms(self):
        s = replace(self.small, seeds=1, shifted_seeds=1, ticks=15,
                    checkpoint_every=5)
        rows = []
        for split, seed, hard in (("main", 300, False), ("hard", 400, True)):
            for pressure in new.PREDATOR_LEVELS:
                for arm in new.ARMS:
                    one = new.simulate(seed, s, arm, pressure, shifted=hard)
                    two = new.simulate(seed, s, arm, pressure, shifted=hard)
                    self.assertEqual(one, two)
                    rows.append(one[0])
        result = new.summarize(rows, s, bootstrap_reps=100)
        self.assertEqual(result["adaptation_proven"], False)
        self.assertTrue(result["causal_lineage_transplant_needed"])
        self.assertEqual(result["main"]["0"]["worlds"], 1)
        self.assertEqual(result["hard"]["3"]["worlds"], 1)
        rows.pop()
        with self.assertRaises(AssertionError):
            new.summarize(rows, s, bootstrap_reps=100)


if __name__ == "__main__":
    unittest.main()
