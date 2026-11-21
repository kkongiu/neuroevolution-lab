"""M5.5C source/heldout isolation and exact nine-gene clone transplant tests."""
import json
import tempfile
import unittest
from pathlib import Path

from neuroevolution import m5_5b as old
from neuroevolution import m5_5c as new


class OwnFounderTransfer(unittest.TestCase):
    def setUp(self):
        founder = tuple(0.1 + i * 0.02 for i in range(9))
        mutant = tuple(x + (0.10 if i in (3, 8) else 0.)
                       for i, x in enumerate(founder))
        rng = tuple(-0.2 + i * 0.05 for i in range(9))
        self.candidates = []
        for k in new.SOURCE_SEEDS:
            self.candidates.append({
                "source_seed": k,
                "own_founder_id": k % 20,
                "newborn_child_id": 21 + k,
                "own_original_founder_genes": list(founder),
                "actual_mutant_descendant_genes": list(mutant),
                "matched_random_control_genes": list(rng),
                "founder_mutant_pair_sha256": new.digest({
                    "founder": founder, "mutant": mutant,
                }),
            })
        self.bank = {
            "protocol_commit": new.PROTOCOL_COMMIT,
            "M5_5B_source_run_commit": new.SOURCE_STUDY_COMMIT,
            "original_source_world_seeds": list(new.SOURCE_SEEDS),
            "heldout_transfer_worlds_examined": 0,
            "candidates": self.candidates,
            "candidate_bank_sha256": new.digest(self.candidates),
        }

    def test_predeclared_source_and_target_worlds_disjoint(self):
        self.assertEqual(new.SOURCE_SEEDS, tuple(range(94000, 94008)))
        self.assertEqual(new.TRANSFER_MAIN, tuple(range(94200, 94224)))
        self.assertEqual(new.TRANSFER_HARD, tuple(range(94300, 94312)))
        self.assertFalse(set(new.SOURCE_SEEDS) &
                         (set(new.TRANSFER_MAIN) | set(new.TRANSFER_HARD)))
        self.assertEqual(len(new.ARMS), 6)
        self.assertEqual(new.BOOT_REPS, 20000)
        self.assertEqual(old.N_GENES, 9)

    def test_source_family_full_nine_gene_exact_reversions(self):
        variants = new.genotypes(self.candidates[0])
        self.assertEqual(variants[new.FOUNDER], variants[new.FULL_REVERT])
        self.assertEqual(variants[new.FEAR_REVERT][:8],
                         variants[new.MUTANT][:8])
        self.assertEqual(variants[new.FEAR_REVERT][8],
                         variants[new.FOUNDER][8])
        self.assertEqual(variants[new.FEAR_ONLY][:8],
                         variants[new.FOUNDER][:8])
        self.assertEqual(variants[new.FEAR_ONLY][8],
                         variants[new.MUTANT][8])
        self.assertNotEqual(variants[new.RANDOM], variants[new.FOUNDER])

    def test_frozen_bank_hash_provenance_and_exact_world_fence(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"bank.json"
            p.write_text(json.dumps(self.bank), encoding="utf-8")
            self.assertEqual(
                new.load_frozen_bank(p)["candidate_bank_sha256"],
                self.bank["candidate_bank_sha256"])
            value = json.loads(p.read_text())
            value["candidates"][0]["actual_mutant_descendant_genes"][8] += .1
            p.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(AssertionError):
                new.load_frozen_bank(p)
        with self.assertRaises(ValueError):
            new.transfer_world(94000, self.candidates[0], new.MUTANT,
                               hard=False, ticks=400)
        with self.assertRaises(ValueError):
            new.transfer_world(94200, self.candidates[0], new.MUTANT,
                               hard=True, ticks=400)
        with self.assertRaises(ValueError):
            new.transfer_world(94300, self.candidates[0], new.MUTANT,
                               hard=False, ticks=400)

    def test_transfer_same_original_physics_for_all_genotypes(self):
        source = self.candidates[0]
        rows = {}
        check = {}
        for arm in new.ARMS:
            row, snapshots = new.transfer_world(
                300, source, arm, hard=False, ticks=25, checkpoint_every=5)
            rows[arm] = row
            check[arm] = snapshots
            self.assertEqual(row["transfer_birth_mutations"], 0)
            self.assertEqual(row["births"], row["original_costed_births"])
            self.assertLess(abs(row["energy_ledger_residual"]), 1e-6)
            self.assertEqual(len(snapshots), 6)
            self.assertTrue(all(abs(
                s["energy_ledger_residual"]) < 1e-6 for s in snapshots))
        self.assertEqual(len({r["world_sha"] for r in rows.values()}), 1)
        self.assertEqual(
            len({r["predator_initial_positions"]
                 for r in rows.values()}), 1)
        self.assertEqual(
            rows[new.FOUNDER]["genotype_sha"],
            rows[new.FULL_REVERT]["genotype_sha"])
        for key in ("survived", "alive_final", "births", "food_eaten",
                    "deaths", "predation_deaths", "observed_enemies",
                    "population_auc"):
            self.assertEqual(rows[new.FOUNDER][key],
                             rows[new.FULL_REVERT][key])
        self.assertEqual(
            [s["population"] for s in check[new.FOUNDER]],
            [s["population"] for s in check[new.FULL_REVERT]])

    def test_evaluation_source_family_not_individual_tick_bootstrap(self):
        rows = []
        for source in self.candidates:
            for split, seeds in (("main", (300, 301)),
                                 ("hard", (400, 401))):
                for seed in seeds:
                    for arm in new.ARMS:
                        row, _ = new.transfer_world(
                            seed, source, arm, hard=(split == "hard"),
                            ticks=8, checkpoint_every=4)
                        rows.append(row)
        analyzed = new.analyze(
            rows, self.bank, (300, 301), (400, 401), bootstrap_reps=100)
        self.assertEqual(analyzed["main"]["independent_source_families"], 8)
        self.assertEqual(
            analyzed["main"]["nested_transfer_seeds_per_source"], 2)
        for split in ("main", "hard"):
            for measure in ("survived", "population_auc"):
                self.assertEqual(analyzed[split]["comparisons"][
                    "full_reversion_minus_own_founder_"+measure
                ]["mean_family_paired_effect"], 0)
        self.assertFalse(analyzed["gates"]["new_avoidance_behavior_evolved"])
        rows.pop()
        with self.assertRaises(AssertionError):
            new.analyze(rows, self.bank, (300, 301), (400, 401),
                        bootstrap_reps=100)


if __name__ == "__main__":
    unittest.main()
