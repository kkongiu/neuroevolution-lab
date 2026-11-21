"""M5.4B0 external optimization is not within-world neural evolution."""
import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from neuroevolution import m5_0 as physical
from neuroevolution import m5_1 as human
from neuroevolution import m5_3b as original_genetics
from neuroevolution import m5_4b0 as b0


class ShortOdorNumericRepresentabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small=b0.Settings(
            development_start=200,development_seeds=2,
            validation_start=300,validation_seeds=2,
            seed_start=700,seeds=2,shifted_start=900,shifted_seeds=2,
            initial_pool=6,generations=1,offspring_per_generation=4,
            keep_elites=2,ticks=35,size=11,initial_food=9,food_cap=14,
            initial_population=8,max_population=18,checkpoint_every=10,
            no_food_controls=0
        )
        cls.small.validate()

    def test_sealed_protocol_and_unseen_world_boundaries(self):
        b0.Settings().validate()
        self.assertEqual(b0.BOOT_REPS,20000)
        self.assertEqual(b0.ALL_ARMS,(
            "witness","scrambled","random_0","random_1",
            "random_2","random_3","scripted_greedy_novelty"
        ))
        self.assertEqual(b0.Settings().odor_radius,2)
        self.assertEqual(b0.Settings().shifted_odor_radius,2)
        for changed in (
            replace(b0.Settings(),seed_start=86000),
            replace(b0.Settings(),validation_start=88200),
            replace(b0.Settings(),development_start=88300),
            replace(b0.Settings(),ticks=400,generations=4),
            replace(b0.Settings(),ticks=400,odor_radius=5),
            replace(b0.Settings(),random_controls=3),
        ):
            with self.assertRaises(ValueError):
                changed.validate()

    def test_native_numerical_actor_and_births_are_exact_clones(self):
        genes=(.6,1.2,.4,.1,-.1,.1,-.1,.3,0.,1.2)
        self.assertEqual(len(original_genetics.to_weights(genes)),401)
        for shifted in (False,True):
            s=self.small
            a=b0.simulate_neural(700,s,genes,shifted=shifted,detailed=True)
            bb=b0.simulate_neural(700,s,genes,shifted=shifted,detailed=True)
            self.assertEqual(a,bb)
            row,ck,trace=a
            self.assertEqual(row["all_actions_native_m50_neural"],1)
            self.assertEqual(row["programmed_policy"],0)
            self.assertEqual(row["internal_birth_mutations"],0)
            self.assertTrue(all(
                abs(r["energy_ledger_residual"])<1e-6 for r in ck
            ))
            digest=physical.genome_digest(
                original_genetics.to_weights(genes)
            )
            self.assertEqual(row["genotype_digest"],digest)
            for evt in trace["events"]:
                if evt["type"]=="birth":
                    self.assertEqual(evt["genome"],digest)
                    self.assertNotIn(tuple(evt["location"]),
                                     set(map(tuple,trace["walls"])))

    def test_unmodified_original_world_map_food_schedule(self):
        s=self.small
        g=(.6,1.2,.4,.1,-.1,.1,-.1,.3,0.,1.2)
        for hard in (False,True):
            source=physical.create_ecology(
                701,s,"heritable_no_mutation",shifted=hard
            )
            _,_,trace=b0.simulate_neural(
                701,s,g,shifted=hard,detailed=True
            )
            self.assertEqual(
                trace["walls"],[list(p) for p in sorted(source.world.walls)]
            )
            self.assertEqual(
                trace["initial_food"],[list(p) for p in source.world.initial_food]
            )
            self.assertEqual(
                trace["initial_founders"],
                [list(a.position) for a in source.agents.values()]
            )
            self.assertEqual(source.world.odor_radius,2)

    def test_researcher_programmed_script_is_clearly_separate(self):
        row,checks,trace=b0.scripted_ecological_control(
            700,self.small,detailed=True
        )
        self.assertEqual(row["all_actions_native_m50_neural"],0)
        self.assertEqual(row["programmed_policy"],1)
        self.assertEqual(row["internal_birth_mutations"],0)
        self.assertIsNone(row["genotype_digest"])
        self.assertEqual(row["energy_ledger_residual"],0.)
        self.assertTrue(all(
            abs(r["energy_ledger_residual"])<1e-6 for r in checks
        ))
        self.assertEqual(trace["arm"],"greedy_novelty")
        self.assertTrue(trace["walls"])

    def test_external_optimizer_uses_only_frozen_development_and_validation(self):
        observed=[]
        true=b0.simulate_neural
        def wrapper(seed,settings,genes,**kwargs):
            observed.append((seed,kwargs.get("shifted",False)))
            return true(seed,settings,genes,**kwargs)
        with patch.object(b0,"simulate_neural",side_effect=wrapper):
            a=b0.external_search(self.small)
        self.assertEqual(
            len(a[0]), self.small.initial_pool+
            self.small.generations*self.small.offspring_per_generation
        )
        self.assertEqual(
            len(a[1]),len(a[0])*self.small.development_seeds
        )
        self.assertEqual(
            len(a[2]),self.small.keep_elites*self.small.validation_seeds
        )
        self.assertTrue(all(
            seed in (200,201,300,301) and not shift
            for seed,shift in observed
        ))
        selected=a[3]
        self.assertIn(selected["candidate_id"],
                      a[4]["all_finalist_ids"])
        self.assertEqual(len(selected["gene_values"]),10)
        self.assertEqual(
            selected["gene_digest"],
            original_genetics.gene_digest(tuple(selected["gene_values"]))
        )
        self.assertEqual(
            selected["expanded_genome_digest"],
            physical.genome_digest(original_genetics.to_weights(
                tuple(selected["gene_values"])
            ))
        )
        self.assertTrue(all(
            e["parent_candidate_id"] is not None or
            e["external_optimization_round"]==0
            for e in a[0]
        ))

    def test_external_search_is_deterministic_and_controllers_distinct(self):
        a=b0.external_search(self.small)
        b=b0.external_search(self.small)
        self.assertEqual(a,b)
        bank=b0.fixed_bank(a[3],self.small)
        self.assertEqual(tuple(bank),b0.NEURAL_ARMS)
        self.assertEqual(len({tuple(x) for x in bank.values()}),6)
        self.assertEqual(sorted(bank["witness"]),
                         sorted(bank["scrambled"]))

    def test_paired_world_bootstrap_preserves_sign_and_zero(self):
        self.assertEqual(
            b0.paired_bootstrap([1.]*5,88701,200),(1.,1.)
        )
        self.assertEqual(
            b0.paired_bootstrap([0.]*5,88701,200),(0.,0.)
        )
        self.assertTrue(
            b0.paired_bootstrap([-1.]*5,88701,200)[1]<0
        )
        with self.assertRaises(ValueError):
            b0.paired_bootstrap([],88701,200)

    def test_two_complete_smoke_runs_are_identical_and_archive_nulls(self):
        with tempfile.TemporaryDirectory() as location:
            a=Path(location)/"a";b=Path(location)/"b"
            first=b0.run(self.small,a,boots=140)
            second=b0.run(self.small,b,boots=140)
            self.assertEqual(first,second)
            self.assertEqual(first["search_candidates"],10)
            self.assertEqual(first["development_trial_count"],20)
            self.assertEqual(first["validation_trial_count"],4)
            self.assertEqual(first["sealed_trial_count"],28)
            for filename in (
                "frozen_selected_genotype_and_controls.json",
                "external_search_all_96_gene_records.json",
                "external_search_development_metrics.csv",
                "external_search_validation_metrics.csv",
                "heldout_world_metrics.csv",
                "heldout_population_checkpoints.csv",
                "summary.json",
                "events_main_700_witness.json",
                "events_shifted_900_scripted_greedy_novelty.json"
            ):
                self.assertEqual(
                    (a/filename).read_bytes(),(b/filename).read_bytes()
                )
            with (a/"external_search_development_metrics.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),20)
            with (a/"heldout_world_metrics.csv").open() as f:
                r=list(csv.DictReader(f))
            self.assertEqual(len(r),28)
            self.assertTrue(all(
                abs(float(x["energy_ledger_residual"]))<1e-6
                for x in r
            ))


if __name__=="__main__":
    unittest.main()
