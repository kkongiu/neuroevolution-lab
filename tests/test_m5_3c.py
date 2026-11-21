"""M5.3C genetic ancestry and unseen-world transplant audit tests."""
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from neuroevolution import m5_0 as base
from neuroevolution import m5_3b as prior
from neuroevolution import m5_3c as study


class ProspectiveGenotypeTransplantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = study.recover_one(study.SOURCES[0])
        cls.s = study.Settings(
            seed_start=700, seeds=2, shifted_start=900, shifted_seeds=2,
            ticks=40, size=11, initial_food=9, food_cap=14,
            initial_population=8, max_population=18, checkpoint_every=10,
        )

    def test_sealed_protocol_settings_and_nonoverlapping_origins(self):
        study.Settings().validate()
        self.s.validate()
        self.assertEqual(study.SOURCES,
                         (82006,82015,82018,82019,82023))
        self.assertEqual(len(study.ARMS),5)
        self.assertEqual(study.BOOT_REPS,20000)
        for s in (
            replace(study.Settings(),seed_start=82006),
            replace(study.Settings(),shifted_start=85000,
                    seed_start=85001),
            replace(study.Settings(),ticks=400,
                    obstacle_density=.1),
            replace(study.Settings(),ticks=400,
                    seed_start=84001),
            replace(study.Settings(),ticks=400,
                    initial_gene_sd=.62),
        ):
            with self.assertRaises(ValueError):
                s.validate()

    def test_selected_genes_are_historical_unmodified_m5_3b_genes(self):
        sample = self.source
        old,_,trace = prior.simulate(
            study.SOURCES[0],study.original_settings(),
            "sparse_2",detailed=True
        )
        expected = min(
            trace["final_agents"],key=lambda a:(-a["generation"],a["id"])
        )
        self.assertEqual(sample["selected_final_id"],expected["id"])
        self.assertEqual(
            tuple(sample["genotypes"]["descendant"]["genes"]),
            tuple(expected["gene_values"])
        )
        self.assertEqual(
            sample["selected_final_generation"],expected["generation"]
        )
        self.assertEqual(
            sample["original_source_births"],old["births"]
        )
        self.assertEqual(
            sample["selected_final_founder_id"],expected["founder"]
        )

    def test_parentage_and_all_ten_gene_genomes_audited(self):
        s=self.source
        self.assertEqual(
            len(s["genealogy"]),s["selected_final_generation"]
        )
        self.assertEqual(
            s["genealogy"][-1]["child"],s["selected_final_id"]
        )
        current=s["selected_final_founder_id"]
        for step in s["genealogy"]:
            self.assertEqual(step["parent"],current)
            self.assertEqual(len(step["mutated_indices"]),2)
            self.assertEqual(
                step["gene_digest"],prior.gene_digest(
                    tuple(step["gene_values"])
                )
            )
            self.assertEqual(
                step["expanded_genome_digest"],
                base.genome_digest(prior.to_weights(
                    tuple(step["gene_values"])
                ))
            )
            current=step["child"]
        for arm in study.ARMS:
            stored=s["genotypes"][arm]
            genes=tuple(stored["genes"])
            self.assertEqual(len(genes),10)
            self.assertEqual(
                stored["gene_digest"],prior.gene_digest(genes)
            )
            self.assertEqual(
                stored["native_401_genome_digest"],
                base.genome_digest(prior.to_weights(genes))
            )
        self.assertNotEqual(
            s["genotypes"]["descendant"]["genes"],
            s["genotypes"]["ancestor"]["genes"]
        )
        self.assertNotEqual(
            s["genotypes"]["descendant"]["genes"],
            s["genotypes"]["last_step_reverted"]["genes"]
        )
        self.assertEqual(
            sorted(s["genotypes"]["descendant"]["genes"]),
            sorted(s["genotypes"]["scrambled"]["genes"])
        )

    def test_last_step_reversion_is_exactly_parental(self):
        s=self.source
        a=s["genotypes"]["descendant"]["genes"]
        b=s["genotypes"]["last_step_reverted"]["genes"]
        changed=[
            i for i,(x,y) in enumerate(zip(a,b)) if x!=y
        ]
        self.assertEqual(changed,s["last_step_mutated_indices"])

    def test_unaltered_native_neural_actor_clones_and_ledger(self):
        for arm in study.ARMS:
            for hard in (False,True):
                a=study.simulate_transplant(
                    702,self.s,self.source,arm,shifted=hard,detailed=True
                )
                self.assertEqual(
                    a,study.simulate_transplant(
                        702,self.s,self.source,arm,shifted=hard,detailed=True
                    )
                )
                row,checkpoints,trace=a
                self.assertEqual(row["all_actions_original_numeric_neural"],1)
                self.assertEqual(row["mutations_during_transplant"],0)
                self.assertEqual(row["births_are_fixed_clones"],1)
                self.assertAlmostEqual(row["energy_ledger_residual"],0.)
                self.assertTrue(all(
                    abs(ck["energy_ledger_residual"])<1e-6
                    for ck in checkpoints
                ))
                self.assertTrue(all(
                    e["genome"]==row["genome_digest"]
                    for e in trace["events"] if e["type"]=="birth"
                ))
                walls=set(map(tuple,trace["walls"]))
                self.assertTrue(all(
                    tuple(e["location"]) not in walls
                    for e in trace["events"] if e["type"]=="birth"
                ))

    def test_same_ecology_food_and_map_for_all_arms(self):
        info=[]
        for arm in study.ARMS:
            _,_,trace=study.simulate_transplant(
                700,self.s,self.source,arm,detailed=True
            )
            info.append((trace["walls"],trace["initial_food"]))
        self.assertEqual(len({json.dumps(i) for i in info}),1)

    def test_invalid_source_arm_or_corrupted_genome_rejected(self):
        with self.assertRaises(ValueError):
            study.recover_one(82000)
        with self.assertRaises(ValueError):
            study.simulate_transplant(
                700,self.s,self.source,"not_a_predeclared_arm"
            )
        bad=json.loads(json.dumps(self.source))
        bad["genotypes"]["descendant"]["genes"][0]+=0.1
        with self.assertRaises(AssertionError):
            study.simulate_transplant(
                700,self.s,bad,"descendant"
            )

    def test_crossed_family_world_bootstrap_not_agent_level(self):
        rows=[]
        for family in study.SOURCES:
            for seed in (1,2,3,4):
                for arm in study.ARMS:
                    survived=int(arm=="descendant")
                    rows.append({
                        "source_seed":family,"seed":seed,
                        "arm":arm,"split":"main","survived":survived,
                        "births":0,"food_eaten":0,"alive_final":0
                    })
        cmp=study.paired_crossed_cluster_contrast(
            rows,"main","ancestor",study.BOOT_SEED,200
        )
        self.assertEqual(cmp["mean_delta"],1.)
        self.assertEqual(
            cmp["crossed_cluster_bootstrap_95_descriptive"],[1.,1.]
        )
        self.assertEqual(cmp["source_families_positive"],5)
        self.assertTrue(cmp["supported_with_four_families"])
        rows=[r for r in rows if not (
            r["source_seed"]==study.SOURCES[0] and
            r["seed"]==1 and r["arm"]=="ancestor"
        )]
        with self.assertRaises(AssertionError):
            study.paired_crossed_cluster_contrast(
                rows,"main","ancestor",study.BOOT_SEED,100
            )

    def test_complete_reproducible_smoke_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/"a",Path(folder)/"b"
            out1=study.run(self.s,a,200)
            out2=study.run(self.s,b,200)
            self.assertEqual(out1,out2)
            self.assertEqual(out1["simulation_count"],100)
            self.assertEqual(
                out1["origins"],list(study.SOURCES)
            )
            self.assertEqual(len(out1["candidate_genome_digests"]),5)
            for filename in (
                "frozen_source_candidate_bank.json",
                "summary.json","world_metrics.csv",
                "population_checkpoints.csv",
                "events_main_82006_descendant.json",
                "events_shifted_82023_ancestor.json",
            ):
                self.assertEqual(
                    (a/filename).read_bytes(),(b/filename).read_bytes()
                )
            with (a/"world_metrics.csv").open(newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))),100)
            src=json.loads(
                (a/"frozen_source_candidate_bank.json").read_text()
            )
            self.assertEqual(len(src),5)


if __name__=="__main__":
    unittest.main()
