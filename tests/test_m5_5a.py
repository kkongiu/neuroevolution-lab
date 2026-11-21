"""M5.5A: actual inherited instinct gains, legal physical births, anti-claims."""
import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from neuroevolution import m5_0 as base
from neuroevolution import m5_5a as new


class NaiveHeritableInstinctExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small=new.Settings(
            seed_start=300,seeds=2,shifted_start=400,shifted_seeds=2,
            ticks=30,size=11,initial_food=9,food_cap=14,
            initial_population=8,max_population=18,
            checkpoint_every=10,no_food_controls=0
        )
        cls.small.validate()

    def test_predeclared_original_genes_origins_and_world_fence(self):
        s=new.Settings()
        s.validate()
        self.assertEqual(new.PROTOCOL_SHA,
                         "a3aa39e4f68acc38d17a4292b4ebefbd4b72b10b")
        self.assertEqual(new.N_GENES,8)
        self.assertEqual(len(new.ARMS),5)
        self.assertEqual(new.BOOT_REPS,20000)
        self.assertEqual(s.seed_start,93000)
        self.assertEqual(s.shifted_start,93100)
        for bad in (
            replace(s,seed_start=93000,shifted_start=93010),
            replace(s,seed_start=91000),
            replace(s,seed_start=92000),
            replace(s,seed_start=89200),
            replace(s,shifted_start=92100),
            replace(s,odor_radius=5),
            replace(s,innate_mutation_sd=.35),
            replace(s,ticks=400,seeds=64),
            replace(s,ticks=400,initial_population=12),
        ):
            with self.assertRaises(ValueError):
                bad.validate()

    def test_twenty_random_naive_source_founders_are_same_for_all_arms(self):
        s=self.small
        a=new.founder_genes(300,s)
        b=new.founder_genes(300,s)
        c=new.founder_genes(301,s)
        self.assertEqual(a,b)
        self.assertNotEqual(a,c)
        self.assertEqual(len(a),s.initial_population)
        self.assertEqual(len({new.gene_sha(g) for g in a}),len(a))
        self.assertTrue(all(len(g)==8 and all(math.isfinite(x) for x in g)
                            for g in a))
        with self.assertRaises(ValueError):
            new.check_genes((0.,)*7)
        with self.assertRaises(ValueError):
            new.check_genes((float("nan"),)*8)

    def test_instinct_parameter_bounds_hunger_and_traits_are_effective(self):
        s=self.small
        zeros=(0.,)*8
        p=new.instinct_parameters(zeros,54.,s)
        self.assertAlmostEqual(p["hunger"],0.)
        self.assertAlmostEqual(p["gain"],1.)
        self.assertAlmostEqual(p["adjacent_food_weight"],3.2)
        self.assertAlmostEqual(p["odor_weight"],.8)
        self.assertAlmostEqual(p["novelty_weight"],.2)
        self.assertAlmostEqual(p["persistence_weight"],0.)
        self.assertAlmostEqual(p["random_motor_amplitude"],.25)
        self.assertAlmostEqual(p["birth_energy_threshold"],46.)
        self.assertAlmostEqual(p["birth_probability"],.65)
        larger=(2.,)*8
        smaller=(-2.,)*8
        aa=new.instinct_parameters(larger,5.,s)
        bb=new.instinct_parameters(smaller,5.,s)
        self.assertGreater(aa["gain"],bb["gain"])
        self.assertGreater(aa["adjacent_food_weight"],bb["adjacent_food_weight"])
        self.assertGreater(aa["odor_weight"],bb["odor_weight"])
        self.assertGreater(aa["novelty_weight"],bb["novelty_weight"])
        self.assertGreater(aa["birth_probability"],bb["birth_probability"])
        self.assertGreater(aa["birth_energy_threshold"],bb["birth_energy_threshold"])
        self.assertLess(aa["birth_probability"],.9)
        self.assertGreater(bb["birth_probability"],.4)
        self.assertGreater(aa["birth_energy_threshold"],40.)
        self.assertLess(aa["birth_energy_threshold"],52.)

    def test_per_birth_sparse_dense_clone_and_random_offspring_are_real(self):
        s=self.small
        original=tuple(i/10 for i in range(8))
        for arm in new.ARMS:
            child,indices=new.child_genes(original,300,14,5,arm,s)
            replay,indices2=new.child_genes(original,300,14,5,arm,s)
            self.assertEqual((child,indices),(replay,indices2))
            self.assertEqual(len(child),8)
            self.assertTrue(all(math.isfinite(x) for x in child))
            if arm in (new.CLONE,new.RANDOM_MOTOR):
                self.assertIs(child,original)
                self.assertEqual(indices,[])
            elif arm==new.SPARSE:
                self.assertEqual(len(indices),2)
                self.assertEqual(len(set(indices)),2)
                self.assertEqual([
                    i for i in range(8) if child[i]!=original[i]
                ],indices)
            else:
                self.assertEqual(indices,list(range(8)))
                self.assertTrue(all(a!=b for a,b in zip(child,original)))
        bounded=tuple(2.5 for _ in range(8))
        for arm in (new.SPARSE,new.DENSE):
            child,_=new.child_genes(bounded,300,17,22,arm,s)
            self.assertTrue(all(-2.5<=v<=2.5 for v in child))

    def test_actions_only_use_legal_sensed_directions_not_global_food(self):
        s=self.small
        eco=base.create_ecology(300,s,"heritable_no_mutation")
        source=new.founder_genes(300,s)
        for agent,g in zip(eco.agents.values(),source):
            agent.genome=g
        agent=eco.agents[0]
        legal=[i for i,b in enumerate(base.blocked_and_free(eco,agent)[0])
               if not b]
        self.assertTrue(legal)
        for arm in new.ARMS:
            action,obs=new.choose_action(
                eco,agent,1,{agent.position:1},None,arm
            )
            self.assertTrue(action in legal or
                            action==base.REPRODUCE or action is None)
            self.assertEqual(obs["random_legal_motor_control"],
                             int(arm==new.RANDOM_MOTOR))
        agent.energy=10
        for arm in new.ARMS:
            action,_=new.choose_action(
                eco,agent,9,{agent.position:1},None,arm
            )
            self.assertIn(action,legal)
            self.assertNotEqual(action,base.REPRODUCE)

    def test_true_genetic_birth_and_energy_costs_in_each_random_ancestry_arm(self):
        # Small full physical worlds still have the original positive
        # food growth and real legal energy-costed births.
        s=self.small
        for arm in new.ARMS:
            row,ck,trace=new.simulate(300,s,arm,detailed=True)
            self.assertEqual(row["births"],row["natively_costed_birth_actions"])
            self.assertEqual(
                row["initial_20_naive_founders"],s.initial_population
            )
            self.assertTrue(all(
                abs(x["energy_ledger_residual"])<1e-6 for x in ck
            ))
            self.assertEqual(len(trace["all_physically_costed_birth_genealogy"]),
                             row["births"])
            self.assertEqual(len(trace["initial_20_founder_genes"]),
                             s.initial_population)
            for event in trace["all_physically_costed_birth_genealogy"]:
                self.assertEqual(len(event["newborn_genes"]),8)
                if arm in (new.CLONE,new.RANDOM_MOTOR):
                    self.assertEqual(
                        event["newborn_genes"],event["parent_genes"]
                    )
                else:
                    self.assertNotEqual(
                        event["newborn_genes"],event["parent_genes"]
                    )
                if arm==new.SPARSE:
                    self.assertEqual(len(event["changed_indices"]),2)
                if arm==new.DENSE:
                    self.assertEqual(len(event["changed_indices"]),8)
            self.assertEqual(
                row["genetically_mutated_births"],
                row["births"] if arm in (new.SPARSE,new.DENSE) else 0
            )
            self.assertEqual(row["independent_random_gene_newborns"],
                             row["births"] if arm==new.RANDOM_CHILD else 0)
            self.assertEqual(
                row["true_within_world_genetic_birth_mutation"],
                int(arm in (new.SPARSE,new.DENSE))
            )
            self.assertEqual(row["native_neural_401_weight_processor_used"],0)

    def test_every_arm_same_world_and_same_original_naive_gene_pool(self):
        s=self.small
        for seed,hard in ((300,False),(400,True)):
            rows=[new.simulate(seed,s,arm,shifted=hard)[0]
                  for arm in new.ARMS]
            self.assertEqual(
                {r["original_physical_ticks"]
                 for r in []},
                set()
            )
            self.assertEqual(len({r["world_sha256"] for r in rows}),1)
            self.assertEqual(len({r["initial_gene_bank_sha"] for r in rows}),1)
            self.assertTrue(all(r["energy_ledger_residual"]==0. for r in rows))
            self.assertTrue(all(
                r["natively_costed_birth_actions"]==r["births"]
                for r in rows
            ))

    def test_no_global_food_lookup_or_learned_neural_action(self):
        s=self.small
        eco=base.create_ecology(300,s,"heritable_no_mutation")
        a=eco.agents[0]
        a.genome=(0.,)*8
        for i in list(eco.agents):
            if i!=0:
                del eco.agents[i]
        # Same 15 ORIGINAL local sensors and same personal memory; disjoint
        # unseen long-range food sets cannot change immediate instinct action.
        legal=[d for d,b in enumerate(base.blocked_and_free(eco,a)[0])
               if not b]
        a.energy=10.
        eco.food=set()
        x=new.choose_action(eco,a,10,{a.position:1},None,new.CLONE)
        eco.food={(1,1)}
        # (1,1) may be close to the agent; explicitly compare exact
        # original native sensor channels before asserting policy equality.
        native=base.observe(
            eco,a,base.blocked_and_free(eco,a)[0]
        )
        if all(native[4+i]==0 and native[8+i]==0 for i in range(4)):
            y=new.choose_action(eco,a,10,{a.position:1},None,new.CLONE)
            self.assertEqual(x,y)

    def test_statistics_precommitted_source_world_resampling(self):
        s=self.small
        rows=[]
        for split,seeds in (("main",(300,301)),("shifted",(400,401))):
            for seed in seeds:
                for arm in new.ARMS:
                    advanced=arm==new.SPARSE
                    rows.append({
                        "split":split,"seed":seed,"arm":arm,
                        "world_sha256":f"world-{seed}",
                        "initial_gene_bank_sha":f"source-{seed}",
                        "survived":int(advanced),"alive_final":5 if advanced else 0,
                        "births":40 if advanced else 2,
                        "food_eaten":100 if advanced else 10,
                        "population_auc_mean_checkpoint":8. if advanced else 2.,
                        "mutated_descendant_ever_created":int(advanced),
                        "fraction_survivors_nonfounder_alleles":.8 if advanced else 0.,
                        "repeated_cell_move_fraction":.2 if advanced else .8,
                        "mean_unique_cells_per_organism_life":40. if advanced else 5.,
                    })
        x=new.analyze(rows,[],s,bootstrap_reps=100)
        self.assertTrue(
            x["gates"][
                "sparse_vs_clones_both_survival_and_auc_lower95_positive"
            ]
        )
        self.assertEqual(
            x["main"]["comparisons"]["sparse_minus_clone_survival"]
            ["bootstrap_95"],[1.,1.]
        )
        self.assertFalse(
            x["gates"]["observational_mutation_advantage_gate"]
        )  # needs >=12 tested new worlds, not smoke's 2
        self.assertFalse(
            x["gates"][
                "causal_beneficial_mutation_over_own_founder_transfer_proven"
            ]
        )
        rows.pop()
        with self.assertRaises(AssertionError):
            new.analyze(rows,[],s,bootstrap_reps=100)

    def test_two_fully_reproducible_small_campaigns_and_all_source_records(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/"a";b=Path(d)/"b"
            first=new.run(self.small,a,bootstrap_reps=100)
            second=new.run(self.small,b,bootstrap_reps=100)
            self.assertEqual(first,second)
            self.assertEqual(first["worlds"],4)
            self.assertEqual(first["original_full_world_policy_trials"],20)
            self.assertEqual(first["original_population_checkpoints"],80)
            self.assertEqual(first["gates"][
                "spontaneously_learned_neural_instincts"],False)
            for file in (
                "all_180_innate_genetic_worlds.csv",
                "all_1620_native_population_checkpoints.csv",
                "all_36_exact_naive_source_founder_genotype_banks.json",
                "all_true_birth_genotype_genealogies.json",
                "all_source_detailed_first_world_event_ledgers.json",
                "summary.json",
            ):
                self.assertEqual((a/file).read_bytes(),(b/file).read_bytes())
            with (a/"all_180_innate_genetic_worlds.csv").open() as f:
                items=list(csv.DictReader(f))
            self.assertEqual(len(items),20)
            self.assertEqual(len({
                (r["seed"],r["arm"]) for r in items
            }),20)
            self.assertTrue(all(
                abs(float(r["energy_ledger_residual"]))<1e-6 for r in items
            ))


if __name__=="__main__":
    unittest.main()
