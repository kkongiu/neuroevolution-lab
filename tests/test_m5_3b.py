"""M5.3B de novo genotype-native neural evolutionary ecology controls."""
import csv
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from neuroevolution import m5_0 as base
from neuroevolution import m5_3b as lab


class SharedGeneDeNovoTests(unittest.TestCase):
    def setUp(self):
        self.s = lab.Settings(
            seed_start=700, seeds=2, shifted_start=900, shifted_seeds=2,
            ticks=40, size=11, initial_food=9, food_cap=14,
            initial_population=8, max_population=18,
            checkpoint_every=10,
        )

    def test_frozen_geometry_and_world_seed_disjointness(self):
        self.s.validate()
        self.assertEqual(lab.N_GENES, 10)
        self.assertEqual(base.genome_size(12), 401)
        self.assertEqual(set(lab.ARMS), {
            "sparse_2","dense_10","clone","random_child"
        })
        self.assertTrue(set(range(82000,82024)).isdisjoint(
            range(83000,83012)
        ))
        with self.assertRaises(ValueError):
            replace(self.s, hidden=2).validate()
        with self.assertRaises(ValueError):
            replace(self.s, sparse_gene_count=0).validate()
        with self.assertRaises(ValueError):
            replace(self.s, ticks=400, sparse_gene_count=3).validate()

    def test_gene_expansion_is_numerical_and_has_zero_recurrence(self):
        genes=tuple(i*.1 for i in range(10))
        full=lab.to_weights(genes)
        self.assertEqual(len(full),401)
        stride=base.INPUTS+12+1
        for i in range(12):
            self.assertTrue(all(
                full[i*stride+base.INPUTS+k] == 0.
                for k in range(12)
            ))
        self.assertEqual(
            full[4*stride + base.INPUTS +12],
            -6.24*genes[7]+genes[8]
        )
        with self.assertRaises(ValueError):
            lab.to_weights((1.,2.))
        with self.assertRaises(ValueError):
            lab.to_weights(tuple(float("nan") if i==0 else 0.
                                  for i in range(10)))

    def test_action_logits_respond_to_genotype_not_runtime_script(self):
        genes=[0.] * lab.N_GENES
        genes[0]=1.0  # positive food-channel gene
        sensors=[0.] * base.INPUTS
        sensors[8+2]=1.
        sensors[12]=.5
        actor=base.Agent(
            id=0, founder=0, parent=None, generation=0, position=(2,2),
            energy=20, genome=lab.to_weights(tuple(genes)), hidden=[0.]*12
        )
        logits=base.network_logits(actor,tuple(sensors),self.s)
        self.assertEqual(max(range(4), key=lambda i: logits[i]),2)
        genes[0]=-1.0  # inhibitory/aversive gene
        actor.genome=lab.to_weights(tuple(genes))
        actor.hidden=[0.]*12
        aversive=base.network_logits(actor,tuple(sensors),self.s)
        self.assertNotEqual(
            max(range(4),key=lambda i: aversive[i]),2
        )
        genes=[0.]*lab.N_GENES
        genes[7]=1.5
        genes[9]=2.
        actor.genome=lab.to_weights(tuple(genes))
        actor.hidden=[0.]*12
        sensors[12]=1.0
        hungry=base.network_logits(actor,tuple(sensors),self.s)
        self.assertEqual(max(range(5),key=lambda i: hungry[i]),4)
        actor.hidden=[0.]*12
        sensors[12]=.1
        low=base.network_logits(actor,tuple(sensors),self.s)
        self.assertNotEqual(max(range(5),key=lambda i: low[i]),4)

    def test_all_arms_start_same_random_genes_and_food_candidates(self):
        f=lab.founder_genes(701,self.s)
        self.assertEqual(f,lab.founder_genes(701,self.s))
        self.assertNotEqual(f,lab.founder_genes(702,self.s))
        self.assertEqual(len(f),self.s.initial_population)
        self.assertTrue(any(x<0 for g in f for x in g))
        self.assertTrue(any(x>0 for g in f for x in g))
        ecologies=[
            base.create_ecology(701,self.s,"heritable_no_mutation")
            for arm in lab.ARMS
        ]
        self.assertTrue(all(
            eco.world==ecologies[0].world and
            base._food_schedule(701,self.s,eco,False)==
            base._food_schedule(701,self.s,ecologies[0],False)
            for eco in ecologies
        ))

    def test_mutations_only_at_real_birth_and_exact_scope(self):
        genes=tuple(i*.1 for i in range(10))
        for arm,expected in (
            ("sparse_2",2),("dense_10",10),
            ("clone",0),("random_child",10)
        ):
            child,which=lab.child_genes(
                genes,self.s,701,7,3,arm
            )
            again=lab.child_genes(genes,self.s,701,7,3,arm)
            self.assertEqual((child,which),again)
            self.assertEqual(len(which),expected)
            self.assertEqual(len(set(which)),expected)
            self.assertEqual(len(child),10)
            if arm=="clone":
                self.assertEqual(child,genes)
            else:
                self.assertNotEqual(child,genes)
        a,idx=lab.child_genes(genes,self.s,702,5,4,"sparse_2")
        self.assertEqual(
            [i for i,(x,y) in enumerate(zip(genes,a)) if x!=y],
            list(idx)
        )
        with self.assertRaises(ValueError):
            lab.child_genes((1.,),self.s,700,1,0,"sparse_2")

    def test_original_physical_act_can_birth_genotype_encoded_neural_agent(self):
        eco=base.create_ecology(700,self.s,"heritable_no_mutation")
        agent=next(a for a in eco.agents.values()
                   if base.blocked_and_free(eco,a)[1])
        gene=[0.]*10
        gene[7]=1.5
        gene[9]=2.
        agent.genome=lab.to_weights(tuple(gene))
        agent.hidden=[0.]*12
        agent.age=5
        agent.energy=50.
        before=agent.energy
        next_id=eco.next_id
        base.act(eco,agent.id,5)
        self.assertEqual(eco.next_id,next_id+1)
        self.assertEqual(eco.births,1)
        child=eco.agents[next_id]
        self.assertEqual(child.parent,agent.id)
        self.assertEqual(child.generation,1)
        self.assertEqual(child.genome,agent.genome)
        self.assertEqual(child.energy,self.s.newborn_energy)
        self.assertEqual(
            agent.energy,
            before-self.s.movement_cost-self.s.reproduction_energy_cost
        )

    def test_seed_determinism_genome_trace_and_energy_ledger(self):
        for arm in lab.ARMS:
            for shifted in (False,True):
                a=lab.simulate(
                    702,self.s,arm,shifted=shifted,detailed=True
                )
                b=lab.simulate(
                    702,self.s,arm,shifted=shifted,detailed=True
                )
                self.assertEqual(a,b)
                result,checks,trace=a
                self.assertEqual(result["all_actions_native_neural"],1)
                self.assertEqual(result["externally_selected_founders"],0)
                self.assertEqual(result["genetic_births"],result["births"])
                self.assertEqual(result["energy_ledger_residual"],0.)
                self.assertTrue(all(
                    abs(row["energy_ledger_residual"])<1e-6
                    for row in checks
                ))
                events=[e for e in trace["events"] if e["type"]=="birth"]
                self.assertEqual(len(events),result["births"])
                expected={"sparse_2":2,"dense_10":10,
                          "clone":0,"random_child":10}[arm]
                self.assertTrue(all(
                    len(e["changed_gene_indices"])==expected
                    and e["genome"]==
                        base.genome_digest(lab.to_weights(
                            tuple(e["gene_values"])
                        ))
                    for e in events
                ))
                self.assertTrue(all(
                    tuple(e["location"]) not in set(map(tuple,trace["walls"]))
                    for e in events
                ))

    def test_paired_world_contrast_unit_is_world(self):
        rows=[]
        for seed,scores in (
            (1,(1,0,0,0)),(2,(1,0,1,0)),(3,(0,0,0,0))
        ):
            rows.extend({
                "seed":seed,"split":"main","arm":arm,"survived":score
            } for arm,score in zip(lab.ARMS,scores))
        stats=lab.contrasts(rows,"main","survived",5)
        self.assertAlmostEqual(
            stats["sparse_2_minus_random_child"]["mean_delta"],2/3
        )
        self.assertAlmostEqual(
            stats["sparse_2_minus_clone"]["mean_delta"],1/3
        )

    def test_two_runs_write_identical_archives(self):
        with tempfile.TemporaryDirectory() as path:
            a,b=Path(path)/"a",Path(path)/"b"
            sa,sb=lab.run(self.s,a),lab.run(self.s,b)
            self.assertEqual(sa,sb)
            self.assertIn("novel_mutation_vs_clone",
                          sa["preregistered_gates"])
            for name in (
                "summary.json","world_metrics.csv",
                "population_checkpoints.csv","population_curves.svg",
                "events_main_sparse_2.json",
                "events_shifted_random_child.json"
            ):
                self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())
            with (a/"world_metrics.csv").open(newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4*(self.s.seeds+self.s.shifted_seeds))


if __name__=="__main__":
    unittest.main()
