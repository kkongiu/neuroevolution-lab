"""M5.4B0-R: exact 18-gene numeric bounded recurrence and causality tests."""
import csv
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from neuroevolution import m5_0 as physics
from neuroevolution import m5_3b as original
from neuroevolution import m5_4b0r as rec


class EighteenGeneRecurrentArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small=rec.Settings(
            development_start=400,development_seeds=2,
            validation_start=500,validation_seeds=2,
            seed_start=600,seeds=2,shifted_start=700,shifted_seeds=2,
            external_pool=6,external_generations=1,external_offspring=4,
            external_elites=2,ticks=35,size=11,
            initial_food=9,food_cap=14,initial_population=8,
            max_population=18,checkpoint_every=10,no_food_controls=0
        )
        cls.small.validate()
        cls.genes=(.7,.8,.9,.1,.2,-.1,0.,.4,.2,1.,
                   .2,-.3,.4,-.5,.6,-.7,.8,-.9)

    def test_protocol_origins_and_independent_fresh_worlds(self):
        rec.Settings().validate()
        self.assertEqual(rec.PROTOCOL_COMMIT,
                         "7ffd8e06510a4c08a80c7610d7e0ff2617aec1a7")
        self.assertEqual(rec.N_GENES,18)
        self.assertEqual(rec.BOOT_REPS,20000)
        self.assertEqual(len(rec.ARMS),8)
        for bad in (
            replace(rec.Settings(),seed_start=88200),
            replace(rec.Settings(),development_start=88000),
            replace(rec.Settings(),validation_start=89200),
            replace(rec.Settings(),shifted_start=89200),
            replace(rec.Settings(),odor_radius=5),
            replace(rec.Settings(),external_pool=0),
            replace(rec.Settings(),ticks=400,external_pool=64),
        ):
            with self.assertRaises(ValueError):
                bad.validate()

    def test_exact_10_gene_original_bitwise_identity_when_recurrence_off(self):
        front=self.genes[:10]
        old=original.to_weights(front)
        self.assertEqual(len(old),401)
        self.assertEqual(rec.to_weights(self.genes,recurrence_enabled=False),old)
        self.assertEqual(rec.to_weights(front),old)
        self.assertEqual(rec.to_weights(front,recurrence_enabled=False),old)
        self.assertEqual(
            physics.genome_digest(old),
            physics.genome_digest(
                rec.to_weights(self.genes,recurrence_enabled=False))
        )
        self.assertEqual(
            rec.to_weights(self.genes[:10]+(0.,)*8),old
        )

    def test_only_eight_recurrent_weights_change_and_are_bounded(self):
        baseline=original.to_weights(self.genes[:10])
        weights=rec.to_weights(self.genes)
        stride=physics.INPUTS+12+1
        changed={idx for idx,(a,b) in enumerate(zip(baseline,weights))
                 if a!=b}
        wanted=set()
        for i in range(4):
            diagonal=i*stride+physics.INPUTS+i
            cyclic=i*stride+physics.INPUTS+(i+1)%4
            wanted.update((diagonal,cyclic))
            self.assertAlmostEqual(
                weights[diagonal],.65*math.tanh(self.genes[10+i])
            )
            self.assertAlmostEqual(
                weights[cyclic],.35*math.tanh(self.genes[14+i])
            )
            self.assertLess(abs(weights[diagonal]),.65)
            self.assertLess(abs(weights[cyclic]),.35)
        self.assertEqual(changed,wanted)
        self.assertEqual(len(changed),8)
        self.assertNotEqual(weights,baseline)
        self.assertEqual(len(weights),401)
        extreme=self.genes[:10]+(1e12,-1e12,1e12,-1e12)*2
        self.assertTrue(all(math.isfinite(x) for x in rec.to_weights(extreme)))
        self.assertEqual(
            rec.to_weights(extreme,recurrence_enabled=False),baseline
        )
        for invalid in (
            (0.,)*17,(0.,)*19,self.genes[:10]+(math.nan,)*8,
            self.genes[:10]+(math.inf,)*8
        ):
            with self.assertRaises(ValueError):
                rec.to_weights(invalid)

    def test_recurrence_is_actual_neural_timestep_memory_not_scripted_motor(self):
        original_w=original.to_weights(self.genes[:10])
        rec_w=rec.to_weights(self.genes)
        state=physics.Agent(
            id=0,founder=0,parent=None,generation=0,
            position=(5,5),energy=36.,genome=rec_w,hidden=[0.]*12
        )
        settings=self.small
        empty=(0.,)*physics.INPUTS
        a=physics.network_logits(state,empty,settings)
        memory=list(state.hidden)
        b=physics.network_logits(state,empty,settings)
        self.assertNotEqual(a,b)
        self.assertNotEqual(memory,state.hidden)
        state.genome=original_w
        state.hidden=[0.]*12
        a=physics.network_logits(state,empty,settings)
        b=physics.network_logits(state,empty,settings)
        self.assertEqual(a,b)  # exactly old zero-recurrence architecture

    def test_identical_physical_world_and_exact_unmutated_real_births(self):
        for shifted in (False,True):
            row,ch,trace=rec.simulate_numeric(
                600,self.small,self.genes,
                recurrence_enabled=True,shifted=shifted,detailed=True
            )
            row2,ch2,trace2=rec.simulate_numeric(
                600,self.small,self.genes,
                recurrence_enabled=True,shifted=shifted,detailed=True
            )
            self.assertEqual((row,ch,trace),(row2,ch2,trace2))
            self.assertEqual(row["native_neural_actions"],1)
            self.assertEqual(row["scripted_behavior"],0)
            self.assertEqual(row["in_world_birth_mutations"],0)
            self.assertEqual(row["recurrent_genes_enabled"],1)
            self.assertTrue(all(abs(k["energy_ledger_residual"])<1e-6
                                for k in ch))
            self.assertEqual(
                row["genotype_digest"],
                physics.genome_digest(rec.to_weights(self.genes))
            )
            self.assertTrue(all(
                e["genome"]==row["genotype_digest"]
                for e in trace["events"] if e["type"]=="birth"
            ))
            orig=physics.create_ecology(
                600,self.small,"heritable_no_mutation",shifted=shifted
            )
            self.assertEqual(trace["initial_positions"],
                             [list(a.position) for a in orig.agents.values()])
            self.assertEqual(trace["initial_food"],
                             [list(p) for p in orig.world.initial_food])

    def test_zero_recurrence_exactly_same_observed_world_outcomes(self):
        settings=self.small
        full=tuple(self.genes)
        a=rec.simulate_numeric(
            600,settings,full,recurrence_enabled=False,detailed=True
        )
        b=rec.simulate_numeric(600,settings,full[:10],detailed=True)
        self.assertEqual(a[0]["survived"],b[0]["survived"])
        self.assertEqual(a[0]["genotype_digest"],b[0]["genotype_digest"])
        self.assertEqual(a[0]["births"],b[0]["births"])
        self.assertEqual(a[0]["food_eaten"],b[0]["food_eaten"])
        self.assertEqual(a[0]["alive_final"],b[0]["alive_final"])
        self.assertEqual(a[1],b[1])
        self.assertEqual(a[2]["events"],b[2]["events"])
        self.assertEqual(a[0]["recurrent_genes_enabled"],0)

    def test_researcher_script_is_NOT_neural_and_can_have_inert_genomes(self):
        row,cks,_=rec.scripted_control(600,self.small)
        self.assertEqual(row["native_neural_actions"],0)
        self.assertEqual(row["scripted_behavior"],1)
        self.assertIsNone(row["in_world_birth_mutations"])
        self.assertIsNone(row["genotype_digest"])
        self.assertAlmostEqual(row["energy_ledger_residual"],0.)
        self.assertTrue(all(abs(x["energy_ledger_residual"])<1e-6
                            for x in cks))

    def test_two_external_optimizers_equal_budget_no_leakage(self):
        tracked=[]
        native=rec.simulate_numeric
        def spy(seed,settings,genes,**kwargs):
            tracked.append((seed,len(genes),kwargs.get("shifted",False)))
            return native(seed,settings,genes,**kwargs)
        with patch.object(rec,"simulate_numeric",side_effect=spy):
            recurrent=rec.external_search(self.small,"recurrent")
            feedforward=rec.external_search(self.small,"feedforward")
        self.assertEqual(len(recurrent[0]),10)
        self.assertEqual(len(feedforward[0]),10)
        self.assertEqual(len(recurrent[1]),20)
        self.assertEqual(len(feedforward[1]),20)
        self.assertEqual(len(recurrent[2]),4)
        self.assertEqual(len(feedforward[2]),4)
        self.assertEqual(set(n for _,n,_ in tracked),{10,18})
        self.assertTrue(all(seed in (400,401,500,501) and not hard
                            for seed,_,hard in tracked))
        self.assertTrue(all(len(r["genes"])==18 for r in recurrent[0]))
        self.assertTrue(all(len(r["genes"])==10 for r in feedforward[0]))
        self.assertEqual(recurrent[4]["chosen_candidate_id"],
                         recurrent[3]["candidate_id"])
        self.assertNotIn(600,[seed for seed,_,_ in tracked])

    def test_external_replay_and_bounded_controls_are_reproducible(self):
        a=rec.external_search(self.small,"recurrent")
        b=rec.external_search(self.small,"recurrent")
        self.assertEqual(a,b)
        ff=rec.external_search(self.small,"feedforward")
        arms=rec.frozen_controls(a[3],ff[3],self.small)
        self.assertEqual(tuple(arms),rec.ARMS)
        self.assertEqual(arms["recurrent_witness"][0],
                         arms["recurrence_clamped"][0])
        self.assertEqual(rec.to_weights(
            arms["recurrence_clamped"][0],recurrence_enabled=False
        ),original.to_weights(a[3]["genes"][:10]))
        self.assertEqual(
            sorted(arms["recurrent_scrambled"][0]),
            sorted(arms["recurrent_witness"][0])
        )
        self.assertEqual(len(arms["random_recurrent_0"][0]),18)
        self.assertEqual(len(arms["random_feedforward_0"][0]),10)
        self.assertIsNone(arms["scripted_greedy_novelty"])

    def test_paired_stats_and_predeclared_positive_gate(self):
        settings=self.small
        rows=[]
        for split,seeds in (
            ("main",(600,601)),("shifted",(700,701))
        ):
            for seed in seeds:
                for arm in rec.ARMS:
                    is_rec=arm=="recurrent_witness"
                    row={
                        "split":split,"seed":seed,"arm":arm,
                        "survived":int(is_rec or
                            arm=="scripted_greedy_novelty"),
                        "births":6 if is_rec else 2,
                        "food_eaten":10 if is_rec else 3,
                        "max_generation":3,
                    }
                    rows.append(row)
        a=rec.analyze(rows,settings,bootstrap_reps=110)
        self.assertEqual(
            a["main"]["contrasts"]["recurrent_minus_recurrence_clamped"]
            ["world_paired_bootstrap95"],[1.,1.]
        )
        self.assertEqual(
            a["main"]["contrasts"][
                "recurrent_minus_feedforward_witness"
            ]["mean_paired_survival_difference"],1.
        )
        # The smoke contains only TWO worlds; the registered 12-world
        # viability gate must correctly remain negative.
        self.assertFalse(a["gates"]["main_rec_survive_ge12"])
        self.assertFalse(a["gates"]["main_representability_and_recurrence_gain"])
        rows.pop()
        with self.assertRaises(AssertionError):
            rec.analyze(rows,settings,bootstrap_reps=100)

    def test_full_deterministic_smoke_archives_failed_candidates_and_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            a=Path(temp)/"a"
            b=Path(temp)/"b"
            one=rec.run(self.small,a,bootstrap_reps=110)
            two=rec.run(self.small,b,bootstrap_reps=110)
            self.assertEqual(one,two)
            self.assertEqual(one["external_genome_candidate_count"],20)
            self.assertEqual(one["external_development_trial_count"],40)
            self.assertEqual(one["external_validation_trial_count"],8)
            self.assertEqual(one["sealed_trial_count"],32)
            self.assertEqual(one["total_physical_world_trials"],80)
            for filename in (
                "frozen_full_precision_gene_bank.json",
                "external_768_development_worlds.csv",
                "external_96_validation_worlds.csv",
                "external_192_all_candidate_genes.json",
                "sealed_288_world_outcomes.csv",
                "sealed_population_checkpoints.csv","summary.json",
                "events_main_600_recurrent_witness.json",
                "events_shifted_700_scripted_greedy_novelty.json",
            ):
                self.assertEqual((a/filename).read_bytes(),
                                 (b/filename).read_bytes())
            with (a/"sealed_288_world_outcomes.csv").open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),32)
            self.assertTrue(all(
                abs(float(x["energy_ledger_residual"]))<1e-6
                for x in rows
            ))


if __name__=="__main__":
    unittest.main()
