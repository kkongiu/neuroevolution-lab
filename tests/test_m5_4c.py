"""M5.4C: preregistered matched 2×2 causal motor vs birth diagnostic."""
import csv
import json
import math
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

from neuroevolution import m5_0 as base
from neuroevolution import m5_1 as scripted
from neuroevolution import m5_4b0r as previous
from neuroevolution import m5_4c as m5


class FourArmForagingAndBirthCausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small=m5.Settings(
            seed_start=300,seeds=2,shifted_start=400,shifted_seeds=2,
            ticks=35,size=11,initial_food=9,food_cap=14,
            initial_population=8,max_population=18,
            checkpoint_every=10,no_food_controls=0
        )
        cls.small.validate()

    def test_preregistered_original_source_unchanged_and_fresh_worlds(self):
        full=m5.Settings()
        full.validate()
        self.assertEqual(m5.PROTOCOL_SHA,
                         "c351085a15ebb9a56af5f3d198c2f3939905ab80")
        self.assertEqual(len(m5.ARMS),4)
        self.assertEqual(m5.BOOT_REPS,20000)
        self.assertEqual(full.seed_start,91000)
        self.assertEqual(full.shifted_start,91100)
        self.assertEqual(full.scripted_reproduction_threshold,46.)
        genes=m5.source_genes()
        self.assertEqual(len(genes),18)
        self.assertEqual(
            base.genome_digest(previous.to_weights(genes)),
            m5.PINNED_401_DIGEST
        )
        self.assertEqual(genes,m5.source_genes())
        for s in (
            replace(full,seed_start=89200),
            replace(full,seed_start=91100),
            replace(full,shifted_start=91000),
            replace(full,shifted_start=89300),
            replace(full,odor_radius=5),
            replace(full,scripted_reproduction_threshold=33.),
            replace(full,ticks=400,seeds=40),
            replace(full,ticks=400,scripted_reproduction_threshold=47.),
        ):
            with self.assertRaises(ValueError):
                s.validate()

    def test_neural_neural_arm_matches_old_18gene_physical_baseline_bitwise(self):
        for shifted in (False,True):
            for seed in (300,301):
                study,check,trace=m5.simulate(
                    seed,self.small,m5.NN,shifted=shifted,detailed=True
                )
                original,old_check,old_trace=previous.simulate_numeric(
                    seed,self.small,m5.source_genes(),
                    shifted=shifted,detailed=True
                )
                for key in (
                    "survived","alive_final","births","deaths","food_eaten",
                    "food_spawned","food_remaining","max_generation","extinct_at"
                ):
                    self.assertEqual(study[key],original[key],key)
                self.assertEqual(
                    study["final_energy_ledger_residual"],
                    original["energy_ledger_residual"]
                )
                self.assertEqual(
                    study["original_401_genome_digest"],
                    original["genotype_digest"]
                )
                self.assertEqual(
                    trace["physical_birth_and_food_events"],
                    old_trace["events"]
                )
                for a,b in zip(check,old_check):
                    for k in (
                        "tick","population","food_stock","births","deaths",
                        "food_eaten","food_spawned","max_generation",
                        "lineages_alive","total_living_energy",
                        "energy_ledger_residual"
                    ):
                        self.assertEqual(a[k],b[k])

    def test_human_script_script_matches_original_m51_physical_worlds(self):
        s=self.small
        script_settings=scripted.Settings(
            **{
                f.name:getattr(s,f.name)
                for f in fields(scripted.Settings) if hasattr(s,f.name)
            },
            negative_control_worlds=2
        )
        script_settings.validate()
        for shifted in (False,True):
            for seed in (300,301):
                diagnostic,checks,trace=m5.simulate(
                    seed,s,m5.SS,shifted=shifted,detailed=True
                )
                reference,oldchecks,oldtrace=scripted.simulate(
                    seed,script_settings,"greedy_novelty",
                    shifted=shifted,detailed=True
                )
                for key in (
                    "survived","alive_final","births","deaths","food_eaten",
                    "food_spawned","food_remaining","max_generation","extinct_at"
                ):
                    self.assertEqual(diagnostic[key],reference[key],key)
                self.assertEqual(diagnostic["scripted_movement"],1)
                self.assertEqual(diagnostic["scripted_reproduction"],1)
                self.assertEqual(diagnostic["within_world_genetic_mutations"],0)
                # Legacy M5.1 stores MUTATING inert bookkeeping
                # genomes, whereas M5.4C uses exact CLONES of the frozen
                # recurrent neural vector in all four arms. Physical birth
                # and food events must agree; genotype hashes must not.
                def only_physics(events):
                    return [
                        {k:v for k,v in e.items() if k!="genome"}
                        for e in events
                    ]
                self.assertEqual(
                    only_physics(trace["physical_birth_and_food_events"]),
                    only_physics(oldtrace["events"]),
                )
                self.assertEqual(
                    [k["population"] for k in checks],
                    [k["population"] for k in oldchecks],
                )

    def test_scripted_movement_matches_original_M51_when_no_birth(self):
        eco=base.create_ecology(
            300,self.small,"heritable_no_mutation"
        )
        for agent in list(eco.agents.values())[:5]:
            agent.energy=30.
            for tick in (1,2,25):
                blocked,free=base.blocked_and_free(eco,agent)
                sensors=base.observe(eco,agent,blocked)
                legal=[i for i,b in enumerate(blocked) if not b]
                visits={agent.position:2}
                if legal:
                    dx,dy=base.DIRECTIONS[legal[0]]
                    visits[(agent.position[0]+dx,agent.position[1]+dy)]=10
                choice=m5.scripted_movement(
                    eco,agent,tick,visits,sensors,legal
                )
                reference=scripted.scripted_choice(
                    eco,agent,"greedy_novelty",tick,visits
                )
                self.assertEqual(choice,reference)

    def test_all_arms_same_physical_map_and_clonal_numeric_genome(self):
        s=self.small
        for shifted in (False,True):
            rows=[]
            for arm in m5.ARMS:
                row,ck,trace=m5.simulate(
                    300,s,arm,shifted=shifted,detailed=True
                )
                rows.append(row)
                self.assertTrue(all(
                    abs(c["energy_ledger_residual"])<1e-6 for c in ck
                ))
                self.assertEqual(row["final_energy_ledger_residual"],0.)
                self.assertEqual(
                    row["movement_actions"],
                    row["scripted_movement_actions"]+
                    row["neural_movement_actions"]
                )
                self.assertEqual(
                    row["births"],
                    row["scripted_birth_actions"]+
                    row["neural_birth_actions"]
                )
                self.assertEqual(
                    row["births"],row["birth_actions"]
                )
                self.assertTrue(all(
                    event["genome"]==m5.PINNED_401_DIGEST
                    for event in trace["physical_birth_and_food_events"]
                    if event["type"]=="birth"
                ))
                self.assertEqual(
                    row["total_organisms_ever_born"],
                    s.initial_population+row["births"]
                )
                self.assertGreaterEqual(
                    row["distinct_cells_visited_by_any_agent"],s.initial_population
                )
                self.assertGreaterEqual(
                    row["mean_unique_cells_visited_per_agent_life"],1.
                )
                self.assertLessEqual(row["revisit_fraction_of_moves"],1.)
            self.assertEqual(len({r["world_sha256"] for r in rows}),1)
            self.assertEqual(len({
                r["original_401_genome_digest"] for r in rows
            }),1)

    def test_explicit_human_birth_and_neural_birth_are_separable(self):
        eco=base.create_ecology(
            300,self.small,"heritable_no_mutation"
        )
        agent=eco.agents[0]
        agent.genome=previous.to_weights(m5.source_genes())
        agent.energy=50.
        agent.age=5
        self.assertTrue(base.can_reproduce(
            eco,agent,base.blocked_and_free(eco,agent)[1]
        ))
        from dataclasses import asdict
        def perform(arm,logits):
            clone=base.create_ecology(
                300,self.small,"heritable_no_mutation"
            )
            for a in clone.agents.values():
                a.genome=agent.genome
            target=clone.agents[0]
            target.energy=50.
            target.age=5
            visits={a.id:{a.position:1} for a in clone.agents.values()}
            globalcells={a.position for a in clone.agents.values()}
            st={a.id:0 for a in clone.agents.values()}
            lifetime=dict(st)
            d=m5.Diagnostics()
            with patch.object(base,"network_logits",return_value=logits):
                m5._step(
                    clone,0,1,arm,d,visits,globalcells,st,lifetime
                )
            return d,clone
        movement_pref=[10.,0.,0.,0.,-10.]
        neural_birth_pref=[0.,0.,0.,0.,10.]
        for arm in (m5.NN,m5.SN):
            diagnostic,_=perform(arm,movement_pref)
            self.assertEqual(diagnostic.birth_actions,0)
        for arm in (m5.NS,m5.SS):
            diagnostic,_=perform(arm,movement_pref)
            self.assertEqual(diagnostic.birth_actions,1)
            self.assertEqual(diagnostic.scripted_birth_actions,1)
        for arm in (m5.NN,m5.SN):
            diagnostic,_=perform(arm,neural_birth_pref)
            self.assertEqual(diagnostic.birth_actions,1)
            self.assertEqual(diagnostic.neural_birth_actions,1)
        for arm in (m5.NS,m5.SS):
            diagnostic,_=perform(arm,neural_birth_pref)
            self.assertEqual(diagnostic.birth_actions,1)
            self.assertEqual(diagnostic.scripted_birth_actions,1)

    def test_all_human_interventions_only_act_through_their_named_axis(self):
        for arm in m5.ARMS:
            row,_,_=m5.simulate(300,self.small,arm)
            self.assertEqual(
                row["neural_movement"],int(arm in (m5.NN,m5.NS))
            )
            self.assertEqual(
                row["neural_reproduction"],int(arm in (m5.NN,m5.SN))
            )
            self.assertEqual(
                row["scripted_movement"],int(arm in (m5.SN,m5.SS))
            )
            self.assertEqual(
                row["scripted_reproduction"],int(arm in (m5.NS,m5.SS))
            )
            self.assertEqual(
                row["scripted_movement_actions"]>0,arm in (m5.SN,m5.SS)
            )
            if arm in (m5.NN,m5.SN):
                self.assertEqual(row["scripted_birth_actions"],0)
            if arm in (m5.NS,m5.SS):
                self.assertEqual(row["neural_birth_actions"],0)
            self.assertEqual(row["within_world_genetic_mutations"],0)

    def test_diagnostic_records_movement_even_if_agent_starves_and_dies(self):
        eco=base.create_ecology(
            300,self.small,"heritable_no_mutation"
        )
        agent=eco.agents[0]
        for aid in list(eco.agents):
            if aid!=0:
                del eco.agents[aid]
        agent.genome=previous.to_weights(m5.source_genes())
        agent.energy=1.
        visits={0:{agent.position:1}}
        glob={agent.position}
        streak={0:0}
        lifetime={0:0}
        original=agent.position
        d=m5.Diagnostics()
        m5._step(eco,0,1,m5.SN,d,visits,glob,streak,lifetime)
        self.assertNotIn(0,eco.agents)
        self.assertEqual(d.movement_actions,1)
        self.assertEqual(len(visits[0]),2)
        self.assertNotEqual(original,next(
            p for p in visits[0] if p!=original
        ))

    def test_factorial_stats_algebra_and_world_paired_bootstrap(self):
        effects=m5._factorial({
            m5.NN:0.,m5.NS:0.,m5.SN:1.,m5.SS:1.
        })
        self.assertEqual(effects["movement_marginal"],1.)
        self.assertEqual(effects["reproduction_marginal"],0.)
        self.assertEqual(effects["factorial_interaction"],0.)
        self.assertEqual(effects["movement_minus_reproduction"],1.)
        s=self.small
        rows=[]
        for split,seeds in (
            ("main",(300,301)),("shifted",(400,401))
        ):
            for seed in seeds:
                for arm in m5.ARMS:
                    rows.append({
                        "split":split,"seed":seed,"arm":arm,
                        "survived":int(arm in (m5.SN,m5.SS)),
                        "births":10 if arm in (m5.SN,m5.SS) else 0,
                        "food_eaten":100 if arm in (m5.SN,m5.SS) else 0,
                        "alive_final":5 if arm in (m5.SN,m5.SS) else 0,
                        "mean_unique_cells_visited_per_agent_life":3.,
                        "revisit_fraction_of_moves":.5,
                        "odorless_action_fraction":.75,
                        "mean_energy_at_real_birth":46.,
                        "surviving_founder_lineages":1,
                        "world_sha256":f"{seed}-same",
                        "original_401_genome_digest":m5.PINNED_401_DIGEST,
                    })
        a=m5.analyze(rows,s,bootstrap_reps=140)
        self.assertEqual(
            a["main"]["contrasts"]["movement_marginal"]
            ["world_paired_bootstrap_95"],[1.,1.]
        )
        self.assertEqual(
            a["main"]["contrasts"]["reproduction_marginal"]
            ["world_paired_bootstrap_95"],[0.,0.]
        )
        self.assertEqual(
            a["main"]["contrasts"]["food_movement_marginal"]
            ["mean_paired_effect"],100.
        )
        self.assertTrue(a["gates"]["causal_movement_bottleneck_dominant"])
        self.assertFalse(a["gates"]["causal_reproduction_bottleneck_dominant"])
        rows.pop()
        with self.assertRaises(AssertionError):
            m5.analyze(rows,s,bootstrap_reps=110)

    def test_two_deterministic_complete_smoke_runs_preserve_all_negative_rows(self):
        with tempfile.TemporaryDirectory() as where:
            a=Path(where)/"a"
            b=Path(where)/"b"
            first=m5.run(self.small,a,bootstrap_reps=120)
            second=m5.run(self.small,b,bootstrap_reps=120)
            self.assertEqual(first,second)
            self.assertEqual(first["worlds"],4)
            self.assertEqual(first["world_policy_trials"],16)
            self.assertEqual(
                tuple(first["main"]["arm_stats"]),m5.ARMS
            )
            files=(
                "frozen_genotype_and_2x2_intervention.json",
                "all_144_matched_factorial_world_outcomes.csv",
                "all_population_checkpoints.csv",
                "world_geometry_and_rng_sha256.json",
                "summary.json",
                "events_main_300_neural_move_neural_birth.json",
                "events_shifted_400_scripted_move_scripted_birth.json",
            )
            for filename in files:
                self.assertEqual((a/filename).read_bytes(),
                                 (b/filename).read_bytes())
            with (a/"all_144_matched_factorial_world_outcomes.csv").open() as file:
                rows=list(csv.DictReader(file))
            self.assertEqual(len(rows),16)
            self.assertTrue(all(
                abs(float(row["final_energy_ledger_residual"]))<1e-6
                for row in rows
            ))
            self.assertEqual(len({r["original_401_genome_digest"]
                                  for r in rows}),1)
            self.assertEqual(len({r["world_sha256"] for r in rows}),4)


if __name__=="__main__":
    unittest.main()
