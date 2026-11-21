"""M5.4A paired environment factor isolation, genotype-free scripted benchmark."""
import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from neuroevolution import m5_0 as base
from neuroevolution import m5_4a as f


class MatchedEcologicalFactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small=f.Settings(
            seed_start=700,seeds=2,
            shifted_start=900,shifted_seeds=2,
            ticks=40,size=11,initial_food=9,food_cap=14,
            initial_population=8,max_population=18,
            checkpoint_every=10
        )
        cls.small.validate()
        cls.bank,_=f.candidate_bank()

    def test_preregistered_independent_splits_and_fixed_laws(self):
        f.Settings().validate()
        self.assertEqual(len(f.factorial_conditions()),8)
        self.assertEqual(f.BOOT_MAIN,20000)
        self.assertEqual(f.BOOT_VALIDATION,5000)
        with self.assertRaises(ValueError):
            replace(f.Settings(),seed_start=82006).validate()
        with self.assertRaises(ValueError):
            replace(f.Settings(),shifted_start=86000).validate()
        with self.assertRaises(ValueError):
            replace(f.Settings(),ticks=400,odor_radius=4).validate()
        with self.assertRaises(ValueError):
            replace(f.Settings(),hidden=2).validate()

    def test_matched_maps_have_nested_walls_common_food_founders(self):
        for seed in (700,701,702):
            p=f.coupled_world(seed,self.small)
            self.assertEqual(p,f.coupled_world(seed,self.small))
            low,high=p["world_by_wall_flag"]
            self.assertTrue(low.walls<=high.walls)
            self.assertEqual(low.initial_food,high.initial_food)
            self.assertEqual(len(low.initial_food),self.small.initial_food)
            self.assertEqual(
                len(p["initial_founders"]),self.small.initial_population
            )
            self.assertEqual(len(set(p["initial_founders"])),
                             self.small.initial_population)
            self.assertTrue(set(p["initial_founders"]) <= high.reachable())
            self.assertFalse(set(p["initial_founders"])&high.walls)
            self.assertFalse(set(p["initial_founders"])&set(low.initial_food))
            self.assertTrue(
                set(p["common_food_candidate_pool"])<=high.reachable()
            )
            self.assertEqual(p["food_candidate_pool_digest"],
                             f._digest(p["common_food_candidate_pool"]))

    def test_odors_and_regrowth_do_not_rewrite_terrain_or_founders(self):
        p=f.coupled_world(700,self.small)
        sample=f.common_food_schedule(700,self.small,p)
        self.assertEqual(sample,f.common_food_schedule(700,self.small,p))
        for walls in (0,1):
            for smell in (0,1):
                for renewal in (0,1):
                    s=(walls,smell,renewal)
                    eco=f.new_ecology(700,self.small,p,s,
                                      self.bank[f.FAMILIES[0]]["descendant"])
                    self.assertEqual(eco.world.walls,
                                     p["world_by_wall_flag"][walls].walls)
                    self.assertEqual(eco.food,set(p["initial_food"]))
                    self.assertEqual(
                        tuple(a.position for a in eco.agents.values()),
                        p["initial_founders"]
                    )
                    self.assertEqual(eco.settings.odor_radius,
                                     2 if smell else 5)
                    self.assertEqual(
                        eco.settings.regeneration_probability,
                        .40 if renewal else .55
                    )

    def test_regrowth_event_subset_position_and_marks_matched(self):
        p=f.coupled_world(700,self.small)
        schedule=f.common_food_schedule(700,self.small,p)
        low=[[(v<.40,cell) for v,cell in ev] for ev in schedule]
        high=[[(v<.55,cell) for v,cell in ev] for ev in schedule]
        self.assertTrue(all(
            not low[t][i][0] or high[t][i][0]
            for t in range(len(schedule))
            for i in range(self.small.regeneration_attempts)
        ))
        self.assertEqual(
            [[a[1] for a in timestep] for timestep in low],
            [[a[1] for a in timestep] for timestep in high]
        )

    def test_original_neural_act_and_frozen_genome_cannot_mutate(self):
        for policy in f.NEURAL_GENOTYPES:
            genes=self.bank[f.FAMILIES[0]][policy]
            sample=f.coupled_world(700,self.small)
            schedule=f.common_food_schedule(700,self.small,sample)
            for flags in f.factorial_conditions():
                row,checks,trace=f.simulate(
                    700,self.small,sample,schedule,flags,policy,
                    source_seed=f.FAMILIES[0],genes=genes,detailed=True
                )
                self.assertEqual(row["native_neural_action"],1)
                self.assertEqual(row["fixed_clonal_genome"],1)
                self.assertEqual(row["energy_ledger_residual"],0)
                self.assertTrue(all(
                    x["energy_ledger_residual"]==0 for x in checks
                ))
                self.assertTrue(all(
                    x["genome"]==row["genome_digest"]
                    for x in trace["events"] if x["type"]=="birth"
                ))
                self.assertTrue(all(
                    tuple(evt["location"]) not in
                    set(map(tuple,trace["world_walls"]))
                    for evt in trace["events"] if evt["type"]=="birth"
                ))

    def test_scripted_benchmark_is_explicitly_not_neural(self):
        p=f.coupled_world(700,self.small)
        schedule=f.common_food_schedule(700,self.small,p)
        args=(700,self.small,p,schedule,(0,0,0),
              "scripted_greedy_novelty")
        a=f.simulate(*args,detailed=True)
        b=f.simulate(*args,detailed=True)
        self.assertEqual(a,b)
        self.assertEqual(a[0]["scripted_positive_benchmark"],1)
        self.assertEqual(a[0]["native_neural_action"],0)
        self.assertIsNone(a[0]["genome_digest"])
        self.assertAlmostEqual(a[0]["energy_ledger_residual"],0.)
        with self.assertRaises(ValueError):
            f.simulate(*args,genes=self.bank[f.FAMILIES[0]]["descendant"])
        with self.assertRaises(ValueError):
            f.simulate(
                700,self.small,p,schedule,(0,0,0),
                "descendant",source_seed=f.FAMILIES[0],genes=None
            )

    def test_independent_factors_change_only_intended_configuration(self):
        p=f.coupled_world(700,self.small)
        g=self.bank[f.FAMILIES[0]]["descendant"]
        a=f.new_ecology(700,self.small,p,(0,0,0),g)
        b=f.new_ecology(700,self.small,p,(0,1,0),g)
        c=f.new_ecology(700,self.small,p,(0,0,1),g)
        d=f.new_ecology(700,self.small,p,(1,0,0),g)
        self.assertEqual(a.world,b.world)
        self.assertEqual(a.world,c.world)
        self.assertEqual(a.world.initial_food,d.world.initial_food)
        self.assertEqual(tuple(x.position for x in a.agents.values()),
                         tuple(x.position for x in d.agents.values()))
        self.assertEqual(a.settings.regeneration_probability,
                         b.settings.regeneration_probability)
        self.assertEqual(a.settings.odor_radius,c.settings.odor_radius)
        self.assertEqual(a.settings.odor_radius,d.settings.odor_radius)

    def test_statistic_pairs_worlds_families_and_all_four_other_combos(self):
        rows=[]
        for split,worlds in (("main",(700,701)),("validation",(900,901))):
            for seed in worlds:
                for flags in f.factorial_conditions():
                    for source in f.FAMILIES:
                        for kind in f.NEURAL_GENOTYPES:
                            rows.append({
                                "split":split,"seed":seed,"source_seed":source,
                                "kind":kind,"walls_stress":flags[0],
                                "odor_stress":flags[1],"regrowth_stress":flags[2],
                                "survived":int(flags[1]==0),
                                "food_eaten":8,"births":3
                            })
                    rows.append({
                        "split":split,"seed":seed,"source_seed":None,
                        "kind":"scripted_greedy_novelty",
                        "walls_stress":flags[0],
                        "odor_stress":flags[1],"regrowth_stress":flags[2],
                        "survived":1,"food_eaten":8,"births":3
                    })
        result=f.analyze(rows,self.small,boot_main=110,
                         boot_validation=110)
        self.assertEqual(
            result["main"]["effects"]["odor"]["mean_stress_minus_benign"],
            -1.0
        )
        self.assertEqual(
            result["main"]["effects"]["walls"]["mean_stress_minus_benign"],
            0.0
        )
        self.assertTrue(result["main"]["factor_bottleneck_supported"])
        self.assertEqual(
            result["main"]["most_negative_factor"],"odor"
        )
        # The sealed benchmark gate is >=4 of the EIGHT validation
        # worlds. This synthetic smoke has only TWO validation worlds:
        # even 2/2 successes must correctly remain below the gate.
        self.assertEqual(
            result["gates"]["scripted_all_hard_validation_survivors"],2
        )
        self.assertFalse(
            result["gates"]["scripted_all_hard_physical_viability"]
        )
        self.assertTrue(all(
            result["gates"]["validation_sign_concordance"][x]
            for x in ("odor",)
        ))
        rows.pop()
        with self.assertRaises(AssertionError):
            f.analyze(rows,self.small,boot_main=100,boot_validation=100)

    def test_two_full_smoke_runs_have_same_352_trials(self):
        with tempfile.TemporaryDirectory() as temp:
            left=Path(temp)/"a"
            right=Path(temp)/"b"
            a=f.run(self.small,left,boot_main=120,boot_validation=100)
            b=f.run(self.small,right,boot_main=120,boot_validation=100)
            self.assertEqual(a,b)
            self.assertEqual(a["simulation_count"],352)
            for file in (
                "frozen_numerical_genotype_bank.json",
                "world_metrics.csv","summary.json",
                "population_checkpoints.csv","coupled_world_geometry.json",
                "factorial_survival.svg",
                "events_main_700_w0o0r0_descendant_82006.json",
                "events_validation_900_w1o1r1_scripted_greedy_novelty.json",
            ):
                self.assertEqual((left/file).read_bytes(),
                                 (right/file).read_bytes())
            with (left/"world_metrics.csv").open(newline="") as source:
                csv_rows=list(csv.DictReader(source))
            self.assertEqual(len(csv_rows),352)
            self.assertTrue(all(
                abs(float(r["energy_ledger_residual"]))<1e-6
                for r in csv_rows
            ))


if __name__=="__main__":
    unittest.main()
