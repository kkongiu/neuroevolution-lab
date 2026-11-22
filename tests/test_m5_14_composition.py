"""M5.14: novel A/B composition, no answer leakage and matched structural growth."""
import unittest

from neuroevolution.m5_14_composition import (
    SEEDS, ARMS, CAPACITY, INITIAL_ACTIVE, PHASES, TOTAL_STEPS,
    TRAINING_STEPS, starter, start_edges, hidden_rules, curriculum,
    target, sensory, GrowingBrain, run, experiment,
)


class M514CompositionTests(unittest.TestCase):
    def test_heldout_composition_is_nontrivial_and_operator_is_seen(self):
        seed=3700
        h=hidden_rules(seed)
        self.assertEqual(len(h),2)
        self.assertEqual(len(curriculum(seed)),TOTAL_STEPS)
        self.assertEqual(TRAINING_STEPS,sum(n for _,n,learn in PHASES if learn))
        for x in (0,1):
            for y in (0,1):
                self.assertEqual(target("C",x,y,h),
                                 target("A",x,y,h)^target("B",x,y,h))
                self.assertEqual(target("P",x,y,h),x^y)
        self.assertEqual(sensory("P",0,1,0)[9],1.)
        self.assertEqual(sensory("C",0,1,0)[9],1.)
        self.assertEqual(sensory("C",0,1,0)[10:12],(1.,1.))
        self.assertEqual(sensory("A",0,1,0)[10:12],(1.,0.))
        self.assertEqual(sensory("B",0,1,0)[10:12],(0.,1.))
        for task,x,y,enabled in curriculum(seed):
            self.assertTrue(task in ("P","A","B","C"))
            self.assertEqual(len(sensory(task,x,y,0)),19)
        self.assertEqual(sum(n for name,n,can_train in PHASES
                             if name=="C" and not can_train),80)

    def test_same_birth_genomes_no_dormant_synaptic_learning(self):
        seed=3700
        small=GrowingBrain(seed,"fixed_sparse")
        adaptive=GrowingBrain(seed,"adaptive_birth")
        dense=GrowingBrain(seed,"full_dense")
        self.assertEqual(small.active,adaptive.active)
        self.assertEqual(len(small.active),INITIAL_ACTIVE)
        self.assertEqual(len(dense.active),CAPACITY)
        self.assertEqual(len(starter(seed)),CAPACITY)
        self.assertEqual(start_edges(seed),small.active)
        for i in range(CAPACITY):
            if i not in small.active:
                self.assertEqual(small.agent.weights[i],0.)

    def test_matched_yoked_birth_events_and_frozen_probes(self):
        for seed in (3700,3701):
            adaptive=run(seed,"adaptive_birth")
            plan={e["tick"]:e["count"] for e in adaptive["events"]
                  if e["kind"]=="birth"}
            yoked=run(seed,"yoked_random_locations",yoked_plan=plan)
            self.assertEqual(
                sorted((e["tick"],e["count"]) for e in adaptive["events"]),
                sorted((e["tick"],e["count"]) for e in yoked["events"]))
            self.assertEqual(adaptive["synapses_born"],yoked["synapses_born"])
            self.assertEqual(adaptive["mean_active_synapses"],
                             yoked["mean_active_synapses"])
            self.assertTrue(adaptive["all_probe_synapses_unchanged"])
            self.assertTrue(yoked["all_probe_synapses_unchanged"])
            self.assertEqual(adaptive,run(seed,"adaptive_birth"))
            self.assertEqual(len(adaptive["phase_scores"]),len(PHASES))
            self.assertEqual(adaptive["phase_scores"][5]["task"],"C")
            self.assertFalse(adaptive["phase_scores"][5]["learning_enabled"])
            self.assertTrue(adaptive["phase_scores"][6]["learning_enabled"])

    def test_validity_fences_and_evolution_gate(self):
        for seed in (3600,3724,161000,162000,163000):
            with self.assertRaises(ValueError):starter(seed)
            with self.assertRaises(ValueError):run(seed,"adaptive_birth")
        with self.assertRaises(ValueError):
            run(3700,"yoked_random_locations")
        with self.assertRaises(ValueError):experiment((3700,3700))
        result=experiment((3700,3701))
        self.assertFalse(result["selection_or_reproduction"])
        self.assertTrue(result["frozen_pre_C_learning_probe"])
        self.assertTrue(result["matched_yoked_birth_counts_times_and_occupancy"])
        self.assertEqual(result["advance_to_evolution"],
                         all(result["screening_gate"].values()))

if __name__=="__main__":
    unittest.main()
