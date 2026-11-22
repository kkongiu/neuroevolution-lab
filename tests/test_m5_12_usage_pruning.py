"""Scientific invariants for M5.12 probabilistic local synapse pruning."""
import unittest
from neuroevolution.m5_12_usage_pruning import (
    SEEDS,ARMS,INITIAL_EDGES,MIN_EDGES,TOTAL_EDGES,
    SparseBrain,run,experiment,sample_prunes,task,observation,
)

class UsagePruningTests(unittest.TestCase):
    def test_initial_matching_and_dormant_mask(self):
        for arm in ARMS:
            b=SparseBrain(3500,arm)
            n=TOTAL_EDGES if arm=="full_260" else INITIAL_EDGES
            self.assertEqual(len(b.active),n)
            self.assertEqual(sum(w!=0 for w in b.agent.weights),n)
        self.assertEqual(len(task(3500)),800)
        self.assertEqual(len(observation(0,1,0)),19)

    def test_probabilistic_weak_edge_selection_and_uniqueness(self):
        import random
        weights=[0.]*TOTAL_EDGES
        weights[0]=0.
        weights[1]=1.
        chosen=[sample_prunes({0,1},weights,1,random.Random(i))[0]
                for i in range(100)]
        self.assertGreater(chosen.count(0),90)
        self.assertEqual(len(set(sample_prunes({0,1,2},weights,2,
                                               random.Random(3)))),2)

    def test_identical_pruning_counts_timing_and_reproducibility(self):
        arms=("random_prune","magnitude_prune","usage_prune",
              "usage_consolidated")
        for s in (3500,3501):
            records=[run(s,a) for a in arms]
            schedules=[tuple((e["at_tick"],e["pruned"],e["remaining"])
                             for e in record["pruning_events"])
                       for record in records]
            self.assertEqual(len(set(schedules)),1)
            self.assertTrue(all(r["final_active_synapses"]==MIN_EDGES
                                for r in records))
            self.assertTrue(all(r["deleted_synapses"]==
                                INITIAL_EDGES-MIN_EDGES for r in records))
            self.assertEqual(records[0],run(s,"random_prune"))

    def test_no_selection_no_old_sealed_seeds_and_gate(self):
        for invalid in (3400,3524,161000,162000,163000):
            with self.assertRaises(ValueError):run(invalid,"usage_consolidated")
        with self.assertRaises(ValueError):experiment((3500,3500))
        result=experiment((3500,3501))
        self.assertFalse(result["genetic_selection"])
        self.assertTrue(result["matched_pruning_dose_and_timing"])
        self.assertEqual(result["advance_to_evolution"],
                         all(result["screening_gate"].values()))

if __name__=="__main__":
    unittest.main()
