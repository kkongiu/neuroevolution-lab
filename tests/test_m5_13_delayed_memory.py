"""M5.13 controls: true frozen test; matched pruning; disjoint seeds."""
import unittest
from neuroevolution.m5_13_delayed_memory import (
    SEEDS,DELAYS,ARMS,ACQUISITION,PROBE,genome,curriculum,run,experiment,
)
class DelayedMemoryTests(unittest.TestCase):
    def test_curriculum_balanced_and_delays(self):
        for d in DELAYS:
            train,probe=curriculum(3600,d)
            self.assertEqual(len(train),ACQUISITION+d)
            self.assertEqual(len(probe),PROBE)
            self.assertTrue(all(ctx==0 for ctx,_ in train[:ACQUISITION]))
            self.assertTrue(all(ctx==0 for ctx,_ in probe))
            for block in (train,probe):
                for t in range(0,len(block),40):
                    self.assertEqual(sum(c for _,c in block[t:t+40]),20)
    def test_rate_matched_pruning_and_truly_frozen_probe(self):
        for d in (0,160,960):
            rows={arm:run(3600,d,arm) for arm in ARMS}
            self.assertEqual(rows["static"]["final_active"],200)
            self.assertEqual(len({
                tuple(r["prune_events"]) for arm,r in rows.items()
                if arm!="static"
            }),1)
            for r in rows.values():
                self.assertEqual(r["frozen_probe_steps"],40)
                self.assertTrue(r["frozen_probe"])
                self.assertFalse(r["explicit_context_router"])
                self.assertTrue(0<=r["frozen_probe_accuracy"]<=1)
            self.assertEqual(rows["historical_usage"],
                             run(3600,d,"historical_usage"))
    def test_separate_development_cohorts(self):
        for seed in (3500,3620,161000,162000,163000):
            with self.assertRaises(ValueError):genome(seed)
            with self.assertRaises(ValueError):run(seed,0,"static")
        with self.assertRaises(ValueError):run(3600,777,"static")
        with self.assertRaises(ValueError):experiment((3600,3600))
        report=experiment((3600,3601))
        self.assertTrue(report["pruning_dose_timing_matched"])
        self.assertEqual(report["advance_to_evolution"],
                         all(report["screening_gate"].values()))
if __name__=="__main__":unittest.main()
