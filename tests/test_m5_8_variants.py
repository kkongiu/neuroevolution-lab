"""M5.8 matched environmental budget and complete paired experiment."""
import unittest

from neuroevolution.m5_8_variants import variant_world, audit_variants, VARIANTS
from neuroevolution.m5_8_physical_experiment import evaluate
from neuroevolution.m5_8_worlds import region, reachable


class MatchedWorldTests(unittest.TestCase):
    def test_static_and_event_budgets_are_matched(self):
        for seed in (2000,2001,2020,2063):
            original=variant_world(seed,"progressive")
            for variant in VARIANTS:
                w=variant_world(seed,variant)
                self.assertEqual(w,variant_world(seed,variant))
                self.assertEqual(len(w["walls"]),len(original["walls"]))
                self.assertEqual(len(w["food"]),len(original["food"]))
                self.assertEqual(len(w["events"]),len(original["events"]))
                self.assertEqual(
                    sorted((e["tick"],e["kind"],e["duration"],e["intensity"])
                           for e in w["events"]),
                    sorted((e["tick"],e["kind"],e["duration"],e["intensity"])
                           for e in original["events"]))
                cells={(x,y) for x in range(1,10) for y in range(1,10)}
                free=cells-{tuple(p) for p in w["walls"]}
                self.assertEqual(reachable(free),free)
            reversed_world=variant_world(seed,"reversed")
            self.assertEqual(
                sorted([2-region(tuple(p)) for p in original["food"]]),
                sorted(region(tuple(p)) for p in reversed_world["food"]))

    def test_full_variant_audit(self):
        r=audit_variants((2000,2001,2002))
        self.assertEqual(r["world_count"],3)
        self.assertEqual(len(r["variants"]),3)
        for key in ("wall_counts_by_zone","food_counts_by_zone","events_by_zone"):
            self.assertEqual(
                sum(r["variants"]["progressive"][key]),
                sum(r["variants"]["uniform"][key]))
            self.assertEqual(
                sum(r["variants"]["progressive"][key]),
                sum(r["variants"]["reversed"][key]))

    def test_one_seed_six_physical_conditions(self):
        r=evaluate((2000,))
        self.assertEqual(r["physical_trials"],6)
        self.assertEqual(r["total_comparison_arms"],6)
        self.assertTrue(all(abs(x["energy_residual"])<1e-6
                            for x in r["raw_rows"]))
        self.assertTrue(all(x["post_cutoff_weight_updates"]==0
                            for x in r["raw_rows"]))

    def test_sealed_worlds_rejected(self):
        for seed in (1080,161000,162000,163000):
            with self.assertRaises(ValueError):
                evaluate((seed,))
        with self.assertRaises(ValueError):
            evaluate((2000,2000))
        with self.assertRaises(ValueError):
            variant_world(2000,"nonexistent")


if __name__=="__main__":
    unittest.main()
