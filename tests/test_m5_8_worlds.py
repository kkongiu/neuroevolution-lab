"""M5.8 generator integrity: topology, independence, shock reproducibility."""
import unittest

from neuroevolution.m5_8_worlds import (
    DEV_SEEDS, ZONE_FOOD, audit, make_world, reachable, region,
)


class WorldGeneratorTests(unittest.TestCase):
    def test_same_seed_same_map_food_events(self):
        a = make_world(2000)
        self.assertEqual(a, make_world(2000))
        self.assertNotEqual(a["world_sha256"], make_world(2001)["world_sha256"])
        self.assertTrue(a["parameters"]["events_exogenous"])
        self.assertTrue(a["parameters"]["event_proposals_not_applied_to_m5_7"])

    def test_all_accessible_cells_connected_and_food_zone_budget(self):
        for seed in DEV_SEEDS:
            w = make_world(seed)
            occupied = {tuple(x) for x in w["walls"]}
            cells = {(x,y) for x in range(1,10) for y in range(1,10)}
            free = cells - occupied
            self.assertEqual(reachable(free), free)
            food = [tuple(p) for p in w["food"]]
            self.assertEqual(
                [sum(region(p) == z for p in food) for z in range(3)],
                list(ZONE_FOOD))
            self.assertFalse(set(food) & occupied)
            backbone = {tuple(x) for x in w["backbone"]}
            for e in w["events"]:
                self.assertTrue(1 <= e["tick"] <= w["ticks"])
                if e["location"] is not None:
                    self.assertNotIn(tuple(e["location"]), backbone)

    def test_seed_and_horizon_guard_sealed_worlds(self):
        for seed in (1001, 1080, 161000, 162000, 163000, 2096):
            with self.assertRaises(ValueError):
                make_world(seed)
        with self.assertRaises(ValueError):
            make_world(2000, ticks=61)
        with self.assertRaises(ValueError):
            audit((2000,2000))

    def test_full_audit_and_zone_challenge_gradient(self):
        result = audit()
        stats = result["summary"]
        self.assertEqual(stats["world_count"], 64)
        self.assertEqual(stats["independent_world_shas"],64)
        self.assertEqual(len(result["worlds"]),64)
        self.assertEqual(stats["events_by_zone"],
                         sorted(stats["events_by_zone"]))
        self.assertNotEqual(stats["events_by_zone"][0],
                            stats["events_by_zone"][1])
        self.assertNotEqual(stats["events_by_zone"][1],
                            stats["events_by_zone"][2])


if __name__=="__main__":
    unittest.main()
