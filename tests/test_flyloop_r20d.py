import unittest
import math
from neuroevolution import flyloop_r20d as t

class LanternTests(unittest.TestCase):
    def test_layout(self):
        worlds=[t.world(29100,0,90,a) for a in t.ARMS]
        w=worlds[0]
        self.assertAlmostEqual(math.dist((w.body.x,w.body.y),w.lamp),1.37)
        self.assertAlmostEqual(math.dist(w.initial_food[0],w.lamp),3.5)
        for v in worlds:
            self.assertEqual(v.initial_food,w.initial_food)
            self.assertEqual((v.body.x,v.body.y,v.body.heading),(w.body.x,w.body.y,w.body.heading))
        self.assertFalse(worlds[2].food)
    def test_repeat(self):
        for arm in t.ARMS:
            ep=t.episode(29100,0,90,arm,20)
            self.assertEqual(ep,t.episode(29100,0,90,arm,20))
            self.assertEqual(len(ep["trace"]),80)
            self.assertGreaterEqual(ep["thermal_damage"],0.)
    def test_no_food_and_cold(self):
        ep=t.episode(29100,0,60,"no_food",20)
        self.assertIsNone(ep["meal"])
        self.assertIsNone(ep["returned"])
        self.assertTrue(all(v["food_gain"]==0 for v in ep["trace"]))
        ep=t.episode(29100,0,60,"adaptive_cold",20)
        self.assertTrue(all(v["heat"]==0 and v["urgency"]==0 for v in ep["trace"]))
