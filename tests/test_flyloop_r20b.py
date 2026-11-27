import unittest
import math
from neuroevolution import flyloop_r20b as t

class HomeostaticTests(unittest.TestCase):
    def test_layout(self):
        w=t.world(27100,0,30)
        v=t.world(27100,0,90)
        self.assertEqual(w.lamp,v.lamp)
        self.assertEqual(w.initial_food,v.initial_food)
        self.assertAlmostEqual(math.dist(w.lamp,(11,11)),3)
        self.assertAlmostEqual(math.dist(w.initial_food[0],(11,11)),3)
        self.assertNotEqual(w.lamp,w.initial_food[0])
    def test_weights(self):
        l,o=t.weights("adaptive",.7,.7)
        self.assertGreater(o,l)
        l,o=t.weights("adaptive",.1,.7)
        self.assertGreater(l,o)
        self.assertEqual(t.weights("clamped",.1,.7),t.weights("adaptive",.7,.7))
        self.assertEqual(t.weights("fixed",.1,.7),(1.,1.))
    def test_odor_removal(self):
        w=t.world(27100,0,30); eye=t.r.CompoundEye()
        before=t.odor(w,eye,False)
        w.food.clear()
        self.assertFalse(any(t.odor(w,eye,False)[0]))
        self.assertEqual(t.odor(w,eye,True),before)
    def test_determinism_and_energy(self):
        for arm in t.ARMS:
            x=t.episode(27100,0,60,arm,12)
            self.assertEqual(x,t.episode(27100,0,60,arm,12))
            self.assertEqual(len(x["trace"]),48)
            self.assertAlmostEqual(60-sum(f["metabolic_cost"] for f in x["trace"])+sum(f["food_gain"] for f in x["trace"]),x["final_energy"])
    def test_continuation(self):
        w=t.world(27100,0,60)
        w.body.x,w.body.y=w.initial_food[0]
        first=w.micro_integrate((0.,0.))
        self.assertGreater(first["food_gain"],0)
        self.assertFalse(w.food)
        second=w.micro_integrate((0.,0.))
        self.assertEqual(second["food_gain"],0)
