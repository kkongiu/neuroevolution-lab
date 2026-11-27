import unittest
import math
from neuroevolution import flyloop_r21d as t

class EnergyTests(unittest.TestCase):
    def test_capacity(self):
        for e,v in ((90,1),(30,1),(20,2/3),(10,1/3),(0,0)):
            self.assertAlmostEqual(t.capacity(e),v)
            self.assertEqual(t.capacity(e,False),1)
        self.assertEqual(t.limit((.8,-.7),1),(.8,-.7))
        self.assertEqual(t.limit((.8,-.7),0),(0,0))
        v,w=t.limit((1.,1.),.5)
        self.assertEqual(v,.5)
        self.assertAlmostEqual(math.tanh(1.6*w),.5*math.tanh(1.6))
    def test_above_threshold(self):
        for scene in t.SCENARIOS:
            a=t.episode(33100,0,scene,'normal50',20)
            b=t.episode(33100,0,scene,'weak50',20)
            a.pop('arm');b.pop('arm')
            self.assertEqual(a,b)
    def test_parent(self):
        a=t.episode(33100,0,'free','normal50',20)
        b=t.prev.episode(33100,0,'free','clone',20)
        for k in ('captured','survived','ate','thermal_damage','predator_energy','steps'):
            self.assertEqual(a[k],b[k])
    def test_repeat(self):
        for arm in t.ARMS:
            a=t.episode(33100,0,'lantern',arm,20)
            self.assertEqual(a,t.episode(33100,0,'lantern',arm,20))
            self.assertIn(a['prey_cause'],('capture','starvation','injury','censored'))
