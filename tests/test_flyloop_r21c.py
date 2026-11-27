import unittest
from neuroevolution import flyloop_r21c as t

class CloneTests(unittest.TestCase):
    def test_shared_body(self):
        w,p=t.world(32100,0,'lantern')
        pw=t.clone_world(w,p,32100,0,'clone_equal_energy')
        self.assertEqual(pw.plant,w.plant)
        self.assertEqual(pw.body.energy,w.body.energy)
        self.assertEqual(pw.lamp,w.lamp)
        self.assertEqual(pw.rectangles,w.rectangles)
        self.assertEqual(pw.food,[])
        for _ in range(20):
            pw.micro_integrate((1.,0.))
        self.assertGreater(pw.body.speed,0.)
        self.assertLessEqual(pw.body.speed,.95+1e-9)
    def test_cold_invariance(self):
        for scene in ('free','obstacle'):
            a=t.episode(32100,0,scene,'clone',20)
            b=t.episode(32100,0,scene,'clone_no_heat',20)
            a.pop('arm');b.pop('arm')
            self.assertEqual(a,b)
    def test_repeat(self):
        for arm in t.ARMS:
            a=t.episode(32100,0,'lantern',arm,20)
            self.assertEqual(a,t.episode(32100,0,'lantern',arm,20))
            self.assertGreaterEqual(a['predator_damage'],0.)
    def test_legacy(self):
        a=t.episode(32100,0,'lantern','legacy',20)
        b=t.parent.episode(32100,0,'lantern','inertia:escape',20)
        for k in ('captured','survived','ate','thermal_damage','steps'):
            self.assertEqual(a[k],b[k])
