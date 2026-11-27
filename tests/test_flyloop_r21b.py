import unittest
from neuroevolution import flyloop_r21a as a, flyloop_r21b as b

class ThermalInertiaTests(unittest.TestCase):
    def test_thermal_lag(self):
        x=b.thermal_update(0.,1.,"inertia")
        self.assertGreater(x,0.)
        self.assertLess(x,.05)
        self.assertEqual(b.thermal_update(.5,.5,"inertia"),.5)
        self.assertLess(b.thermal_update(.8,.1,"fast_cooling"),b.thermal_update(.8,.1,"inertia"))
        self.assertEqual(b.thermal_update(.1,.8,"fast_cooling"),b.thermal_update(.1,.8,"inertia"))
    def test_baseline(self):
        for scene in ("free","lantern"):
            for presence in ("absent","escape"):
                old=a.episode(31100,0,scene,presence,20)
                new=b.episode(31100,0,scene,"immediate:"+presence,20)
                for k in ("captured","survived","thermal_damage","food_gain","steps","contacts"):
                    self.assertEqual(old[k],new[k])
                self.assertEqual([x["position"] for x in old["trace"]],[x["position"] for x in new["trace"]])
    def test_cold_invariance(self):
        base=b.episode(31100,0,"free","immediate:escape",20)
        for model in b.MODELS:
            ep=b.episode(31100,0,"free",model+":escape",20)
            self.assertEqual(ep["trace"],base["trace"])
    def test_repeat(self):
        for arm in b.ARMS:
            ep=b.episode(31100,0,"lantern",arm,20)
            self.assertEqual(ep,b.episode(31100,0,"lantern",arm,20))
            self.assertTrue(all(0<=x["body_heat"]<=1 for x in ep["trace"]))
            if arm.endswith(":absent"):self.assertFalse(ep["captured"])
