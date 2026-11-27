import unittest
from neuroevolution import flyloop_r20c as t

class ThermalTests(unittest.TestCase):
    def test_field(self):
        w=t.world(28100,0)
        x,y=w.lamp
        self.assertEqual(t.heat(w,x,y,1.),1.)
        self.assertAlmostEqual(t.heat(w,x+1,y,1.),.6)
        self.assertEqual(t.heat(w,x+3,y,1.),0.)
        self.assertEqual(t.heat(w,x,y,0.),0.)
    def test_priority(self):
        samples=(1.,.8,.5,.2,0.,.2,.5,.8)
        b=(.8,0.)
        self.assertEqual(t.protect(b,.3,samples,"priority"),(b,0.))
        cmd,u=t.protect(b,.65,samples,"priority")
        self.assertEqual(u,1.)
        self.assertAlmostEqual(cmd[0],-1.)
        self.assertAlmostEqual(cmd[1],0.)
        cmd,u=t.protect(b,.65,samples,"additive")
        self.assertAlmostEqual(cmd[0],-.2)
    def test_cold_equivalence(self):
        rows=[t.episode(28100,0,"cold",a,20) for a in t.ARMS]
        for ep in rows:
            self.assertEqual(ep["trace"],rows[0]["trace"])
            self.assertEqual(ep["thermal_damage"],0.)
    def test_repeat(self):
        for arm in t.ARMS:
            ep=t.episode(28100,0,"hot",arm,40)
            self.assertEqual(ep,t.episode(28100,0,"hot",arm,40))
            self.assertGreaterEqual(ep["thermal_damage"],0.)
            self.assertLessEqual(ep["final_integrity"],100.)
