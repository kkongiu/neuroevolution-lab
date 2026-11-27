import unittest
from neuroevolution import flyloop_r21a as t

class PredatorTests(unittest.TestCase):
    def test_occlusion(self):
        p=t.r1.Body(x=5,y=5,heading=0)
        q=t.r1.Body(x=7,y=5)
        self.assertGreater(max(t.sense(p,q,[],3.5)),0)
        self.assertEqual(max(t.sense(p,q,[(5.8,4,6.2,6)],3.5)),0)
        self.assertGreater(max(t.sense(p,q,[(5.8,4,6.2,6)],3.5,False)),0)
        self.assertEqual(max(t.sense(p,q,[],1)),0)
    def test_flight(self):
        base=(.1,.2);eye=t.r.CompoundEye()
        self.assertEqual(t.flight(base,(0.,)*12,(0.,)*9,eye),(base,0.))
        threat=(0.,)*6+(1.,)+(0.,)*5
        cmd,u=t.flight(base,threat,(0.,)*9,eye)
        self.assertEqual(u,1.)
        self.assertEqual(cmd[0],1.)
        self.assertAlmostEqual(cmd[1],0.)
    def test_repeat(self):
        for scenario in t.SCENARIOS:
            for arm in t.ARMS:
                x=t.episode(30100,0,scenario,arm,12)
                self.assertEqual(x,t.episode(30100,0,scenario,arm,12))
                if arm=="absent":
                    self.assertFalse(x["captured"])
                    self.assertEqual(x["escape_fraction"],0.)
                if arm=="blind":
                    self.assertEqual(x["escape_fraction"],0.)
    def test_layouts(self):
        for s in (30100,30123):
            for i in range(20):
                for scenario in t.SCENARIOS:
                    w,p=t.world(s,i,scenario)
                    self.assertFalse(w.blocked(w.body.x,w.body.y))
                    self.assertFalse(t.blocked(p.x,p.y,w.rectangles,.45))
