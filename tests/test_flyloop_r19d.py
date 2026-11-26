import unittest
from neuroevolution import flyloop_r19d as t

class StimulusTests(unittest.TestCase):
    def test_mirror(self):
        eye=t.r.CompoundEye()
        p=(0.,0.,0.,0.,.8,.6,.5,.2,0.)
        l=(0.,0.,0.,0.,.8,0.,0.,0.,0.)
        for arm in ("vector","affordance"):
            x=t.command((0.,0.),p,l,.6,eye,arm)
            y=t.command((0.,0.),p[::-1],l[::-1],.6,eye,arm)
            self.assertAlmostEqual(x[0],y[0])
            self.assertAlmostEqual(x[1],-y[1])
    def test_clear_side(self):
        eye=t.r.CompoundEye()
        p=(0.,0.,0.,0.,.8,.8,.8,.8,.8)
        l=(0.,0.,0.,0.,.8,0.,0.,0.,0.)
        self.assertLess(t.command((0.,0.),p,l,.6,eye,"affordance")[1],0.)
    def test_mask_and_repeat(self):
        ep=t.episode(25100,0,"glass","masked",12)
        self.assertTrue(all(not any(f["light"]) for f in ep["trace"]))
        self.assertEqual(ep,t.episode(25100,0,"glass","masked",12))
        for arm in t.ARMS:
            other=t.episode(25100,0,"glass",arm,12)
            for k in ("start","lamp","rectangles","opening","initially_blocked"):
                self.assertEqual(ep[k],other[k])
    def test_sealed(self):
        for arm in t.ARMS:
            ep=t.episode(25100,0,"sealed",arm,20)
            self.assertTrue(ep["initially_blocked"])
            self.assertFalse(ep["crossed"])
            self.assertFalse(ep["encounter"])
