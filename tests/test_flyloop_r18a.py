import math
import unittest
from neuroevolution import flyloop_r18a as r
from neuroevolution import flyloop_r1 as r1

class EyeTests(unittest.TestCase):
    def test_sectors(self):
        for n in (3,9,11):
            eye=r.CompoundEye(n)
            self.assertEqual(len(eye.centers),n)
            self.assertAlmostEqual(eye.centers[0],-eye.centers[-1])
            self.assertAlmostEqual(eye.centers[n//2],0.)
            p,l=eye.observe(r.WindowWorld(21100,0,"free"))
            self.assertEqual(len(p)+len(l),2*n)
            self.assertTrue(all(0<=x<=1 for x in p+l))
        with self.assertRaises(ValueError):
            r.CompoundEye(4)

    def test_glass_and_opaque(self):
        eye=r.CompoundEye()
        worlds=[r.WindowWorld(21100,0,p) for p in ("free","glass","opaque","sealed")]
        for w in worlds:
            w.body.x=9.
            w.body.y=11.
            w.body.heading=0.
            w.lamp=(13.,11.)
            if w.rectangles:
                w.rectangles=[(10.92,0.,11.08,22.)]
        observations=[eye.observe(w) for w in worlds]
        self.assertEqual(observations[0][0][4],0.)
        self.assertGreater(observations[1][0][4],0.)
        self.assertEqual(observations[1][0],observations[2][0])
        self.assertAlmostEqual(max(observations[1][1]),.9*max(observations[0][1]))
        self.assertEqual(max(observations[2][1]),0.)
        self.assertTrue(worlds[1].blocked(11.,11.))
        self.assertFalse(worlds[0].blocked(11.,11.))

    def test_mirror(self):
        eye=r.CompoundEye()
        w=r.WindowWorld(21100,0,"free")
        w.body.x=w.body.y=11.
        w.body.heading=0.
        w.lamp=(13.,12.)
        w.rectangles=[(12.,12.,12.4,12.4)]
        p,l=eye.observe(w)
        w.lamp=(13.,10.)
        w.rectangles=[(12.,9.6,12.4,10.)]
        q,k=eye.observe(w)
        for a,b in zip(p,reversed(q)):
            self.assertAlmostEqual(a,b)
        for a,b in zip(l,reversed(k)):
            self.assertAlmostEqual(a,b)

    def test_opening_and_sealed(self):
        for i in range(20):
            w=r.WindowWorld(21100,i,"glass")
            self.assertFalse(w.blocked(w.wall_x,sum(w.opening)/2))
            self.assertTrue(w.blocked(w.wall_x,.5))
            self.assertTrue(w.blocked(w.wall_x,r1.SIDE-.5))
            s=r.WindowWorld(21100,i,"sealed")
            self.assertTrue(all(s.blocked(s.wall_x,y/10) for y in range(3,218)))

    def test_mask_and_repeat(self):
        a=r.episode(21100,0,"glass","masked",8)
        self.assertTrue(all(not any(t["light"]) and not any(t["effective_light"]) for t in a["trace"]))
        self.assertEqual(a,r.episode(21100,0,"glass","masked",8))
        b=r.episode(21100,0,"glass","clearance",8)
        self.assertEqual(a["start"],b["start"])
        self.assertEqual(a["lamp"],b["lamp"])
        self.assertEqual(a["rectangles"],b["rectangles"])

    def test_clearance_symmetry(self):
        eye=r.CompoundEye()
        p=(.8,)*4+(.5,)+(.1,)*4
        a=r.visual_motor_adapter((0.,0.),p,(.5,0.,0.),.6,eye,True)
        b=r.visual_motor_adapter((0.,0.),tuple(reversed(p)),(.5,0.,0.),.6,eye,True)
        self.assertAlmostEqual(a[0],b[0])
        self.assertAlmostEqual(a[1],-b[1])
        self.assertGreater(a[1],0.)
