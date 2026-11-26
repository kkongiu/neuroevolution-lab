import math
import unittest
from neuroevolution import flyloop_r17 as r

class MemoryTests(unittest.TestCase):
    def test_empty_and_decay(self):
        m=r.Memory()
        self.assertEqual(m.observe((0.,0.,0.)),((0.,0.,0.),False))
        self.assertEqual(m.observe((.8,.4,0.)),((.8,.4,0.),False))
        for _ in range(10):
            odor,active=m.observe((0.,0.,0.))
            self.assertTrue(active)
        self.assertAlmostEqual(max(odor),.4)
        for _ in range(100):
            odor,active=m.observe((0.,0.,0.))
        self.assertFalse(active)
        self.assertEqual(odor,(0.,0.,0.))

    def test_rotation_and_refresh(self):
        m=r.Memory()
        m.observe((.8,0.,0.))
        m.rotate(math.pi/2)
        odor,active=m.observe((0.,0.,0.))
        self.assertTrue(active)
        self.assertGreater(odor[2],odor[1])
        self.assertEqual(m.observe((.2,.2,0.)),((.2,.2,0.),False))
        self.assertAlmostEqual(m.strength,.2)

    def test_attenuation(self):
        w=r.make_world(20100,200,"midpoint")
        x,y=w.body.x,w.body.y
        fx,fy=w.food[0]
        w.body.heading=math.atan2(fy-y,fx-x)
        self.assertEqual(w.odor(),(0.,0.,0.))
        self.assertGreater(max(r.attenuated_odor(w)),0.)
        atten=max(r.attenuated_odor(w))
        w.rectangles=[]
        self.assertAlmostEqual(atten,.25*max(r.attenuated_odor(w)))
        self.assertEqual(w.odor(),r.attenuated_odor(w))

    def test_matched_and_masked(self):
        a=r.episode(20100,300,"masked","original",8)
        b=r.episode(20100,300,"masked","memory",8)
        self.assertEqual(a["food"],b["food"])
        self.assertEqual(a["rectangles"],b["rectangles"])
        self.assertEqual(a["trace"],b["trace"])
        self.assertEqual(a["active"],0)
        self.assertEqual(b["active"],0)
        self.assertEqual(b,r.episode(20100,300,"masked","memory",8))

    def test_initial_visibility(self):
        w=r.make_world(20100,300,"offset")
        self.assertGreater(max(w.odor()),0.)
        self.assertEqual(len(w.rectangles),1)
        self.assertFalse(w.blocked(*w.food[0]))
