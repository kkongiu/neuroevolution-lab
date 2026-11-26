import unittest
import math
from neuroevolution import flyloop_r1 as r1, flyloop_r9 as r9, flyloop_r12 as r12, flyloop_r12c as c

class BodySensesTests(unittest.TestCase):
    def world(self):
        w=c.World(12100,"nearfield")
        w.rectangles=[]
        w.body.heading=0.
        return w

    def test_sector_faces(self):
        box=[(3.,1.,4.,2.)]
        for start,end,heading,index in [
            ((2.6,1.5),(2.8,1.5),0.,0),
            ((2.6,1.5),(2.8,1.5),math.pi,4),
            ((3.5,.6),(3.5,.8),0.,2),
            ((3.5,2.4),(3.5,2.2),0.,6)]:
            s=c.contact_sectors(start,end,heading,box)
            self.assertEqual(sum(s),1)
            self.assertEqual(s[index],1)
        s=c.contact_sectors((2.6,.6),(2.8,.8),0.,box)
        self.assertEqual((s[0],s[2],sum(s)),(1,1,2))

    def test_real_collision_and_release(self):
        w=self.world()
        w.body.x=r1.SIDE-r1.RADIUS-.001
        w.body.speed=r1.V_MAX
        before=w.body.integrity
        self.assertEqual(w.micro_integrate((1,0))["collision"],1)
        self.assertEqual(w.touch[0],1)
        self.assertEqual(w.duration[0],1)
        self.assertEqual(w.actual_speed,0)
        self.assertEqual(w.blocked_fraction,1)
        self.assertAlmostEqual(w.injury,1)
        self.assertLess(w.body.integrity,before)
        w.micro_integrate((1,0))
        self.assertEqual(w.duration[0],2)
        for _ in range(32):
            w.micro_integrate((-1,0))
        self.assertEqual(sum(w.touch),0)
        self.assertEqual(sum(w.duration),0)
        self.assertLess(w.actual_speed,0)
        self.assertEqual(w.injury,0)

    def test_proximity_not_touch(self):
        w=self.world()
        w.body.x=r1.RADIUS+.1
        self.assertGreater(w.extra_senses()[0],.9)
        self.assertEqual(sum(w.touch),0)
        w.micro_integrate((0,0))
        self.assertEqual(sum(w.touch),0)
        self.assertEqual(w.injury,0)

    def test_physics_exact(self):
        for plant in r12.PLANTS:
            a=r12.World(12100,"nearfield",plant=plant)
            b=c.World(12100,"nearfield",plant=plant)
            for command in r12.GRID*4:
                self.assertEqual(a.micro_integrate(command),b.micro_integrate(command))
                self.assertEqual(a.perceive(),b.perceive())
                self.assertEqual(len(b.perceive_extended()),c.SLOW_INPUTS)
                self.assertEqual(len(b.perceive_fast_extended()),c.FAST_INPUTS)

    def test_zero_afferents(self):
        w=self.world()
        a=r1.Circuit(42,"recurrent")
        b=c.SlowCircuit(42)
        f=r9.FastCircuit(42)
        g=c.FastCircuit(42)
        for command in r12.GRID:
            w.micro_integrate(command)
            self.assertEqual(a.forward(w.perceive())[:2],b.forward(w.perceive_extended())[:2])
            self.assertEqual(f.observe(r9.local_fast_sensors(w.perceive())),g.observe(w.perceive_fast_extended()))
