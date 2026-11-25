import copy
import unittest
from neuroevolution import flyloop_r11 as r11, flyloop_r12 as r12

class MotorTests(unittest.TestCase):
    def test_forward_exact(self):
        old = r11.World(12100,"nearfield","static")
        new = r12.World(12100,"nearfield")
        for cmd in r12.GRID * 2:
            self.assertEqual(old.micro_integrate(cmd),new.micro_integrate(cmd))

    def test_brake_and_reverse(self):
        self.assertEqual(r12.target_speed("brake",-1),0)
        self.assertEqual(r12.target_speed("reverse",0),0)
        self.assertLess(r12.target_speed("reverse",-1),0)
        w = r12.World(12100,"nearfield",plant="reverse")
        w.body.speed=0
        self.assertLess(w.micro_integrate((-1,0))["speed"],0)
        w = r12.World(12100,"nearfield",plant="brake")
        w.body.speed=.2
        result=w.micro_integrate((0,0))
        self.assertGreater(result["speed"],0)
        self.assertLess(result["speed"],.2)

    def test_copy_and_determinism(self):
        w=r12.World(12100,"nearfield")
        before=copy.deepcopy(w.body)
        a=r12.probe(w,"reverse",(-1,.5))
        self.assertEqual(a,r12.probe(w,"reverse",(-1,.5)))
        self.assertEqual(vars(w.body),vars(before))
        self.assertEqual(w.frame,0)
        self.assertEqual(len(a["steps"]),32)

    def test_bounds(self):
        w=r12.World(12100,"nearfield")
        for cmd in ((2,0),(0,-2),(float("nan"),0)):
            with self.assertRaises(ValueError):
                w.micro_integrate(cmd)

    def test_equal_steering_and_cost(self):
        worlds=[r12.World(12100,"nearfield",plant=p) for p in r12.PLANTS]
        results=[w.micro_integrate((0,.5)) for w in worlds]
        for key in ("heading","turn","energy"):
            self.assertEqual(len({r[key] for r in results}),1)
