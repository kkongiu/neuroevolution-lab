import unittest
from neuroevolution import flyloop_r13 as r
from neuroevolution import flyloop_r11 as original
from neuroevolution import flyloop_r9 as r9

class TestR13(unittest.TestCase):
    def test_isolation(self):
        a,b=r.isolated("original"),r.isolated("signed")
        self.assertEqual(a.SEEDS,r.SEEDS)
        self.assertNotEqual(original.SEEDS,r.SEEDS)
        self.assertIs(a.World.micro_integrate,r9.World.micro_integrate)
        self.assertIsNot(a.World,b.World)
    def test_signed_schema_and_motion(self):
        a,b=r.isolated("original"),r.isolated("signed")
        wa,wb=a.World(r.SEEDS[0],"nearfield","static"),b.World(r.SEEDS[0],"nearfield","static")
        self.assertEqual(wa.initial_rectangles,wb.initial_rectangles)
        self.assertEqual(set(wa.micro_integrate((-1.,0.))),set(wb.micro_integrate((-1.,0.))))
        self.assertLess(wb.body.speed,0)
        self.assertGreater(wa.body.speed,0)
        self.assertEqual(wa.body.energy,wb.body.energy)
    def test_paired_training_and_audit(self):
        for p in r.PLANTS:
            m=r.isolated(p)
            rows=[m.simulate(r.SEEDS[0],"nearfield","static",a) for a in r.ARMS]
            self.assertEqual(len({r.trace_signature(x) for x in rows}),1)
            self.assertTrue(all(m.audit(rows).values()))
    def test_bootstrap_deterministic(self):
        self.assertEqual(r.bootstrap([1,0,-1,2]),r.bootstrap([1,0,-1,2]))
if __name__=="__main__":
    unittest.main()
