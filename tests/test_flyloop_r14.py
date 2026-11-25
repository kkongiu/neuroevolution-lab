import unittest
from neuroevolution import flyloop_r14 as r
from neuroevolution import flyloop_r13 as r13

class TestR14(unittest.TestCase):
    def test_blend(self):
        self.assertEqual(r.blend((1.,-1.),(-1.,1.),0.),(1.,-1.))
        self.assertEqual(r.blend((1.,-1.),(-1.,1.),1.),(-1.,1.))
        self.assertEqual(r.blend((1.,-1.),(-1.,1.),.5),(0.,0.))
        with self.assertRaises(ValueError):r.blend((0.,0.),(1.,1.),1.1)
    def test_unchanged_full_authority(self):
        m=r13.isolated("original");m.SEEDS=r.SEEDS
        n=r.model_for(1.)
        a=m.simulate(r.SEEDS[0],"nearfield","static","fast_recovery_true")
        b=n.simulate(r.SEEDS[0],"nearfield","static","fast_recovery_true")
        self.assertEqual(a,b)
    def test_paired_training(self):
        rows=[]
        for h in r.AUTH:
            m=r.model_for(h)
            row=m.simulate(r.SEEDS[0],"nearfield","static","fast_recovery_true")
            self.assertTrue(all(m.audit([row]).values()))
            rows.append(row)
        self.assertEqual(len({r13.trace_signature(x) for x in rows}),1)
        self.assertTrue(all(x["actual_recovery_training_lessons"]==rows[0]["actual_recovery_training_lessons"] for x in rows))
if __name__=="__main__":
    unittest.main()
