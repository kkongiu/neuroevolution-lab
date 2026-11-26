import unittest
from neuroevolution import flyloop_r19b as t

class TemporalTests(unittest.TestCase):
    def test_return(self):
        self.assertAlmostEqual(t.option_return([1.]*4,4),1.)
        self.assertLess(t.option_return([1.],4),1.)
    def test_evidence(self):
        e=t.Options()
        state=(1,1,1,0,0)
        self.assertIsNone(e.choose(state))
        for _ in range(8):
            e.add(state,(None,4),0.)
            e.add(state,((1.,0.),4),1.)
        self.assertEqual(e.choose(state),((1.,0.),4))
        for _ in range(8):
            e.add(state,(None,1),1.)
            e.add(state,((1.,0.),1),0.)
        self.assertTrue(e.abort(state,(1.,0.),1))
    def test_reproducibility(self):
        row=t.simulate(23100,"temporal",train_n=3,test_n=1,horizon=8)
        self.assertEqual(row,t.simulate(23100,"temporal",train_n=3,test_n=1,horizon=8))
        for p in t.r.SCENARIOS:
            for mode in ("temporal","no_interrupt"):
                self.assertEqual(row["tests"][mode][p][0]["start"],row["tests"]["sham"][p][0]["start"])
            self.assertEqual(row["tests"]["sham"][p][0]["interventions"],0)
        for ep in row["training"]:
            self.assertTrue(all(0<=o["elapsed"]<=o["option"][1] for o in ep["options"]))
