import unittest
from neuroevolution import flyloop_r19c as t

class InnateMotivationTests(unittest.TestCase):
    def test_no_conflict(self):
        obs=[0.]*50
        baseline=(.5,.1)
        self.assertFalse(t.eligible(obs))
        self.assertEqual(t.choose(None,None,None,obs,baseline,.5),(baseline,False))
        obs[-14]=.4
        self.assertFalse(t.eligible(obs))
        obs[-5]=.2
        self.assertTrue(t.eligible(obs))
        self.assertEqual(t.choose(None,None,None,obs,baseline,.5,True),(baseline,False))
    def test_trace_invariants(self):
        row=t.simulate(24100,"restricted",train_n=3,test_n=1,horizon=12)
        for ep in row["training"]+[ep for phases in row["tests"].values() for eps in phases.values() for ep in eps]:
            for frame in ep["trace"]:
                self.assertEqual(frame["command"][0],frame["baseline"][0])
                if not frame["conflict"]:
                    self.assertEqual(frame["command"],frame["baseline"])
        self.assertEqual(row,t.simulate(24100,"restricted",train_n=3,test_n=1,horizon=12))
    def test_free_identical(self):
        row=t.simulate(24100,"restricted",train_n=3,test_n=1,horizon=12)
        self.assertEqual(row["tests"]["restricted"]["free"][0]["trace"],row["tests"]["sham"]["free"][0]["trace"])
