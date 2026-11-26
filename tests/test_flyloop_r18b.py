import unittest
from neuroevolution import flyloop_r18b as r, flyloop_r16 as r16

class LearnedVisionTests(unittest.TestCase):
    def test_pool(self):
        self.assertEqual(r.pool(tuple(range(9))),(3,3,3,3,4,8,8,8,8))
    def test_fallback(self):
        baseline=(.2,-.3)
        self.assertEqual(r.choose(r.Critic(),r16.Evidence(),[0.]*16,[0.]*50,baseline,.6,"gated"),(baseline,False))
    def test_evidence_gate(self):
        c=r.Critic()
        e=r16.Evidence()
        obs=[0.]*50
        state=r.state_bin(obs)
        for _ in range(8):
            e.add(state,(0.,0.),0.)
            e.add(state,(1.,0.),1.)
        self.assertEqual(r.choose(c,e,[0.]*16,obs,(0.,0.),.6,"gated"),((0.,0.),False))
        c.weights[0][1]=.5
        self.assertEqual(r.choose(c,e,[0.]*16,obs,(0.,0.),.6,"gated"),((1.,0.),True))
    def test_frozen_matched(self):
        row=r.simulate(22100,"full",train_n=3,test_n=1,horizon=8)
        self.assertTrue(any(row["weights"][0]))
        for p in r.SCENARIOS:
            g=row["tests"]["gated"][p][0]
            s=row["tests"]["sham"][p][0]
            for key in ("start","lamp","rectangles","opening"):
                self.assertEqual(g[key],s[key])
            self.assertEqual(s["interventions"],0)
            self.assertTrue(all(len(t["delivered_vision"])==18 for t in g["trace"]))
        self.assertEqual(row,r.simulate(22100,"full",train_n=3,test_n=1,horizon=8))
    def test_pooled_observation(self):
        row=r.episode(22100,0,"glass","pooled",r.Critic(),r16.Evidence(),horizon=8)
        for t in row["trace"]:
            self.assertEqual(t["delivered_vision"],r.pool(t["proximity"])+r.pool(t["light"]))
