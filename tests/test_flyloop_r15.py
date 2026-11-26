import unittest
from neuroevolution import flyloop_r15 as r
from neuroevolution import flyloop_r14 as m

class AttractorTests(unittest.TestCase):
    def test_no_stimulus_or_hunger(self):
        for odor,hunger in (((0.,0.,0.),1.),((1.,1.,0.),0.)):
            self.assertEqual(r.innate_command((.2,-.3),odor,hunger),((.2,-.3),0.))
    def test_symmetry(self):
        l,b=r.innate_command((0.,0.),(.5,.8,.1),.6)
        rr,c=r.innate_command((0.,0.),(.5,.1,.8),.6)
        self.assertAlmostEqual(l[0],rr[0])
        self.assertAlmostEqual(l[1],-rr[1])
        self.assertGreater(l[1],0.)
        self.assertEqual(b,c)
        self.assertTrue(all(-1<=x<=1 for x in l+rr))
    def test_matched_counterfactual(self):
        c=m.Critic()
        a=r.episode(17100,200,"both",c,False,True,False,8)
        b=r.episode(17100,200,"both",c,False,True,True,8)
        self.assertEqual(a["food"],b["food"])
        self.assertTrue(all(not any(w) for w in c.weights))
        self.assertTrue(all(f["odor"]==(0.,0.,0.) for f in b["trace"]))
        self.assertEqual(a,r.episode(17100,200,"both",c,False,True,False,8))
    def test_no_learning(self):
        row=r.simulate(17100,"innate",train_n=1,test_n=1)
        self.assertTrue(all(not any(w) for w in row["weights"]))
    def test_learning(self):
        c=m.Critic()
        r.episode(17100,0,"both",c,True,horizon=8)
        self.assertTrue(any(c.weights[0]))
