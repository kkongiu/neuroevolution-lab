import unittest
from neuroevolution import flyloop_r14b as b, flyloop_r14 as r

class CreditTests(unittest.TestCase):
    def test_returns(self):
        self.assertEqual(b.returns([0.,0.,12.,0.]),[.81,.9,1.,0.])
        self.assertEqual(b.returns([0.]*8+[12.])[0],0.)
    def test_episode(self):
        c=r.Critic()
        x=b.episode(16100,100,"delayed",c,False,True,8)
        self.assertEqual(x,b.episode(16100,100,"delayed",c,False,True,8))
        self.assertEqual(x["food"],b.episode(16100,100,"masked",c,False,True,8)["food"])
        self.assertTrue(all(not any(w) for w in c.weights))
        self.assertAlmostEqual(60.-x["cost"]+x["gain"],x["energy"])
    def test_training(self):
        c=r.Critic()
        b.episode(16100,0,"delayed",c,True,False,8)
        self.assertTrue(any(c.weights[0]))
