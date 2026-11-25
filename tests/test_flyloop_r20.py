import unittest
from neuroevolution import flyloop_r20 as r
from neuroevolution import flyloop_r15 as p

class TestR20(unittest.TestCase):
    def step(self,v,gate,t,training=False):
        return v.step(gate,t//4,t%4,t,(1.,)*8,(0.,0.),(0.,0.),training)
    def test_reflex_and_veto(self):
        v=r.Veto(14800,"veto_episode")
        v.weights[0]=[1.]*13
        self.assertEqual(self.step(v,True,3)[:2],(1.,True))
        self.assertEqual(self.step(v,True,4)[0],0.)
        self.assertEqual(self.step(v,True,5)[0],0.)
        self.assertEqual(self.step(v,False,6)[0],0.)
        self.assertEqual(self.step(v,True,7)[:2],(1.,True))
    def test_release(self):
        v=r.Veto(14800,"veto_episode")
        v.weights[0]=[1.]*13
        self.step(v,True,0)
        self.assertEqual(self.step(v,True,4)[0],0.)
        v.weights[1]=[2.]*13
        self.assertEqual(self.step(v,True,8)[0],1.)
    def test_episode_credit(self):
        v=r.Veto(14800,"veto_episode")
        self.step(v,True,0,True)
        self.step(v,True,4,True)
        for t in range(4,8):v.observe(t//4,t%4,t,{"moved":0.,"collision":1},True,True)
        self.assertEqual(v.updates,0)
        self.step(v,False,8,True)
        self.assertEqual(v.updates,1)
        self.assertEqual(v.events[0]["frames"],4)
        self.assertAlmostEqual(v.events[0]["observed_target"],-1.2)
    def test_physical(self):
        s=r.SEEDS[0];w,_=p.pretrain(s,[{"hidden":[.1,.2,.3,.4],"label":[.3,.4]}])
        rows=[r.simulate(s,"nearfield",a,w) for a in r.ARMS]
        for row in rows:
            self.assertTrue(r.audit(row))
            self.assertEqual(row["fast_final"],w)
        traces=[[f for f in row["trace"] if f["macro"]<r.TRAIN] for row in rows]
        self.assertTrue(all(t==traces[0] for t in traces))
if __name__=="__main__":unittest.main()
