import unittest
import copy
from neuroevolution import flyloop_r19 as r
from neuroevolution import flyloop_r15 as p

class TestR19(unittest.TestCase):
    def test_feedback_and_freeze(self):
        a=r.Delegator(r.SEEDS[0],"learned")
        phi=(1.,)+(0.,)*12
        self.assertEqual(a.observe(1,phi,.5,True),(0,.5))
        self.assertEqual(a.weights[1][0],.015)
        self.assertEqual(a.weights[0],[0.]*13)
        before=copy.deepcopy(a.weights)
        self.assertEqual(a.observe(1,phi,-1.,False),(None,None))
        self.assertEqual(a.weights,before)
    def test_delay(self):
        a=r.Delegator(r.SEEDS[0],"delayed");phi=(1.,)+(0.,)*12
        for i in range(8):self.assertEqual(a.observe(0,phi,i/10,True),(None,None))
        self.assertEqual(a.observe(1,phi,.8,True),(0,0.))
        self.assertEqual(a.observe(1,phi,.9,True),(1,.1))
        self.assertEqual(a.updates,2)
    def test_blend(self):
        self.assertEqual(r.blend((1.,-1.),(-1.,1.),0.),(1.,-1.))
        self.assertEqual(r.blend((1.,-1.),(-1.,1.),.5),(0.,0.))
        self.assertEqual(r.blend((1.,-1.),(-1.,1.),1.),(-1.,1.))
    def test_physical_and_pairing(self):
        s=r.SEEDS[0];w,_=p.pretrain(s,[{"hidden":[.1,.2,.3,.4],"label":[.3,.4]}])
        rows=[r.simulate(s,"nearfield",a,w) for a in r.ARMS]
        for row in rows:
            self.assertTrue(r.audit(row))
            self.assertEqual(row["fast_final"],w)
        traces=[[f for f in row["trace"] if f["macro"]<r.TRAIN] for row in rows]
        self.assertTrue(all(t==traces[0] for t in traces))
        bad=copy.deepcopy(rows[0]);bad["trace"][0]["motor"]=(.99,.99)
        self.assertFalse(r.audit(bad))
    def test_disjoint_seeds(self):
        worlds={r.source_seed(s,g,i) for s in r.SEEDS for g in ("obstacles","nearfield") for i in range(16)}
        self.assertEqual(len(worlds),768)
        self.assertFalse(worlds&set(r.SEEDS))
if __name__=="__main__":unittest.main()
