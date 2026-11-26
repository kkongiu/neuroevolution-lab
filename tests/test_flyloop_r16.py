import unittest
from neuroevolution import flyloop_r16 as r, flyloop_r15 as r15, flyloop_r14 as m, flyloop_r13 as core

class VetoTests(unittest.TestCase):
    def test_fallback(self):
        obs=[0.]*35
        obs[-3:]=[.5,.7,.1]
        base=(.2,-.3)
        cmd,_=r15.innate_command(base,obs[-3:],.6)
        self.assertEqual(r.choose(m.Critic(),r.Evidence(),[0.]*16,obs,base,.6,"gated"),(cmd,False))

    def test_evidence_threshold(self):
        e=r.Evidence()
        state=(0,1,0,1)
        for _ in range(7):
            e.add(state,(1.,0.),.5)
        self.assertIsNone(e.bounds(state,(1.,0.)))
        e.add(state,(1.,0.),.5)
        self.assertEqual(e.bounds(state,(1.,0.)),(.5,.5))

    def test_intervention_requires_both(self):
        obs=[0.]*35
        hidden=[0.]*16
        base=(0.,0.)
        state=r.state_bin(obs)
        e=r.Evidence()
        for _ in range(8):
            e.add(state,base,0.)
            e.add(state,(1.,0.),1.)
        c=m.Critic()
        self.assertEqual(r.choose(c,e,hidden,obs,base,.6,"gated"),(base,False))
        c.weights[0][1]=.5
        self.assertEqual(r.choose(c,e,hidden,obs,base,.6,"gated"),((1.,0.),True))

    def test_obstacle(self):
        w=r.make_world(18100,300,True,True)
        self.assertEqual(len(w.rectangles),1)
        self.assertFalse(w.blocked(w.body.x,w.body.y))
        self.assertFalse(w.blocked(*w.food[0]))
        self.assertEqual(w.rectangles,r.make_world(18100,300,True,True).rectangles)

    def test_sham(self):
        a=r.simulate(18100,"gated",train_n=2,test_n=1)
        b=r.simulate(18100,"sham",train_n=2,test_n=1)
        self.assertEqual(a["training"],b["training"])
        self.assertEqual(a["weights"],b["weights"])
        self.assertEqual(a["evidence"],b["evidence"])
        for phase in ("near","far","masked","obstacle"):
            self.assertEqual(a[phase][0]["food"],b[phase][0]["food"])
            self.assertEqual(b[phase][0]["interventions"],0)
        self.assertEqual(a,r.simulate(18100,"gated",train_n=2,test_n=1))
