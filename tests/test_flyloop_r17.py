import unittest
from neuroevolution import flyloop_r17 as r
from neuroevolution import flyloop_r11 as r11

class TestR17(unittest.TestCase):
    def test_general_split(self):
        x=list(range(1000))
        g,h=r.select_general(x)
        self.assertEqual(len(g),128)
        self.assertEqual(len(h),200)
        self.assertFalse(set(g)&set(h))
        self.assertEqual(len(set(g)),128)
    def test_curriculum(self):
        g=[{"hidden":[.1,.2,.3,.4],"motor":[.6,.2]}]*4
        e=[{"hidden":[.3,.2,.1,0.],"motor":[-.5,.7]}]*2
        f={v:r.train(r.SEEDS[0],g,e,g,v) for v in r.VARIANTS}
        self.assertEqual(f["fixed"]["updates"],0)
        self.assertEqual(f["general"]["updates"],40)
        self.assertEqual(f["recovery"]["updates"],20)
        self.assertEqual(f["curriculum"]["updates"],60)
        self.assertEqual(f["general_matched"]["updates"],60)
        self.assertEqual(f["curriculum_replay"]["updates"],80)
        self.assertEqual(len({f[v]["general_prefix"] for v in ("general","curriculum","general_matched","curriculum_replay")}),1)
        self.assertEqual(r.train(r.SEEDS[0],g,e,g,"curriculum"),f["curriculum"])
    def test_source_target(self):
        s=r.SEEDS[0]
        m=r.source_model(r.source_seed(s,"nearfield",0))
        row=m.simulate(s,"nearfield","static","slow_only")
        self.assertTrue(all(m.audit([row]).values()))
        w=r.train(s,[{"hidden":[.1,.2,.3,.4],"motor":[.5,.6]}],[],[],"general")["weights"]
        t=r.target_model(s,w,.5)
        x=t.simulate(s,"nearfield","static","fast_fixed")
        self.assertTrue(all(t.audit([x]).values()))
        self.assertEqual(x["fast_10_output_coefficients_after_training"],w)
        self.assertEqual(x["supervised_fast_student_updates"],0)
        self.assertNotEqual(row["initial_10_rectangles"],x["initial_10_rectangles"])
if __name__=="__main__":unittest.main()
