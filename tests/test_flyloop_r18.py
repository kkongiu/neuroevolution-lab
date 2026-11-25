import unittest
from neuroevolution import flyloop_r18 as r
from neuroevolution import flyloop_r15 as p
class TestR18(unittest.TestCase):
    def test_nested(self):
        s=r.SEEDS[0]
        xs=[{"slot":i,"world_seed":r.source_seed(s,g,i),"recovery":[(g,i)],"safe":[]} for g in ("obstacles","nearfield") for i in range(16)]
        a,_,wa=r.pool(xs,16);b,_,wb=r.pool(xs,32)
        self.assertEqual(len(a),16);self.assertEqual(len(b),32)
        self.assertTrue(set(a)<set(b));self.assertTrue(set(wa)<set(wb))
    def test_seeds(self):
        x={r.source_seed(s,g,i) for s in r.SEEDS for g in ("obstacles","nearfield") for i in range(16)}
        self.assertEqual(len(x),1024);self.assertFalse(x&set(r.SEEDS))
    def test_bootstrap(self):
        a=[{"postchange_actual_meters":1,"postchange_collision_microframes":2} for _ in r.SEEDS]
        b=[{"postchange_actual_meters":1,"postchange_collision_microframes":4} for _ in r.SEEDS]
        self.assertEqual(r.paired_bootstrap(a,b)["bootstrap95"],[-2,-2])
        a[0]["postchange_actual_meters"]=0
        self.assertEqual(r.ratio(a),64/31)
        self.assertEqual(r.paired_bootstrap(a,b),r.paired_bootstrap(a,b))
    def test_source_target(self):
        s=r.SEEDS[0];m=r.source_model(r.source_seed(s,"nearfield",0))
        a=m.simulate(s,"nearfield","static","slow_only")
        self.assertTrue(all(m.audit([a]).values()))
        w,_=p.pretrain(s,[{"hidden":[.1,.2,.3,.4],"label":[.3,.4]}])
        t=r.target_model(s,w);b=t.simulate(s,"nearfield","static","fast_fixed")
        self.assertTrue(all(t.audit([b]).values()))
        self.assertEqual(p.encoder(a),p.encoder(b))
        self.assertEqual(b["fast_10_output_coefficients_after_training"],w)
        self.assertNotEqual(a["initial_10_rectangles"],b["initial_10_rectangles"])
if __name__=="__main__":unittest.main()
