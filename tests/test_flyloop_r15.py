import unittest
from neuroevolution import flyloop_r15 as r
from neuroevolution import flyloop_r11 as r11

class TestR15(unittest.TestCase):
    def test_dataset(self):
        rec=[{"hidden":[.1,.2,.3,.4],"motor":[.5,-.5],"provenance":[1,2,3]},
             {"hidden":[.2,.3,.4,.5],"motor":[-.3,.7],"provenance":[2,3,4]}]
        safe=[{"motor":[.2,.1],"provenance":[1,4]}]
        true=r.dataset(rec,safe,"true");shuffled=r.dataset(rec,safe,"shuffled")
        self.assertEqual([x["hidden"] for x in true],[x["hidden"] for x in shuffled])
        self.assertEqual(true[0]["label"],shuffled[1]["label"])
        for v in r.VARIANTS[1:]:
            data=r.dataset(rec,safe,v)
            self.assertEqual(r.pretrain(r.SEEDS[0],data)[1],20)
            self.assertEqual(r.pretrain(r.SEEDS[0],data),r.pretrain(r.SEEDS[0],data))
        self.assertEqual(r.dataset(rec,[],"true"),[])
    def test_seed_separation(self):
        worlds=[r.source_seed(s,g,i) for s in r.SEEDS for g in r11.GEOMETRIES for i in range(8)]
        self.assertEqual(len(set(worlds)),384)
        self.assertFalse(set(worlds)&set(r.SEEDS))
    def test_source_target_and_audit(self):
        seed=r.SEEDS[0]
        m=r.source_model(r.source_seed(seed,"nearfield",0))
        source=m.simulate(seed,"nearfield","static","slow_only")
        self.assertTrue(all(m.audit([source]).values()))
        weights,_=r.pretrain(seed,[{"hidden":[.1,.2,.3,.4],"label":[-.5,.5]}])
        t=r.target_model(seed,weights,.5)
        row=t.simulate(seed,"nearfield","static","fast_fixed")
        self.assertTrue(all(t.audit([row]).values()))
        self.assertEqual(row["fast_10_output_coefficients_after_training"],weights)
        self.assertEqual(row["supervised_fast_student_updates"],0)
        self.assertEqual(r.encoder(row),r.encoder(source))
        self.assertNotEqual(row["initial_10_rectangles"],source["initial_10_rectangles"])
if __name__=="__main__":unittest.main()
