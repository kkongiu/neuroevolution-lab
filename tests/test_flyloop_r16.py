import unittest
from neuroevolution import flyloop_r16 as r
from neuroevolution import flyloop_r15 as p
class TestR16(unittest.TestCase):
    def test_seed_isolation(self):
        a,_=r.make_model("early");b,_=r.make_model("aligned")
        self.assertEqual(a.SEEDS,b.SEEDS)
        self.assertNotEqual(p.SEEDS,a.SEEDS)
        self.assertEqual(len({r.source_seed(s,g,i) for s in r.SEEDS for g in ("obstacles","nearfield") for i in range(8)}),384)
    def test_aligned_source(self):
        a,lags=r.make_model("aligned")
        s=r.SEEDS[0]
        found=False
        for i in range(8):
            ws=r.source_seed(s,"nearfield",i)
            m=a.source_model(ws)
            row=m.simulate(s,"nearfield","static","slow_only")
            self.assertTrue(all(m.audit([row]).values()))
            for e in row["real_successful_recovery_episodes"]:
                k=[ws,e["collision_micro"],e["first_recovery_motor_micro"]]
                rec=[{"hidden":e["first_post_contact_fast_hidden"],"motor":e["actual_first_recovery_motor"],"provenance":k}]
                ds=a.dataset(rec,[{"motor":[0.,0.],"provenance":[ws,0]}],"true")
                self.assertEqual(ds[0]["hidden"],row["raw_micro_trace"][k[2]]["four_fast_recurrent_states"])
                self.assertEqual(ds[0]["label"],rec[0]["motor"])
                self.assertEqual(ds[0]["input_micro_tick"],k[2])
                found=True
        self.assertEqual(found,bool(lags))
if __name__=="__main__":unittest.main()
