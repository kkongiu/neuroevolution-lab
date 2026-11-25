import copy
import unittest
from neuroevolution import flyloop_r12 as r
from neuroevolution import flyloop_r9 as r9

class TestReachability(unittest.TestCase):
    def test_original_method(self):
        self.assertIs(r.World.micro_integrate,r9.World.micro_integrate)
        w=r.World(12400,"obstacles")
        a,b=copy.deepcopy(w),copy.deepcopy(w)
        self.assertEqual(r.integrate(a,(.5,-.5),"original"),b.micro_integrate((.5,-.5)))
        self.assertEqual(r.snapshot_record(a),r.snapshot_record(b))
    def test_target_speed(self):
        for p in r.PLANTS[1:]:
            self.assertLess(r.target_speed(-1,p),0)
            self.assertEqual(r.target_speed(0,p),0)
            self.assertGreater(r.target_speed(1,p),0)
        for t in (.1,.5,1):
            self.assertEqual(r.target_speed(t,"original"),r.target_speed(t,"matched_forward"))
    def test_sequences(self):
        self.assertEqual(len(set(r.SEQUENCES)),81)
    def test_recovery(self):
        free={"collision":0,"moved":.01}
        hit={"collision":1,"moved":0.}
        self.assertEqual(r.recovery_step([hit]+[free]*4),5)
        self.assertIsNone(r.recovery_step([free]*3+[hit]+[free]*3))
        self.assertIsNone(r.recovery_step([{"collision":0,"moved":.001}]*4))
    def test_pair_and_death(self):
        w=r.World(12400,"nearfield")
        before=r.snapshot_record(w)
        for p in r.PLANTS:
            self.assertEqual(r.trial(w,p,r.SEQUENCES[0]),r.trial(w,p,r.SEQUENCES[0]))
            self.assertEqual(before,r.snapshot_record(w))
        w.body.energy=0
        for p in r.PLANTS:
            self.assertEqual(r.trial(w,p,r.SEQUENCES[0])["steps"],0)
    def test_source_determinism(self):
        self.assertEqual([r.snapshot_record(w) for w in r.source(12400,"nearfield")],
                         [r.snapshot_record(w) for w in r.source(12400,"nearfield")])
if __name__=="__main__":
    unittest.main()
