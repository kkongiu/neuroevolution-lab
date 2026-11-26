import unittest
from neuroevolution import flyloop_r13b as b
from neuroevolution import flyloop_r13 as r

class AblationTests(unittest.TestCase):
    def test_masks(self):
        w=r.World(14100,"nearfield")
        w.touch=(1.,)*8
        w.duration=(8,)*8
        w.actual_speed=-.5
        w.motor_target=.5
        w.blocked_fraction=.7
        w.injury=.3
        base=b.observe(w,[],"full")
        for arm,indices in b.MASKS.items():
            observed=b.observe(w,[],arm)
            self.assertEqual(observed[:8],base[:8])
            self.assertEqual(observed[8:],tuple(0. if i in indices else x for i,x in enumerate(base[8:])))
        h=[]
        for _ in range(20):
            self.assertFalse(any(b.observe(w,h,"delayed")[8:]))
        self.assertEqual(b.observe(w,h,"delayed")[8:],base[8:])

    def test_paired_split(self):
        a,c=b.simulate(14100,"nearfield","full",train=24,test=8)
        self.assertEqual(a["training"],c["training"])
        self.assertEqual(a["weights"],c["weights"])
        self.assertEqual(a["split_pose"],c["split_pose"])
        self.assertEqual(c["test_rectangles"],a["test_rectangles"][::2])
        for row in (a,c):
            self.assertEqual(row["test"]["updates"],0)
            self.assertEqual(len(row["trace"]),32)
        self.assertEqual([a,c],b.simulate(14100,"nearfield","full",train=24,test=8))

    def test_straight_path(self):
        phase=r.Phase()
        phase.distance=2.
        frames=[dict(heading=0.,position=[1.,0.],actual_speed=1.),
                dict(heading=0.,position=[2.,0.],actual_speed=1.)]
        m=b.metrics(frames,(0.,0.),0.,phase)
        self.assertEqual(m["net_path_ratio"],1.)
        self.assertEqual(m["turning"],0.)
        self.assertEqual(m["reverse_fraction"],0.)
