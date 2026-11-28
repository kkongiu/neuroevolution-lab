import unittest
from neuroevolution import flyloop_r25b as t,flyloop_r25a as old
class PriorityTests(unittest.TestCase):
    def test_fallback(self):
        o=(.9,.4,0.);z=(0.,0.,0.)
        for arm in ('hard','soft'):
            self.assertEqual(t.fuse(o,z,arm),(o,0.))
            v=(0.,1.,0.)
            self.assertEqual(t.fuse(o,v,arm),(v,1.))
    def test_partial(self):
        o=(.9,.4,0.);v=(0.,.5,0.)
        self.assertEqual(t.fuse(o,v,'hard'),(v,1.))
        self.assertEqual(t.fuse(o,v,'soft'),((.45,.45,0.),.5))
        for arm in ('hard','soft'):
            q,alpha=t.fuse(o,(0.,.1,0.),arm)
            self.assertTrue(0<=alpha<=1)
            self.assertTrue(all(0<=x<=1 for x in q))
    def test_regression(self):
        for c in ('odor','vision','combined'):
            for angle in (45,135):
                a=old.episode(35106,angle,1.5,.6,c,30)
                b=t.episode(35106,angle,1.5,.6,c,30)
                self.assertEqual(a['steps'],b['steps'])
                self.assertEqual(a['meal'],b['meal'])
                for x,y in zip(a['trace'],b['trace']):
                    for k in ('position','command','cue','heading','energy'):
                        self.assertEqual(x[k],y[k])
    def test_repeat(self):
        for arm in ('hard','soft'):
            a=t.episode(38100,135,1.5,.6,arm,30)
            self.assertEqual(a,t.episode(38100,135,1.5,.6,arm,30))
            self.assertTrue(0<=a['visual_weight']<=1)
