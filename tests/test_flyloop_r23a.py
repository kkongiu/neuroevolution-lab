import unittest
from neuroevolution import flyloop_r23a as t

class RenewableTests(unittest.TestCase):
    def test_schedule(self):
        for arm,interval in [('renew10',200),('renew5',100)]:
            p=t.schedule((3.,4.),arm,1000)
            self.assertEqual([e[0] for e in p],list(range(0,4000,interval)))
            self.assertEqual(t.digest(p),t.digest(t.schedule((3.,4.),arm,1000)))
        self.assertEqual(len(t.schedule((3.,4.),'finite',1000)),1)
    def test_finite_parent(self):
        for s in (29100,29101):
            a=t.episode(s,s,0,'finite',400)
            b=t.parent.episode(s,0,90,'adaptive_hot',400)
            for k in ('steps','survived','final_energy','thermal_damage'):
                self.assertAlmostEqual(a[k],b[k])
            self.assertEqual(bool(a['meals']),b['meal'] is not None)
    def test_genome_independence(self):
        for arm in t.ARMS:
            a=t.episode(35100,45100,0,arm,100)
            b=t.episode(35101,45100,0,arm,100)
            self.assertEqual(a['plan_hash'],b['plan_hash'])
            self.assertEqual(a['start'],b['start'])
            self.assertEqual(a,t.episode(35100,45100,0,arm,100))
            self.assertEqual(a['births'],a['remaining']+len(a['meals']))
