import unittest
from neuroevolution import flyloop_r23b as t

class MotorTests(unittest.TestCase):
    def test_baseline(self):
        for s in (35100,35106):
            a=t.episode(s,45100,0,'baseline',400)
            b=t.prev.episode(s,45100,0,'renew10',400)
            for k in ('steps','survived','final_energy','thermal_damage','meals','plan_hash'):
                self.assertEqual(a[k],b[k])
    def test_coupling(self):
        self.assertEqual(t.coordinate((1.,0.)),(1.,0.))
        self.assertEqual(t.coordinate((1.,1.)),(.25,1.))
        self.assertEqual(t.coordinate((-1.,-1.)),(-.25,-1.))
    def test_inhibition(self):
        eye=t.r.CompoundEye()
        for cue in ((0.,)*9,(0.,)*8+(1.,),(1.,)+(0.,)*8,(1.,)*9):
            for base in ((1.,-1.),(-1.,1.)):
                a,_=t.motor(base,(0.,)*9,cue,eye,'baseline')
                b,_=t.motor(base,(0.,)*9,cue,eye,'inhibition')
                self.assertEqual(a[0],b[0])
                self.assertTrue(all(-1<=x<=1 for x in b))
    def test_repeat(self):
        for arm in t.ARMS:
            a=t.episode(35106,45100,0,arm,30)
            self.assertEqual(a,t.episode(35106,45100,0,arm,30))
