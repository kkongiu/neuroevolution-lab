import unittest
from neuroevolution import flyloop_r23a as old, flyloop_r24b as new
class GroundForagingTests(unittest.TestCase):
    def test_legacy(self):
        a=old.episode(35106,45100,0,'renew10',100)
        b=new.episode(35106,45100,0,'legacy',100)
        for k in ('steps','final_energy','thermal_damage','meals','returns','departures','plan_hash','survived'):
            self.assertEqual(a[k],b[k],k)
    def test_ground(self):
        a=new.episode(35106,45100,0,'ground',100)
        self.assertEqual(a,new.episode(35106,45100,0,'ground',100))
        self.assertLessEqual(a['peak_lateral'],.8+1e-9)
        self.assertEqual(a['births'],a['remaining']+len(a['meals']))
        self.assertGreater(a['distance'],0)
