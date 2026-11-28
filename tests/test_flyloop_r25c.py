import unittest
from neuroevolution import flyloop_r25c as t,flyloop_r24b as old
class AquariumTests(unittest.TestCase):
    def test_baseline(self):
        a=old.episode(35106,45100,0,'ground',100)
        b=t.episode(35106,45100,0,'odor',100)
        for k in ('steps','final_energy','thermal_damage','distance','rotation','reverse_seconds','flips','departures','returns','plan_hash','births','remaining'):
            self.assertEqual(a[k],b[k],k)
        self.assertEqual(a['meals'],[{k:m[k] for k in ('tick','energy','gain')} for m in b['meals']])
        self.assertEqual(len(a['trace']),len(b['trace']))
        for x,y in zip(a['trace'],b['trace']):
            self.assertEqual(x,{k:y[k] for k in x})
    def test_repeat_and_supply(self):
        for arm in t.ARMS:
            a=t.episode(39100,45100,1,arm,60)
            self.assertEqual(a,t.episode(39100,45100,1,arm,60))
            self.assertEqual(a['births'],a['remaining']+len(a['meals']))
            self.assertLessEqual(a['peak_lateral'],.8+1e-9)
            self.assertLessEqual(a['post_meal_returns'],len(a['returns']))
