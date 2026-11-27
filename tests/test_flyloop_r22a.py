import unittest
from neuroevolution import flyloop_r22a as t
class EcologyTests(unittest.TestCase):
    def test_repeat(self):
        for scene in t.SCENARIOS:
            for arm in t.ARMS:
                a=t.episode(34100,0,scene,arm,20)
                self.assertEqual(a,t.episode(34100,0,scene,arm,20))
                self.assertEqual(len(a['causes']),2 if arm=='one' else 3)
                self.assertLessEqual(sum(a['gains'][:-1]),12*len(a['initial_food']))
    def test_absent(self):
        a=t.episode(34100,0,'lantern','two_no_predator',50)
        self.assertEqual(a['causes'][-1],'absent')
        self.assertEqual(a['captures'],[])
    def test_persistence(self):
        a=t.episode(34100,0,'lantern','two',100)
        for step,i in a['captures']:
            self.assertLessEqual(step,a['steps'])
            self.assertEqual(a['causes'][i],'capture')
        self.assertEqual(len(a['captures']),len(set(i for _,i in a['captures'])))
        self.assertEqual(a['trace'][-1]['step'],a['steps'])
