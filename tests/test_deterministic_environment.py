import unittest
from neuroevolution.deterministic_environment import make_plan,Inventory

class EnvironmentTests(unittest.TestCase):
    def test_replay(self):
        p=make_plan(42,1000,100,rectangles=((10,9,12,13),))
        q=make_plan(42,1000,100,rectangles=((10,9,12,13),))
        self.assertEqual(p,q)
        self.assertEqual(p.digest(),q.digest())
        self.assertNotEqual(p.digest(),make_plan(43,1000,100,rectangles=p.rectangles).digest())
        self.assertEqual([e.tick for e in p.events],list(range(0,1000,100)))
        for e in p.events:
            x,y=e.position
            self.assertFalse(9.45<=x<=12.55 and 8.45<=y<=13.55)
    def test_independent_consumption(self):
        p=make_plan(42,1000,100)
        a,b=Inventory(p),Inventory(p)
        for tick in range(1000):
            births=a.advance(tick)
            self.assertEqual(births,b.advance(tick))
            for e in births:
                self.assertEqual(a.consume(e.resource_id),e)
        self.assertEqual(len(a.available),0)
        self.assertEqual(len(b.available),len(p.events))
        self.assertEqual(p.digest(),make_plan(42,1000,100).digest())
    def test_invalid(self):
        with self.assertRaises(ValueError):make_plan(42,10,0)
        with self.assertRaises(ValueError):make_plan(42,10,rectangles=((0,0,22,22),))
        with self.assertRaises(ValueError):Inventory(make_plan(42,10)).advance(1)
