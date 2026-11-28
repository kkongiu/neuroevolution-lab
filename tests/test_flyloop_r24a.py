import unittest,math
from neuroevolution import flyloop_r24a as t
class PhysicsTests(unittest.TestCase):
    def test_rest(self):
        for m in t.MODELS:
            x=t.run_case(m,'rest')
            self.assertEqual(x['distance'],0)
            self.assertEqual(x['rotation'],0)
    def test_mirror(self):
        for m in t.MODELS:
            a=t.run_case(m,'turn_1.0_1');b=t.run_case(m,'turn_1.0_-1')
            self.assertAlmostEqual(a['final']['x'],b['final']['x'])
            self.assertAlmostEqual(a['final']['y'],-b['final']['y'])
    def test_bounds(self):
        for m in ('ground','air'):
            for c in t.CASES:
                x=t.run_case(m,c)
                self.assertLessEqual(x['peak_lateral'],.8+1e-9)
                self.assertLessEqual(x['peak_torque'],.30)
                self.assertLessEqual(x['peak_force'],1.8+1e-9)
        self.assertEqual(t.run_case('air','reverse')['reverse_seconds'],0)
        self.assertGreater(t.run_case('ground','reverse')['reverse_seconds'],0)
    def test_convergence_and_replay(self):
        a=t.run()
        self.assertEqual(a,t.run())
        for x in a['convergence']:
            self.assertLess(x['position_error'],.06)
            self.assertLess(x['heading_error'],.06)
