import math,unittest
from neuroevolution import flyloop_r24c as t
class AlignmentTests(unittest.TestCase):
    def test_geometry(self):
        for angle in t.ANGLES:
            x=t.episode(35106,angle,'ground',.6,'odor',5)
            self.assertAlmostEqual(math.dist(x['source'],x['start'][:2]),3.)
            self.assertTrue(x['first_second_complete'])
            self.assertEqual(x['steps'],20)
    def test_masked(self):
        a=t.episode(35106,0,'ground',.6,'masked',5)
        b=t.episode(35106,135,'ground',.6,'masked',5)
        self.assertEqual(a['initial_cue']['peak'],0.)
        for x,y in zip(a['trace'],b['trace']):
            self.assertEqual(x['position'],y['position'])
            self.assertEqual(x['command'],y['command'])
    def test_repeat(self):
        for plant in t.PLANTS:
            a=t.episode(36100,90,plant,.6,'odor',15)
            self.assertEqual(a,t.episode(36100,90,plant,.6,'odor',15))
            self.assertGreater(a['initial_cue']['peak'],0.)
