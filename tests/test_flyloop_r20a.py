import unittest
from neuroevolution import flyloop_r20a as t

class MultisensoryTests(unittest.TestCase):
    def test_free_monotonic(self):
        w=t.r.WindowWorld(26100,0,"free")
        x,y=w.lamp
        self.assertGreater(t.odor_at(w,x-1,y,True),t.odor_at(w,x-3,y,True))
        self.assertEqual(t.odor_at(w,x-9,y,True),0.)
    def test_wall(self):
        w=t.r.WindowWorld(26100,0,"sealed")
        self.assertEqual(t.odor_at(w,w.body.x,w.body.y,True),0.)
        w=t.r.WindowWorld(26100,0,"glass")
        self.assertGreater(t.odor_at(w,w.body.x,w.body.y,True),0.)
        self.assertEqual(t.odor_at(w,w.wall_x,0.,True),0.)
    def test_cues(self):
        l=(.4,0.,.1); o=(.2,.3,0.)
        self.assertEqual(t.combine_cues(l,o,"light_only"),l)
        self.assertEqual(t.combine_cues(l,o,"odor_only"),o)
        self.assertEqual(t.combine_cues(l,o,"combined_routed"),(.4,.3,.1))
    def test_baseline_and_repeat(self):
        ep=t.episode(26100,0,"glass","light_only",12)
        old=t.d.episode(26100,0,"glass","affordance",12)
        self.assertEqual([f["command"] for f in ep["trace"]],[f["command"] for f in old["trace"]])
        self.assertEqual(ep,t.episode(26100,0,"glass","light_only",12))
        for arm in t.ARMS:
            row=t.episode(26100,0,"sealed",arm,12)
            self.assertFalse(row["crossed"])
            self.assertTrue(all(not any(f["odor_samples"]) for f in row["trace"]))
