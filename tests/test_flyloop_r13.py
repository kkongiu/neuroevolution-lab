import unittest
from neuroevolution import flyloop_r13 as r

class BodyLearningTests(unittest.TestCase):
    def test_critic_updates(self):
        c=r.Critic()
        phi=r.features([0.]*16,[0.]*32,(.5,-.5))
        before=c.predict(phi)
        c.learn(phi,(1.,0.))
        after=c.predict(phi)
        self.assertGreater(after[0],before[0])
        self.assertLess(after[1],before[1])

    def test_ablation_and_freeze(self):
        results={a:r.simulate(13100,"nearfield",a,train=24,test=8) for a in r.ARMS}
        for arm,row in results.items():
            self.assertEqual(row["training"]["updates"],0 if arm=="fixed" else 24)
            self.assertEqual(row["test"]["updates"],0)
            self.assertEqual(row["test"]["microsteps"],32)
            self.assertTrue(all(len(x["observed"])==32 for x in row["trace"]))
        self.assertTrue(all(not any(x["observed"][8:]) for x in results["original"]["trace"]))
        self.assertTrue(all(not any(x["observed"][8:]) for x in results["delayed"]["trace"][:20]))
        for i in range(0,24,4):
            self.assertEqual(len({tuple(row["trace"][i]["command"]) for row in results.values()}),1)
        self.assertEqual(results["full"]["weights"],r.simulate(13100,"nearfield","full",train=24,test=0)["weights"])
        self.assertEqual(results["full"],r.simulate(13100,"nearfield","full",train=24,test=8))

    def test_recovery(self):
        p=r.Phase()
        p.frames=[(1,0.)]+[(0,.01)]*32
        out=p.report()
        self.assertEqual(out["opportunities"],1)
        self.assertEqual(out["recoveries"],1)
        p.frames=[(1,0.)]*33
        self.assertEqual(p.report()["recoveries"],0)
