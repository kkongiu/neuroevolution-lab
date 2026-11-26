import unittest
from neuroevolution import flyloop_r16b as b

class CurriculumTests(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual([sum(b.obstacle_training(c,i) for i in range(80)) for c in b.CURRICULA],[0,40,80])
        with self.assertRaises(ValueError):
            b.obstacle_training("invalid",0)

    def test_matched_evaluation(self):
        row=b.simulate(19100,"mixed",train_n=2,test_n=1,horizon=8)
        self.assertEqual([e["obstacle_training"] for e in row["training"]],[False,True])
        self.assertEqual(len(row["training"][0]["rectangles"]),0)
        self.assertEqual(len(row["training"][1]["rectangles"]),1)
        for phase in b.PHASES:
            g=row["evaluations"]["gated"][phase][0]
            s=row["evaluations"]["sham"][phase][0]
            self.assertEqual(g["food"],s["food"])
            self.assertEqual(g["rectangles"],s["rectangles"])
            self.assertEqual(s["interventions"],0)
        self.assertEqual(row["evaluations"]["gated"]["obstacle"][0]["food"],
                         row["evaluations"]["gated"]["masked"][0]["food"])
        self.assertEqual(row,b.simulate(19100,"mixed",train_n=2,test_n=1,horizon=8))

    def test_summary(self):
        rows=[b.simulate(19100,c,train_n=2,test_n=1,horizon=8) for c in b.CURRICULA]
        s=b.summarize(rows)
        for c,n in zip(b.CURRICULA,(0,1,2)):
            self.assertEqual(s[c]["obstacle_training_episodes"],n)
            self.assertEqual(s[c]["gated"]["obstacle"]["episodes"],1)
            self.assertEqual(s[c]["sham"]["obstacle"]["intervention_fraction"],0.)
