"""M5.7: local learned forecast must affect actions, not future oracle."""
import random
import unittest

from neuroevolution.m5_6_prediction import LocalPredictor
from neuroevolution.m5_7_decision import choose_from_model, predicted_value
from neuroevolution.m5_7_screen import evaluate


class ModelDecisionTests(unittest.TestCase):
    def test_cannot_decide_before_learning(self):
        p=LocalPredictor()
        obs=(0.,)*19
        self.assertIsNone(choose_from_model(
            p,obs,(0,1,2),mode="model"))
        self.assertEqual(p.learned_transitions,0)
        self.assertEqual(p.probes,[])

    def test_trained_weights_really_change_hypothetical_choice(self):
        p=LocalPredictor()
        p.probes=[((0.,)*24,(0.,)*19)]*4
        p.learned_transitions=1
        # Training outcome of action 2 points toward food; this is a
        # unit-test fixture, NOT hardcoded world knowledge in agent.
        p.weights[14][19+2]=1.
        obs=(0.,)*19
        before=list(map(list,p.weights))
        chosen=choose_from_model(p,obs,(0,1,2,3),mode="model")
        self.assertEqual(chosen,2)
        self.assertEqual(p.weights,before)
        self.assertEqual(choose_from_model(p,obs,(0,1,2,3),
                         mode="model"),chosen)
        self.assertIsNone(choose_from_model(p,obs,(0,1),
                          mode="disabled"))

    def test_utility_is_a_transparent_survival_proxy(self):
        a=[0.]*19
        b=a[:]
        b[14]=.8
        self.assertGreater(predicted_value(tuple(b)),predicted_value(tuple(a)))
        b[15]=1
        self.assertLess(predicted_value(tuple(b)),predicted_value(tuple(a)))
        with self.assertRaises(ValueError):
            predicted_value((0.,)*18)

    def test_same_history_and_energy_conservation(self):
        r=evaluate((1080,))
        self.assertEqual(r["physical_trials"],3)
        self.assertTrue(r["engineered_utility_not_emergent"])
        self.assertTrue(all(abs(x["energy_residual"])<1e-6
                            for x in r["raw_rows"]))
        self.assertTrue(all(x["post_cutoff_weight_updates"]==0
                            for x in r["raw_rows"]))
        self.assertEqual(len({x["predictor_reward_total"]
                              for x in r["raw_rows"]}),1)

    def test_holdout_rejected(self):
        for seed in (1079,161000,162000,163000):
            with self.assertRaises(ValueError):
                evaluate((seed,))
        with self.assertRaises(ValueError):
            evaluate((1080,1080))


if __name__=="__main__":
    unittest.main()
