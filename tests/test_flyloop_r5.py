"""R5 physically executed action interventions and causal provenance."""
import math
import unittest

from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r2 as r2
from neuroevolution import flyloop_r4 as r4
from neuroevolution import flyloop_r5 as r5


class R5Tests(unittest.TestCase):

    def test_registered_preselected_dimensions(self):
        self.assertEqual(r5.SEEDS,tuple(range(10300,10324)))
        self.assertEqual(r5.SCENES,("obstacles","nearfield"))
        self.assertEqual(r5.TRAIN_STEPS,300)
        self.assertEqual(r5.HORIZON,600)
        self.assertEqual(r5.PROBE_STEPS,tuple(range(300,600,10)))
        self.assertEqual(r5.LONG_STEPS,6)
        self.assertEqual(r5.SENSOR_EFFECT_THRESHOLD,.005)
        self.assertEqual(r5.LONG_EFFECT_THRESHOLD,.05)
        self.assertEqual(len(r5.SEEDS)*len(r5.SCENES),48)
        self.assertEqual(48*30*5,7200)
        self.assertEqual(48*30*4,5760)
        self.assertEqual(r4.HEAD_PARAMS,84)
        self.assertEqual(r1.N,16)
        self.assertTrue(set(r5.SEEDS).isdisjoint(r2.SEEDS))
        with self.assertRaises(ValueError):
            r5.World(9950,"obstacles")
        with self.assertRaises(ValueError):
            r5.World(10300,"food")

    def test_exact_five_experimentally_forced_commands(self):
        policy=(.4,-.11)
        v=r5.commands(policy)
        self.assertEqual(list(v),list(r5.ACTIONS))
        self.assertEqual(v["natural"],policy)
        self.assertEqual(v["steer_left"],(.4,-.85))
        self.assertEqual(v["steer_right"],(.4,.85))
        self.assertEqual(v["throttle_low"],(.05,-.11))
        self.assertEqual(v["throttle_high"],(.95,-.11))
        self.assertEqual(len(set(v.values())),5)
        self.assertTrue(all(
            -1<=c<=1 for cmd in v.values() for c in cmd
        ))

    def test_cloned_world_branch_never_mutates_real_world_or_predictor(self):
        w=r5.World(10300,"obstacles")
        brain=r2.MotorLearner(10300,"recurrent_frozen")
        heads=r4.ActionSensorHeads(10300)
        s=w.perceive()
        _,h,_,_,natural=brain.command(s)
        original_body=vars(w.body).copy()
        original_frame=w.frame
        original_head=r5.sha(heads.weights)
        before_rectangles=r5.sha(w.rectangles)
        branch=r5.physical_branches(
            w,h,s[1:4],natural,heads,(0.,0.),None,300
        )
        self.assertEqual(vars(w.body),original_body)
        self.assertEqual(w.frame,original_frame)
        self.assertEqual(r5.sha(w.rectangles),before_rectangles)
        self.assertEqual(r5.sha(heads.weights),original_head)
        self.assertEqual(len(branch["trials"]),5)
        self.assertEqual(len(branch["contrasts"]),4)
        real_step=w.integrate(natural)
        self.assertEqual(branch["trials"][0]["step1_physics"],real_step)
        self.assertEqual(branch["trials"][0]["real_step1_next_wall_rays"],
                         list(w.perceive()[1:4]))
        for t in branch["trials"]:
            self.assertEqual(t["physical_pre_body"],original_body)
            self.assertTrue(all(
                0<=v<=1 for v in t["real_step1_next_wall_rays"]
            ))
            self.assertTrue(all(
                key in t["model_predictions"] for key in
                ("true","blind","yoked","persistence")
            ))
        self.assertGreaterEqual(branch["six_step_left_right_xy_distance"],0.)

    def test_frozen_original_nervous_system_and_heads_only_train_real_300(self):
        life=r5.simulate(10301,"nearfield")
        self.assertEqual(life["original_life_steps"],600)
        self.assertEqual(life["probe_count"],30)
        self.assertEqual(life["real_intervention_one_step_trials"],150)
        self.assertEqual(life["actual_pair_contrast_count"],120)
        self.assertFalse(life["original_neural_coefficient_changed"])
        self.assertEqual(
            life["training_updates_per_head"],
            {"true":300,"blind":300,"yoked":300}
        )
        self.assertEqual(
            life["frozen_three_heads_after_300_real_steps"],
            life["final_three_heads"]
        )
        self.assertTrue(all(
            t["head_was_trained"]==(t["tick"]<300)
            for t in life["real_original_life"]
        ))
        for p in life["probe_snapshots_and_real_branches"]:
            self.assertEqual(p["frozen_3_head_weights_sha256"],
                             r5.sha(life["final_three_heads"]))
            self.assertEqual(
                p["preaction_16_neurons"],
                life["real_original_life"][p["tick"]]["real_16_neurons"],
            )
        self.assertTrue(all(r5._audit([life]).values()))

    def test_yoke_is_real_past_action_with_no_future_from_first_step(self):
        life=r5.simulate(10302,"obstacles")
        original=life["real_original_life"]
        for t in original:
            tick=t["tick"]
            source=t["past_yoked_source_tick"]
            if tick<6:
                self.assertIsNone(source)
                self.assertEqual(t["past_yoked_motor"],[0.,0.])
            else:
                self.assertTrue(0<=source<=tick-6)
                self.assertEqual(
                    t["past_yoked_motor"],
                    original[source]["natural_executed_motor"],
                )
        self.assertTrue(all(r5._audit([life]).values()))

    def test_one_step_sensor_deltas_are_observed_not_predicted(self):
        life=r5.simulate(10303,"nearfield")
        for probe in life["probe_snapshots_and_real_branches"]:
            base=probe["trials"][0]
            for trial,contrast in zip(
                probe["trials"][1:],probe["contrasts"]
            ):
                actual=[
                    v-u for v,u in zip(
                        trial["real_step1_next_wall_rays"],
                        base["real_step1_next_wall_rays"],
                    )
                ]
                predicted=[
                    v-u for v,u in zip(
                        trial["model_predictions"]["true"],
                        base["model_predictions"]["true"],
                    )
                ]
                self.assertEqual(
                    contrast["actual_sensor_change"],actual
                )
                self.assertEqual(
                    contrast["predicted_true_sensor_change"],predicted
                )
                self.assertAlmostEqual(
                    contrast["true_change_prediction_mse"],
                    r5.squared(actual,predicted),places=12
                )
                self.assertAlmostEqual(
                    contrast["zero_change_baseline_mse"],
                    r5.squared(actual,(0.,0.,0.)),places=12
                )

    def test_worlds_and_identical_replay(self):
        a=r5.simulate(10304,"obstacles")
        b=r5.simulate(10304,"obstacles")
        self.assertEqual(r5.sha(a),r5.sha(b))
        near=r5.simulate(10304,"nearfield")
        self.assertNotEqual(a["original_world_rectangles"],
                            near["original_world_rectangles"])
        self.assertEqual(a["initial_original_neural_coefficients_sha256"],
                         near["initial_original_neural_coefficients_sha256"])
        self.assertTrue(all(r5._audit([a,near]).values()))


if __name__=="__main__":
    unittest.main()
