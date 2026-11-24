"""FlyLoop R3: prospective causal predictor and physical motor tests."""
import copy
import math
import unittest

from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r2 as r2
from neuroevolution import flyloop_r3 as r3


class FlyLoopR3Tests(unittest.TestCase):
    def test_locked_design_16_neurons_66_predictor_384_lives(self):
        self.assertEqual(r3.SEEDS,tuple(range(9700,9724)))
        self.assertEqual(r3.SCENES,("obstacles","nearfield"))
        self.assertEqual(len(r3.ARMS),8)
        self.assertEqual(len(r3.SEEDS)*2*len(r3.ARMS),384)
        self.assertTrue(set(r3.SEEDS).isdisjoint(r2.SEEDS))
        self.assertEqual(r3.HEAD_FEATURES,22)
        self.assertEqual(r3.HEAD_PARAMETERS,66)
        self.assertEqual(r1.N,16)
        self.assertEqual(r1.PARAMS,434)
        self.assertEqual(r3.LATE_START,300)
        with self.assertRaises(ValueError):
            r3.simulate(9400,"nearfield","predict_motor")
        with self.assertRaises(ValueError):
            r3.simulate(9700,"food","predict_motor")
        with self.assertRaises(ValueError):
            r3.simulate(9700,"nearfield","wall_following")

    def test_frozen_predictor_is_same_sample_persistence_baseline(self):
        head=r3.SensorHead(9700)
        ray=(.08,.42,.95)
        features=head.features(
            (0.2,)*r1.N,ray,(.5,-.1)
        )
        self.assertEqual(len(features),22)
        self.assertEqual(head.predict(ray,features),list(ray))
        true_target=(.15,.61,.83)
        target,source=head.training_target(
            0,true_target,"predict_passive"
        )
        self.assertEqual(source,0)
        self.assertEqual(target,true_target)
        self.assertTrue(head.train(features,head.predict(ray,features),target))
        self.assertEqual(head.update_events,1)
        self.assertNotEqual(head.predict(ray,features),list(ray))
        self.assertEqual(len(head.weights),3)
        self.assertTrue(all(len(row)==22 for row in head.weights))

    def test_sham_label_is_own_observation_six_steps_older(self):
        head=r3.SensorHead(9701)
        initial=head.sham_rng.getstate()
        for tick in range(18):
            obs=[tick/30.,(tick+1)/30.,(tick+2)/30.]
            target,source=head.training_target(
                tick,obs,"predict_sham_passive"
            )
            if tick<6:
                self.assertIsNone(source)
                self.assertIsNone(target)
            else:
                self.assertTrue(0<=source<=tick-6)
                self.assertEqual(target,head.observed_next[source])
                self.assertNotEqual(target,obs)
        self.assertNotEqual(initial,head.sham_rng.getstate())

    def test_true_and_sham_passive_follow_same_real_physical_trajectory(self):
        for scene in r3.SCENES:
            a=r3.simulate(9700,scene,"predict_passive")
            b=r3.simulate(9700,scene,"predict_sham_passive")
            self.assertEqual(a["steps_alive"],b["steps_alive"])
            self.assertEqual(a["initial_coefficients_sha256"],
                             b["initial_coefficients_sha256"])
            self.assertEqual(a["final_coefficients_sha256"],
                             b["final_coefficients_sha256"])
            self.assertEqual(a["initial_coefficients"],a["final_coefficients"])
            for x,y in zip(a["trace"],b["trace"]):
                for key in (
                    "position","executed_motor_command",
                    "raw_sensor_inputs","hidden_neural_state",
                    "observed_next_wall_rays","injected_motor_noise",
                    "collision","energy","integrity",
                ):
                    self.assertEqual(x[key],y[key])
            self.assertNotEqual(a["head_final_weights"],
                                b["head_final_weights"])
            self.assertEqual(
                a["head_training_events"],a["steps_alive"]
            )
            self.assertEqual(
                b["head_training_events"],max(0,b["steps_alive"]-6)
            )

    def test_head_precedes_real_world_step_and_predictions_match_later(self):
        life=r3.simulate(9701,"nearfield","predict_motor")
        self.assertGreater(life["steps_alive"],0)
        for i,f in enumerate(life["trace"]):
            self.assertEqual(
                f["current_wall_rays"],
                f["raw_sensor_inputs"][1:4]
            )
            self.assertAlmostEqual(
                r3.mse(f["predicted_next_wall_rays"],
                       f["observed_next_wall_rays"]),
                f["true_prediction_mse"],places=8
            )
            if i+1<life["steps_alive"]:
                self.assertEqual(
                    f["observed_next_wall_rays"],
                    life["trace"][i+1]["current_wall_rays"],
                )
            self.assertTrue(all(0<=x<=1 for x in
                                f["predicted_next_wall_rays"]))
        self.assertEqual(life["head_training_events"],life["steps_alive"])
        self.assertEqual(life["motor_training_events"],life["steps_alive"])

    def test_fixed_recurrent_coefficient_substrate_and_legal_output_updates(self):
        for arm in r3.ARMS:
            life=r3.simulate(9702,"nearfield",arm)
            for key in ("w_input","w_recurrent","bias"):
                self.assertEqual(life["initial_coefficients"][key],
                                 life["final_coefficients"][key])
            if arm in (
                "recurrent_frozen","predict_passive",
                "predict_sham_passive","inertial_random",
            ):
                self.assertEqual(
                    life["initial_coefficients"],
                    life["final_coefficients"],
                )
            else:
                self.assertEqual(
                    life["motor_training_events"],life["steps_alive"]
                )
            integrity=r3._audit([life])
            self.assertTrue(all(integrity.values()),(arm,integrity))

    def test_same_initial_body_world_and_motor_random_perturbations(self):
        lives=[
            r3.simulate(9703,"obstacles",arm)
            for arm in r3.ARMS
        ]
        self.assertEqual(len({r3.sha(l["rectangles"]) for l in lives}),1)
        self.assertEqual(len({r3.sha(l["initial_pose"]) for l in lives}),1)
        self.assertEqual(len({
            l["initial_coefficients_sha256"] for l in lives
        }),1)
        neural=[l for l in lives if l["arm"]!="inertial_random"]
        minimum=min(l["steps_alive"] for l in neural)
        first=[s["injected_motor_noise"] for s in neural[0]["trace"][:minimum]]
        self.assertTrue(all(
            [s["injected_motor_noise"] for s in l["trace"][:minimum]]==first
            for l in neural
        ))

    def test_reproducible_life_original_R1_physics_not_altered(self):
        for scene in r3.SCENES:
            for arm in ("r2_progress","predict_motor","predict_sham",
                        "predict_frozen"):
                a=r3.simulate(9704,scene,arm)
                b=r3.simulate(9704,scene,arm)
                self.assertEqual(r3.sha(a),r3.sha(b))
                self.assertGreater(a["distance"],0)
                self.assertEqual(a["steps_alive"],len(a["trace"]))
                self.assertLessEqual(a["steps_alive"],600)
                self.assertTrue(all(r3._audit([a]).values()))


if __name__=="__main__":
    unittest.main()
