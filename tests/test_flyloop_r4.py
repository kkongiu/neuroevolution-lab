"""Prospective R4 tests: actual motor command vs zero vs older yoke."""
import copy
import unittest

from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r2 as r2
from neuroevolution import flyloop_r4 as r4


class FlyLoopR4Tests(unittest.TestCase):
    def test_preregistered_independent_seeds_336_lives_84_coefficients(self):
        self.assertEqual(r4.SEEDS,tuple(range(9950,9974)))
        self.assertTrue(set(r4.SEEDS).isdisjoint(r2.SEEDS))
        self.assertEqual(r4.SCENES,("obstacles","nearfield"))
        self.assertEqual(len(r4.ARMS),7)
        self.assertEqual(len(r4.SEEDS)*2*7,336)
        self.assertEqual(r4.FEATURES,28)
        self.assertEqual(r4.HEAD_PARAMS,84)
        self.assertEqual(r1.PARAMS,434)
        self.assertEqual(r4.YOKED_LAG,6)
        with self.assertRaises(ValueError):
            r4.simulate(9700,"obstacles","agency_gain")
        with self.assertRaises(ValueError):
            r4.simulate(9950,"food","agency_gain")
        with self.assertRaises(ValueError):
            r4.simulate(9950,"obstacles","follow_wall")

    def test_three_head_features_isolate_actual_vs_blank_vs_past_motor(self):
        h=r4.ActionSensorHeads(9950)
        neurons=[.25]*r1.N
        rays=[.5,.3,.9]
        command=(.8,-.2)
        phi,old,src=h.inputs(0,neurons,rays,command)
        self.assertIsNone(src)
        self.assertEqual(old,(0.,0.))
        self.assertEqual(len(phi["true"]),28)
        self.assertEqual(phi["true"][19:21],list(command))
        self.assertEqual(phi["blind"][19:21],[0.,0.])
        self.assertEqual(phi["blind"][21:27],[0.]*6)
        self.assertEqual(phi["true"][21:27],
                         [x*y for x in rays for y in command])
        for name in r4.PREDICTOR_NAMES:
            self.assertEqual(h.predict(name,rays,phi[name]),rays)
        for t in range(8):
            h.motor_history.append((t/20.,-t/25.))
        phi,yoked,src=h.inputs(8,neurons,rays,command)
        self.assertIsNotNone(src)
        self.assertLessEqual(src,2)
        self.assertEqual(yoked,h.motor_history[src])
        self.assertNotEqual(phi["true"],phi["blind"])

    def test_sham_command_never_future_last_or_fabricated(self):
        heads=r4.ActionSensorHeads(9951)
        for tick in range(30):
            p,command,source=heads.inputs(
                tick,[0.]*16,[0.]*3,(.15,-.3)
            )
            self.assertTrue(
                source is None or 0<=source<=tick-r4.YOKED_LAG
            )
            if tick<r4.YOKED_LAG:
                self.assertIsNone(source)
                self.assertEqual(command,(0.,0.))
            else:
                self.assertIsNotNone(source)
                self.assertEqual(command,heads.motor_history[source])
            heads.motor_history.append((tick/100.,-tick/90.))

    def test_observation_updates_head_after_physical_target_only(self):
        h=r4.ActionSensorHeads(9952)
        rays=[.2,.4,.6]
        phi,_,_=h.inputs(0,[.1]*16,rays,(.5,-.3))
        pred={
            name:h.predict(name,rays,phi[name])
            for name in r4.PREDICTOR_NAMES
        }
        self.assertTrue(all(pred[n]==rays for n in r4.PREDICTOR_NAMES))
        w=copy.deepcopy(h.weights)
        h.train_all(phi,pred,[.3,.2,.9])
        self.assertNotEqual(h.weights,w)
        self.assertEqual(h.updates,{"true":1,"blind":1,"yoked":1})
        self.assertEqual(h.weights["blind"][0][19:27],[0.]*8)
        self.assertEqual(len(h.weights["true"]),3)

    def test_same_world_paired_across_seven_controls_and_physical_audit(self):
        for scene in r4.SCENES:
            lives=[r4.simulate(9950,scene,arm) for arm in r4.ARMS]
            self.assertEqual(len({r4.sha(r["rectangles"]) for r in lives}),1)
            self.assertEqual(len({r4.sha(r["initial_pose"]) for r in lives}),1)
            self.assertEqual(len({
                r["initial_coefficients_sha256"] for r in lives
            }),1)
            for r in lives:
                report=r4._audit([r])
                self.assertTrue(all(report.values()),(scene,r["arm"],report))
                self.assertEqual(r["steps_alive"],len(r["trace"]))
                for field in ("w_input","w_recurrent","bias"):
                    self.assertEqual(
                        r["initial_coefficients"][field],
                        r["final_coefficients"][field]
                    )
                if r["arm"] in (
                    "recurrent_frozen","passive_probes","inertial_random"
                ):
                    self.assertEqual(
                        r["initial_coefficients"],
                        r["final_coefficients"],
                    )

    def test_passive_probes_and_frozen_have_identical_body_trajectories(self):
        for scene in r4.SCENES:
            frozen=r4.simulate(9951,scene,"recurrent_frozen")
            passive=r4.simulate(9951,scene,"passive_probes")
            self.assertEqual(frozen["steps_alive"],passive["steps_alive"])
            for a,b in zip(frozen["trace"],passive["trace"]):
                for key in (
                    "position","executed_motor_command",
                    "pre_wall_rays","post_wall_rays",
                    "injected_motor_noise","hidden_neural_state",
                ):
                    self.assertEqual(a[key],b[key])
                self.assertEqual(
                    a["prediction_true"],a["pre_wall_rays"]
                )
                self.assertEqual(
                    a["prediction_blind"],a["pre_wall_rays"]
                )

    def test_deterministic_complete_lifetime_all_arms(self):
        for arm in r4.ARMS:
            a=r4.simulate(9952,"nearfield",arm)
            b=r4.simulate(9952,"nearfield",arm)
            self.assertEqual(r4.sha(a),r4.sha(b))
            self.assertEqual(a["steps_alive"],len(a["trace"]))
            self.assertTrue(0<a["steps_alive"]<=600)
            self.assertTrue(all(r4._audit([a]).values()))


if __name__=="__main__":
    unittest.main()
