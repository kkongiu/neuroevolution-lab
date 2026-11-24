"""R7 regression checks: within-life causal online model-action cycle."""
import unittest
from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r2 as r2
from neuroevolution import flyloop_r4 as r4
from neuroevolution import flyloop_r7 as r7


class R7Tests(unittest.TestCase):
    def test_frozen_dimensions_seeds_and_six_controls(self):
        self.assertEqual(r7.SEEDS,tuple(range(10900,10916)))
        self.assertEqual(r7.GEOMETRIES,("obstacles","nearfield"))
        self.assertEqual(r7.REGIMES,("static","removal"))
        self.assertEqual(len(r7.ARMS),6)
        self.assertEqual(16*2*2*6,384)
        self.assertEqual(r7.HORIZON,480)
        self.assertEqual(r7.CHANGE_TICK,240)
        self.assertEqual(r7.WARMUP,60)
        self.assertEqual(r1.N,16)
        self.assertEqual(r1.PARAMS,434)
        self.assertEqual(r4.HEAD_PARAMS,84)
        self.assertTrue(set(r7.SEEDS).isdisjoint(r2.SEEDS))
        with self.assertRaises(ValueError):
            r7.World(10600,"nearfield","removal")
        with self.assertRaises(ValueError):
            r7.World(10900,"food","removal")
        with self.assertRaises(ValueError):
            r7.World(10900,"nearfield","teleport")
        with self.assertRaises(ValueError):
            r7.simulate(10900,"nearfield","removal","escape_reflex")

    def test_exact_five_continuous_local_candidates_no_global_input(self):
        base=(.4,-.1)
        choices=r7.candidates(base)
        self.assertEqual(len(choices),5)
        self.assertEqual(choices[0],base)
        self.assertEqual(choices[1][0],.4)
        self.assertAlmostEqual(choices[1][1],-.45)
        self.assertAlmostEqual(choices[2][1],.25)
        self.assertEqual(choices[3],(.2,-.1))
        self.assertAlmostEqual(choices[4][0],.6)
        self.assertTrue(all(
            -1<=v<=1 for choice in choices for v in choice
        ))
        saturated=r7.candidates((1.,1.))
        self.assertEqual(saturated[2],(1.,1.))
        self.assertEqual(saturated[4],(1.,1.))
        head=r4.ActionSensorHeads(10900)
        predictions=r7.predicted_candidates(
            head,[.1]*16,[.3,.5,.7],choices
        )
        self.assertEqual(
            r7.selected_candidate(predictions["true"])[0],0
        )
        self.assertEqual(predictions["true"][0],[.3,.5,.7])

    def test_exogenous_midlife_removal_and_no_entrapment(self):
        for geom in r7.GEOMETRIES:
            static=r7.World(10900,geom,"static")
            removed=r7.World(10900,geom,"removal")
            self.assertEqual(static.rectangles,removed.rectangles)
            self.assertEqual(len(removed.rectangles),10)
            removed.apply_environment(239)
            self.assertEqual(len(removed.rectangles),10)
            removed.apply_environment(240)
            self.assertTrue(removed.switched)
            self.assertEqual(len(removed.rectangles),5)
            self.assertEqual(
                removed.rectangles,list(static.rectangles[::2])
            )
            self.assertFalse(removed.blocked(removed.body.x,removed.body.y))
            static.apply_environment(240)
            self.assertFalse(static.switched)
            self.assertEqual(len(static.rectangles),10)

    def test_passive_learns_but_cannot_change_motor_trajectory(self):
        for geometry in r7.GEOMETRIES:
            frozen=r7.simulate(10900,geometry,"removal","recurrent_frozen")
            passive=r7.simulate(10900,geometry,"removal","passive_predictor")
            self.assertEqual(frozen["real_physical_steps"],
                             passive["real_physical_steps"])
            for a,b in zip(frozen["raw_trace"],passive["raw_trace"]):
                for key in (
                    "physical_body_outcome","physically_executed_motor",
                    "actual_next_three_wall_sensors",
                    "original_16_recurrent_states",
                    "original_exogenous_motor_noise",
                ):
                    self.assertEqual(a[key],b[key])
            self.assertEqual(
                frozen["head_update_events"],
                {"true":0,"blind":0,"yoked":0},
            )
            self.assertEqual(
                passive["head_update_events"],
                {h:passive["real_physical_steps"] for h in r7.HEADS},
            )
            self.assertEqual(
                frozen["original_434_neural_coefficients_at_birth"],
                frozen["original_434_neural_coefficients_at_end"],
            )
            self.assertEqual(
                passive["original_434_neural_coefficients_at_birth"],
                passive["original_434_neural_coefficients_at_end"],
            )

    def test_world_regimes_identical_through_239_for_all_arm_types(self):
        for arm in r7.ARMS:
            static=r7.simulate(10901,"nearfield","static",arm)
            removed=r7.simulate(10901,"nearfield","removal",arm)
            for x,y in zip(
                static["raw_trace"][:r7.CHANGE_TICK],
                removed["raw_trace"][:r7.CHANGE_TICK],
            ):
                self.assertEqual(x,y)
            self.assertFalse(static["environment_removed_five_walls"])
            self.assertTrue(removed["environment_removed_five_walls"])
            self.assertEqual(len(static["postchange_rectangles"]),10)
            self.assertEqual(len(removed["postchange_rectangles"]),5)

    def test_online_head_predicts_prior_to_actual_motor_and_strict_old_yoke(self):
        for arm in ("true_closed_loop","yoked_closed_loop","random_menu"):
            life=r7.simulate(10902,"obstacles","removal",arm)
            self.assertGreater(life["real_physical_steps"],240)
            self.assertEqual(life["head_update_events"],{
                name:life["real_physical_steps"] for name in r7.HEADS
            })
            self.assertEqual(
                life["initial_brain_sha256"],life["final_brain_sha256"]
            )
            for frame in life["raw_trace"]:
                t=frame["tick"]
                idx=frame["selected_candidate_index"]
                cmd=frame["five_local_motor_candidates"][idx]
                self.assertEqual(cmd,frame["physically_executed_motor"])
                if t<r7.WARMUP and arm!="r2_progress":
                    self.assertEqual(idx,0)
                src=frame["historic_yoked_source_tick"]
                if src is None:
                    self.assertLess(t,6)
                    self.assertEqual(frame["historic_yoked_motor"],[0.,0.])
                else:
                    self.assertTrue(0<=src<=t-6)
                    self.assertEqual(
                        frame["historic_yoked_motor"],
                        life["raw_trace"][src]["physically_executed_motor"],
                    )
                self.assertEqual(
                    len(frame["preaction_candidate_model_predictions"]["true"]),5
                )
                if t>=r7.WARMUP and arm in r7.MODEL_SCORER_ARMS:
                    head=("true" if arm=="true_closed_loop" else "yoked")
                    self.assertEqual(
                        idx,r7.selected_candidate(
                            frame["preaction_candidate_model_predictions"][head]
                        )[0]
                    )

    def test_real_lives_and_reproducible_raw_physical_audit(self):
        cases=[
            ("obstacles","static","true_closed_loop"),
            ("nearfield","removal","yoked_closed_loop"),
            ("nearfield","removal","r2_progress"),
            ("nearfield","static","random_menu"),
        ]
        rows=[]
        for geometry,regime,arm in cases:
            first=r7.simulate(10903,geometry,regime,arm)
            second=r7.simulate(10903,geometry,regime,arm)
            self.assertEqual(r7.sha(first),r7.sha(second))
            self.assertEqual(first["real_physical_steps"],
                             len(first["raw_trace"]))
            self.assertTrue(first["real_physical_steps"]<=480)
            rows.append(first)
        report=r7._audit(rows)
        self.assertTrue(all(report.values()),report)


if __name__=="__main__":
    unittest.main()
