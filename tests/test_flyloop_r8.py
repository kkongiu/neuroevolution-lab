"""R8: real collision, displacement and integrity consequences during one life."""
import copy
import unittest

from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r2 as r2
from neuroevolution import flyloop_r4 as r4
from neuroevolution import flyloop_r7 as r7
from neuroevolution import flyloop_r8 as r8


class R8Tests(unittest.TestCase):
    def test_protocol_dimensions_and_fresh_fixed_seeds(self):
        self.assertEqual(r8.SEEDS,tuple(range(11200,11216)))
        self.assertTrue(set(r8.SEEDS).isdisjoint(r7.SEEDS))
        self.assertEqual(r8.GEOMETRIES,("obstacles","nearfield"))
        self.assertEqual(r8.REGIMES,("static","removal"))
        self.assertEqual(len(r8.ARMS),6)
        self.assertEqual(16*2*2*6,384)
        self.assertEqual(r8.HORIZON,320)
        self.assertEqual(r8.CHANGE_TICK,160)
        self.assertEqual(r8.WARMUP,50)
        self.assertEqual(r8.HEAD_FEATURES,28)
        self.assertEqual(r1.N,16)
        self.assertEqual(r1.INPUTS,8)
        self.assertEqual(r1.PARAMS,434)
        self.assertEqual(r4.HEAD_PARAMS,84)
        self.assertEqual(r8.START_OFFSETS,(.25,.5,.25))
        self.assertEqual(r8.SHAM_LAG,6)
        with self.assertRaises(ValueError):
            r8.World(10900,"nearfield","removal")
        with self.assertRaises(ValueError):
            r8.World(11200,"nofood","removal")
        with self.assertRaises(ValueError):
            r8.simulate(11200,"nearfield","removal","flee")

    def test_physical_targets_come_only_from_actually_executed_physics(self):
        world=r8.World(11200,"nearfield","static")
        initial_integrity=world.body.integrity
        outcome=world.integrate((.4,.4))
        targets=r8.PhysicalConsequenceHeads.real_target(
            outcome,initial_integrity
        )
        self.assertEqual(len(targets),3)
        self.assertEqual(targets[0],outcome["collision"])
        self.assertAlmostEqual(
            targets[1],outcome["moved"]/(r1.V_MAX*r1.DT),places=7
        )
        self.assertEqual(targets[2],targets[0])
        self.assertTrue(all(0<=t<=1 for t in targets))

    def test_sham_training_reads_real_outcome_of_same_life_at_least_six_old(self):
        h=r8.PhysicalConsequenceHeads(11200)
        phi=r4.ActionSensorHeads.make_features(
            [0.]*16,[.1,.2,.3],(.5,-.2)
        )
        self.assertEqual(len(phi),28)
        first={
            name:h.predict_features(name,phi) for name in r8.HEAD_NAMES
        }
        self.assertEqual(first["true"],[.25,.5,.25])
        self.assertEqual(first["older_sham"],[.25,.5,.25])
        old=[]
        for tick in range(25):
            source_prediction={
                name:h.predict_features(name,phi) for name in r8.HEAD_NAMES
            }
            target=[float(tick%2),.1+(tick/100.),float(tick%2)]
            sham,source=h.learn_after_physics(
                tick,phi,source_prediction,target
            )
            if tick<r8.SHAM_LAG:
                self.assertIsNone(source)
                self.assertIsNone(sham)
            else:
                self.assertTrue(0<=source<=tick-r8.SHAM_LAG)
                self.assertEqual(sham,old[source])
            old.append(target)
        self.assertEqual(h.updates["true"],25)
        self.assertEqual(h.updates["older_sham"],19)
        self.assertEqual(len(h.weights["true"]),3)
        self.assertTrue(all(len(w)==28 for w in h.weights["true"]))
        self.assertNotEqual(h.weights["true"],h.weights["older_sham"])

    def test_candidate_score_is_preaction_and_not_scripted_collision_reflex(self):
        sensory=[[.1,.2,.3]]*5
        physical=[[.8,.1,.8],[.2,.5,.2],[.2,.8,.2],
                  [.1,.2,.1],[.3,.8,.3]]
        score=r8.candidate_score(sensory,physical)
        self.assertEqual(r8.choose(score),2)
        self.assertEqual(r8.choose([0.]*5),0)
        self.assertEqual(len(score),5)
        self.assertEqual(r7.candidates((.4,-.2))[0],(.4,-.2))

    def test_physical_and_sensory_heads_stay_in_same_organism(self):
        for arm in ("frozen","r7_novelty","r8_true_physics",
                    "r8_older_sham","random_menu","r2_progress"):
            life=r8.simulate(11200,"nearfield","removal",arm)
            self.assertEqual(life["real_physical_steps"],
                             len(life["trace"]))
            self.assertTrue(0<life["real_physical_steps"]<=320)
            self.assertEqual(
                life["physical_head_updates"],
                {"true":life["real_physical_steps"],
                 "older_sham":max(0,life["real_physical_steps"]-6)}
            )
            self.assertEqual(
                life["sensory_head_updates"],
                {name:life["real_physical_steps"]
                 for name in r8.ORIGINAL_SENSOR_HEADS}
            )
            for k in ("w_input","w_recurrent","bias"):
                self.assertEqual(
                    life["original_434_neural_coefficients_at_birth"][k],
                    life["original_434_neural_coefficients_at_end"][k]
                )
            if arm!="r2_progress":
                self.assertEqual(
                    life["original_434_neural_coefficients_at_birth"],
                    life["original_434_neural_coefficients_at_end"]
                )
            self.assertTrue(all(r8._audit([life]).values()))

    def test_real_executed_command_and_physical_outcome_are_logged_consistently(self):
        for arm in ("r8_true_physics","r8_older_sham","r7_novelty"):
            life=r8.simulate(11201,"obstacles","static",arm)
            for t,frame in enumerate(life["trace"]):
                self.assertEqual(frame["tick"],t)
                index=frame["selected_candidate_index"]
                self.assertEqual(
                    frame["physically_executed_motor"],
                    frame["five_local_motor_candidates"][index]
                )
                if t<r8.WARMUP:
                    self.assertEqual(index,0)
                if arm=="r8_true_physics" and t>=r8.WARMUP:
                    self.assertEqual(index,r8.choose(
                        r8.candidate_score(
                            frame["preaction_candidate_sensory_predictions"]["true"],
                            frame["preaction_candidate_physical_predictions"]["true"]
                        )
                    ))
                self.assertEqual(
                    frame["actual_real_physical_outcomes"][0],
                    float(frame["physical_body_outcome"]["collision"])
                )
                src=frame["older_sham_physical_source_tick"]
                if src is None:
                    self.assertLess(t,6)
                else:
                    self.assertTrue(src<=t-6)
                    self.assertEqual(
                        frame["older_sham_target_from_real_history"],
                        life["trace"][src]["actual_real_physical_outcomes"]
                    )

    def test_removal_and_static_have_same_frames_before_change(self):
        for arm in r8.ARMS:
            a=r8.simulate(11202,"nearfield","static",arm)
            b=r8.simulate(11202,"nearfield","removal",arm)
            self.assertEqual(a["trace"][:r8.CHANGE_TICK],
                             b["trace"][:r8.CHANGE_TICK])
            self.assertFalse(a["environment_removed_five_walls"])
            self.assertTrue(b["environment_removed_five_walls"])
            self.assertEqual(len(a["postchange_rectangles"]),10)
            self.assertEqual(len(b["postchange_rectangles"]),5)

    def test_reproducibility_and_true_frozen_prediction_audit(self):
        for arm in ("frozen","r8_true_physics","r8_older_sham"):
            a=r8.simulate(11203,"obstacles","removal",arm)
            b=r8.simulate(11203,"obstacles","removal",arm)
            self.assertEqual(r8.sha(a),r8.sha(b))
            self.assertTrue(all(r8._audit([a]).values()))


if __name__=="__main__":
    unittest.main()
