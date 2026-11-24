"""R6 physically intervened training and genuinely held-out transfer tests."""
import copy
import math
import unittest

from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r2 as r2
from neuroevolution import flyloop_r4 as r4
from neuroevolution import flyloop_r5 as r5
from neuroevolution import flyloop_r6 as r6


class FlyLoopR6Tests(unittest.TestCase):
    def test_protocol_exact_splits_17_gates_inputs(self):
        self.assertEqual(r6.TRAIN_SEEDS,tuple(range(10500,10524)))
        self.assertEqual(r6.TEST_SEEDS,tuple(range(10600,10624)))
        self.assertTrue(set(r6.TRAIN_SEEDS).isdisjoint(r6.TEST_SEEDS))
        self.assertTrue(set(r6.TRAIN_SEEDS+r6.TEST_SEEDS).isdisjoint(r5.SEEDS))
        self.assertEqual(len(r6.SNAPSHOT_TICKS),10)
        self.assertEqual(r6.SNAPSHOT_TICKS[0],40)
        self.assertEqual(r6.SNAPSHOT_TICKS[-1],220)
        self.assertEqual(r6.SCENES,("obstacles","nearfield"))
        self.assertEqual(len(r6.ACTIONS),5)
        self.assertEqual(r6.BRANCH_STEPS,3)
        self.assertEqual(r6.INFORMATIVE_THRESHOLD,.02)
        self.assertEqual(r6.PHYSICAL_DIVERGENCE_THRESHOLD,.04)
        self.assertEqual(r4.HEAD_PARAMS,84)
        self.assertEqual(r1.N,16)
        self.assertEqual(r1.PARAMS,434)
        self.assertEqual(24*2*2,96)
        self.assertEqual(96*10*5,4800)
        with self.assertRaises(ValueError):
            r6.World(10300,"obstacles")
        with self.assertRaises(ValueError):
            r6.World(10500,"food")

    def test_train_and_test_action_intervention_settings_are_different(self):
        natural=(.45,-.25)
        training=r6.forced_commands(natural,"train")
        test=r6.forced_commands(natural,"test")
        self.assertEqual(list(training),list(r6.ACTIONS))
        self.assertEqual(list(test),list(r6.ACTIONS))
        self.assertEqual(training["natural"],test["natural"])
        self.assertEqual(training["steer_left"],(.45,-.95))
        self.assertEqual(training["steer_right"],(.45,.95))
        self.assertEqual(training["throttle_low"],(.10,-.25))
        self.assertEqual(training["throttle_high"],(.90,-.25))
        self.assertEqual(test["steer_left"],(.45,-.55))
        self.assertEqual(test["steer_right"],(.45,.55))
        self.assertEqual(test["throttle_low"],(.30,-.25))
        self.assertEqual(test["throttle_high"],(.70,-.25))
        with self.assertRaises(ValueError):
            r6.forced_commands(natural,"unregistered")

    def test_three_equal_sized_heads_training_targets_are_physically_observed(self):
        heads=r4.ActionSensorHeads(10500)
        body=r6.World(10500,"obstacles")
        neural=r2.MotorLearner(10500,"recurrent_frozen")
        readings=body.perceive()
        _,hidden,_,_,natural=neural.command(readings)
        before_world=copy.deepcopy(vars(body.body))
        snap=r6.collect_snapshot(
            body,40,hidden,readings,natural,"train",heads
        )
        self.assertEqual(vars(body.body),before_world)
        self.assertEqual(body.frame,0)
        self.assertEqual(heads.updates,{"true":5,"blind":5,"yoked":5})
        self.assertEqual(len(snap["trials"]),5)
        self.assertEqual(len(snap["contrasts"]),4)
        self.assertEqual(snap["trials"][0]["forced_real_motor"],list(natural))
        all_motors=[t["forced_real_motor"] for t in snap["trials"]]
        for j,t in enumerate(snap["trials"]):
            self.assertEqual(
                t["same_state_rotated_training_motor"],
                all_motors[(j+1)%5],
            )
            self.assertGreaterEqual(len(t["actual_physical_steps"]),1)
            self.assertLessEqual(len(t["actual_physical_steps"]),3)
            self.assertEqual(
                t["actual_three_step_terminal_wall_rays"],
                t["actual_physical_steps"][-1]["observed_wall_rays"],
            )
            self.assertEqual(
                set(t["head_predictions_before_executing"]),
                {"true","blind","yoked","persistence"},
            )
        self.assertNotEqual(snap["head_hash_before_snapshot"],
                            snap["head_hash_after_snapshot"])

    def test_heldout_head_weight_frozen_no_test_label_contamination(self):
        heads=r4.ActionSensorHeads(10501)
        train=r6.collect_life(10501,"nearfield","train",heads)
        trained=copy.deepcopy(heads.weights)
        number_training_updates=heads.updates.copy()
        test=r6.collect_life(10601,"nearfield","test",heads)
        self.assertEqual(heads.weights,trained)
        self.assertEqual(heads.updates,number_training_updates)
        self.assertEqual(train["phase"],"train")
        self.assertEqual(test["phase"],"test")
        self.assertEqual(len(test["real_natural_life"]),240)
        self.assertEqual(test["actual_snapshot_count"],10)
        self.assertEqual(test["physically_executed_3step_trials"],50)
        self.assertEqual(test["forced_vs_natural_contrasts"],40)
        self.assertEqual(
            train["initial_neural_434_coefficients"],
            train["final_neural_434_coefficients"],
        )

    def test_original_body_and_natural_clone_match_exactly(self):
        heads=r4.ActionSensorHeads(10502)
        life=r6.collect_life(10502,"obstacles","train",heads)
        recorded={r["tick"]:r for r in life["real_natural_life"]}
        for p in life["snapshots_with_five_physical_branches"]:
            natural=p["trials"][0]
            original=recorded[p["tick"]]
            self.assertEqual(
                natural["actual_physical_steps"][0]["physics"],
                original["physical_outcome"],
            )
            self.assertEqual(
                natural["actual_physical_steps"][0]["observed_wall_rays"],
                original["real_next_three_wall_rays"],
            )
        self.assertEqual(life["natural_life_real_frames"],240)

    def test_dead_intervention_branches_stay_physically_dead(self):
        heads=r4.ActionSensorHeads(10503)
        world=r6.World(10503,"obstacles")
        world.body.x=.301
        world.body.y=10.
        world.body.heading=math.pi
        world.body.speed=1.
        world.body.integrity=.25
        neural=r2.MotorLearner(10503,"recurrent_frozen")
        s=world.perceive()
        _,h,_,_,motor=neural.command(s)
        p=r6.collect_snapshot(world,40,h,s,motor,"test",heads)
        self.assertEqual(world.frame,0)
        self.assertEqual(world.body.integrity,.25)
        self.assertTrue(any(t["early_absorbing_death"] for t in p["trials"]))
        for t in p["trials"]:
            if t["early_absorbing_death"]:
                self.assertTrue(
                    t["terminal_body"]["energy"]<=0 or
                    t["terminal_body"]["integrity"]<=0
                )
                self.assertLess(t["real_integrations"],3)
        self.assertEqual(heads.updates,{"true":0,"blind":0,"yoked":0})

    def test_replay_causality_and_sensory_model_training_independent_of_test(self):
        heads=r4.ActionSensorHeads(10500)
        train=r6.collect_life(10500,"obstacles","train",heads)
        trained=copy.deepcopy(heads.weights)
        before=r6.sha(heads.weights)
        test=r6.collect_life(10600,"obstacles","test",heads)
        report=r6._audit([train],[test],trained,before,r6.sha(heads.weights))
        self.assertTrue(report["independently_replayed_original_and_all_physical_clones"])
        self.assertTrue(report["paired_initial_434_coefficients_and_world_physics"])
        self.assertTrue(report["preaction_16_neurons_eight_senses_and_true_blind_shuffled_inputs"])
        self.assertTrue(report["exact_training_only_LMS_updates_and_test_weights_frozen"])
        self.assertEqual(report["recomputed_training_trial_updates"],
                         train["physically_executed_3step_trials"])

    def test_each_seed_full_trial_reproducible(self):
        def one():
            heads=r4.ActionSensorHeads(10504)
            train=r6.collect_life(10504,"nearfield","train",heads)
            test=r6.collect_life(10604,"nearfield","test",heads)
            return r6.sha([train,test,heads.weights])
        self.assertEqual(one(),one())


if __name__=="__main__":
    unittest.main()
