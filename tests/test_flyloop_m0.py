"""Test real embodied causal evidence and sealed source, never fit science seeds."""
from __future__ import annotations
import copy
import hashlib
import unittest

from neuroevolution import m4_2,m4_4a,m5_6_online,m5_6_curiosity
from neuroevolution import flyloop_m0 as fl

class FlyLoopM0Tests(unittest.TestCase):
    def test_science_protocol_locked_and_distinct_dev(self):
        self.assertEqual(fl.SEEDS,tuple(range(44000,44016)))
        self.assertEqual(fl.DEV_SEEDS,tuple(range(43000,43004)))
        self.assertFalse(set(fl.SEEDS)&set(fl.DEV_SEEDS))
        self.assertEqual(fl.GATES,(2,3,7,8))
        self.assertEqual(fl.BOUT_STEPS,150)
        self.assertEqual(fl.ARMS,("frozen","plastic","plastic_novelty"))
        self.assertIs(fl.Learner,m5_6_online.Learner)
        self.assertEqual(fl.LEARNING_RATE,m5_6_online.RATE)
        self.assertEqual(fl.EXPLORATION_CAP,m5_6_curiosity.IntrinsicReward().maximum)

    def test_true_physics_walls_hazard_food_energy_not_fake_reward(self):
        world=fl.World(2)
        self.assertTrue(world.blocked((5,5)))
        self.assertFalse(world.blocked((5,2)))
        self.assertTrue(world.blocked((0,5)))
        self.assertFalse(world.blocked(fl.SPAWN))
        self.assertNotIn(fl.SPAWN,world.walls)
        body=fl.Body(position=(4,5))
        start=body.energy
        bump=world.transition(body,1,0)
        self.assertTrue(bump["wall_collision"])
        self.assertFalse(bump["actual_moved"])
        self.assertEqual(body.position,(4,5))
        self.assertAlmostEqual(body.energy,start-fl.BASE_MOVE_COST-fl.WALL_EXTRA_COST)
        self.assertEqual(bump["physical_reward"],fl.FEEDBACK_WALL)
        body=fl.Body(position=(4,3))
        moved=world.transition(body,0,0)
        self.assertEqual(body.position,(4,2))
        self.assertFalse(moved["wall_collision"])
        body=fl.Body(position=(3,4))
        meal=world.transition(body,0,0)
        self.assertTrue(meal["actual_food"])
        self.assertEqual(body.food_eaten,1)
        self.assertEqual(body.far_food_eaten,0)
        self.assertAlmostEqual(body.energy,min(fl.ENERGY_MAX,fl.ENERGY_INITIAL-fl.BASE_MOVE_COST+fl.FOOD_ENERGY))
        self.assertFalse(world.transition(body,2,1)["actual_food"])
        body=fl.Body(position=(4,5))
        danger=world.transition(body,2,0)
        self.assertTrue(danger["hazard_event"])
        self.assertEqual(body.damage_events,1)
        self.assertAlmostEqual(body.energy,fl.ENERGY_INITIAL-fl.BASE_MOVE_COST-fl.HAZARD_EXTRA_COST)

    def test_observations_only_current_sense_and_19_channels(self):
        world=fl.World(2)
        a=fl.Body(position=(4,5))
        sensors=world.observe(a,0)
        self.assertEqual(len(sensors),19)
        self.assertEqual(sensors[1],1.)
        self.assertEqual(sensors[18],1.)
        self.assertIn(1.,sensors[:4])
        self.assertIn(1.,sensors[4:8])
        world.food_ready[fl.FOODS[0]]=999
        other=world.observe(a,0)
        self.assertNotEqual(other[8:12],sensors[8:12])
        with self.assertRaises(ValueError):
            fl.World(9)

    def test_original_m56_neural_instance_with_exact_260_weights(self):
        a,w=fl.initial_agent(fl.DEV_SEEDS[0])
        b,v=fl.initial_agent(fl.DEV_SEEDS[0])
        self.assertEqual(w,v)
        self.assertEqual(len(w),260)
        self.assertEqual(len(a.weights),260)
        obs=fl.World(2).observe(fl.Body(),0)
        action,cache=a.action(obs,(0,1,2,3),__import__("random").Random(17))
        self.assertIn(action,range(4))
        self.assertEqual(len(cache[0]),19)
        self.assertEqual(a.learn(cache,-.2,enabled=False),0.)
        self.assertEqual(a.update_count,0)

    def test_matched_three_arms_same_source_initial_genome_and_reset(self):
        seed=fl.DEV_SEEDS[1]
        group=fl.one_seed(seed,trace=True)
        rows=group["rows"]
        self.assertEqual(len(rows),14)
        self.assertEqual(len({r["weights_initial_sha256"] for r in rows
                              if r["phase"]=="source_a"}),1)
        for row in rows:
            self.assertTrue(fl.audit_physical_trace(row))
            self.assertEqual(row["gate"],fl.GATES[fl.PHASES.index(row["phase"])])
            self.assertLessEqual(row["steps_survived"],150)
            self.assertEqual(len(row["trace"]),row["steps_survived"])
            self.assertGreaterEqual(row["final_energy_physical"],0.)
            self.assertLessEqual(row["final_energy_physical"],fl.ENERGY_MAX)
            if row["arm"]=="frozen":
                self.assertEqual(row["learning_updates_this_bout"],0)
                self.assertEqual(row["intrinsic_novelty_bonus_total_NOT_physical"],0)
            if row["phase"]=="shifted_eval":
                self.assertEqual(row["weights_initial_sha256"],row["weights_final_sha256"])
                self.assertEqual(row["learning_updates_this_bout"],0)
            for item in row["trace"]:
                if item["wall_collision"]:
                    self.assertEqual(item["intrinsic_bonus_only_internal"],0)
        frozen=[r for r in rows if r["arm"]=="frozen"]
        self.assertEqual(len({r["weights_final_sha256"] for r in frozen}),1)
        reset=next(r for r in rows if r["arm"]=="plastic_novelty_reset" and r["phase"]=="shifted_eval")
        source=next(r for r in rows if r["arm"]=="plastic_novelty" and r["phase"]=="source_a")
        self.assertEqual(reset["weights_initial_sha256"],source["weights_initial_sha256"])

    def test_audit_detects_unearned_food_and_future_lookahead(self):
        row=fl.one_seed(fl.DEV_SEEDS[2])["rows"][0]
        original=copy.deepcopy(row)
        self.assertTrue(fl.audit_physical_trace(row))
        altered=copy.deepcopy(row)
        altered["trace"][0]["energy_after"]+=3
        self.assertFalse(fl.audit_physical_trace(altered))
        altered=copy.deepcopy(row)
        altered["trace"][0]["observation_19_before_action"][0]=5.
        self.assertFalse(fl.audit_physical_trace(altered))
        self.assertEqual(row,original)

    def test_read_only_source_modules_untouched_by_flyloop(self):
        # Source SHA256 is verified by comparing PR changed files;
        # this test checks direct binding to the legacy class/method.
        self.assertEqual(fl.Learner.__module__,"neuroevolution.m5_6_online")
        self.assertEqual(fl.Learner.learn.__name__,"learn")
        self.assertFalse(hasattr(fl.World,"optimal_action"))
        self.assertFalse(hasattr(fl.World,"pathfind"))
        self.assertFalse(hasattr(fl.World,"flee"))
        with self.assertRaises(ValueError):
            fl.one_seed(123456)
        with self.assertRaises(ValueError):
            fl.run_bout(fl.DEV_SEEDS[0],"oracle",0,fl.initial_agent(fl.DEV_SEEDS[0])[0])

if __name__=="__main__":
    unittest.main()
