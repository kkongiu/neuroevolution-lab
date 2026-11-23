"""Tests for FlyLoop M0 ecology, sensor-only actor and honest ablations."""
import unittest

from neuroevolution import flyloop_m0 as f


class FlyLoopM0Tests(unittest.TestCase):
    def test_world_is_persistent_local_and_barrier_changes(self):
        world=f.World(7100)
        self.assertEqual((world.body.x,world.body.y),(2,6))
        self.assertEqual(world.body.energy,70.)
        self.assertEqual(world.body.health,100.)
        self.assertEqual(len(world.food_cooldown),16)
        self.assertEqual(len(world.hazards),9)
        self.assertEqual(world.openings,f.OLD_OPENINGS)
        self.assertEqual(len(set(world.food_cooldown)&set(world.hazards)),0)
        self.assertEqual(len(world.observe()[0]),5)
        self.assertEqual(len(world.observe()[0][0]),8)
        self.assertEqual(len(world.observe()[1]),6)
        self.assertNotIn("x",world.observe()[1])
        self.assertEqual(world._walls(f.OLD_OPENINGS)&{(6,2),(6,10)},set())
        self.assertTrue({(6,5),(6,8)} <= world.walls)
        world.tick=240
        world.switch_if_due()
        self.assertEqual(world.openings,f.NEW_OPENINGS)
        self.assertTrue({(6,2),(6,10)} <= world.walls)
        self.assertEqual(world.walls&{(6,5),(6,8)},set())
        self.assertTrue(world.alive_at_switch)

    def test_collision_physical_penalty_no_illegal_motion(self):
        world=f.World(7100)
        world.body.x=1
        world.body.y=6
        initial=world.body.energy
        # WEST into x=0 boundary
        r,done=world.step(3)
        self.assertEqual((world.body.x,world.body.y),(1,6))
        self.assertAlmostEqual(r,-.75)
        self.assertAlmostEqual(world.body.energy,initial-.75)
        self.assertEqual(world.body.hits,1)
        self.assertFalse(done)
        with self.assertRaises(ValueError):
            world.step(7)

    def test_actual_consumption_and_repeat_visits(self):
        world=f.World(7100)
        p=next(iter(world.food_cooldown))
        world.body.x=p[0]
        world.body.y=p[1]
        world.tick=3
        world.body.energy=30.
        r,done=world.step(4)  # wait while on food
        self.assertEqual(world.body.food_eaten,1)
        self.assertGreater(r,10)
        self.assertEqual(world.food_cooldown[p],3+f.REGROW_TICKS)
        self.assertFalse(done)
        world.tick=3+f.REGROW_TICKS
        world.step(4)
        self.assertEqual(world.body.food_eaten,2)
        self.assertEqual(world.body.food_returns,1)

    def test_online_weights_and_frozen_ablation(self):
        r0=f.simulate(7100,"frozen")
        r1=f.simulate(7100,"plastic")
        r2=f.simulate(7100,"plastic_novelty",novelty_coefficient=0)
        self.assertEqual(r0["start_map"],r1["start_map"])
        self.assertEqual(r0["initial_weights"],r1["initial_weights"])
        self.assertEqual(r0["initial_weights"],r0["final_weights"])
        self.assertEqual(r0["online_parameter_updates"],0)
        self.assertGreater(r1["online_parameter_updates"],0)
        self.assertNotEqual(r1["initial_weights"],r1["final_weights"])
        # Same curriculum, policy, random streams, gradients, predictions,
        # and sensor novelty counts, with novelty beta=0.
        for key in r1:
            if key not in ("arm","novelty_coefficient"):
                self.assertEqual(r1[key],r2[key],key)
        self.assertGreaterEqual(r1["steps_alive"],1)
        self.assertLessEqual(r1["steps_alive"],480)

    def test_seed_determinism_and_crossing_physics(self):
        for arm in f.ARMS:
            a,b=f.simulate(7101,arm),f.simulate(7101,arm)
            self.assertEqual(f._sha(a),f._sha(b))
            for line in a["trace"]:
                if "action" not in line:
                    self.assertEqual(line["event"],"gate_closure_crush")
                    continue
                x,y=line["position"]
                valid=f.NEW_OPENINGS if line["step"]>=240 else f.OLD_OPENINGS
                self.assertNotIn((x,y),f.World._walls(valid))
                fx,fy=line["from"]
                self.assertLessEqual(abs(fx-x)+abs(fy-y),1)
                self.assertIn(line["action"],range(5))
                self.assertGreaterEqual(line["energy"],0)
                self.assertGreaterEqual(line["health"],0)
        self.assertEqual(f.WORLD_SEEDS,tuple(range(7100,7132)))

    def test_reward_depends_on_actual_physiology_not_hidden_labels(self):
        world=f.World(7102)
        physical_before=(world.body.energy,world.body.health)
        world.step(4)
        self.assertAlmostEqual(world.trace[-1]["physical_reward"],-.25)
        self.assertEqual(world.body.health,physical_before[1])
        self.assertEqual(world.body.energy,physical_before[0]-.25)

    def test_only_three_arms_and_no_evolution(self):
        self.assertEqual(f.ARMS,("frozen","plastic","plastic_novelty"))
        self.assertEqual(f.N_ACTOR+f.N_CRITIC,14)
        with self.assertRaises(ValueError):
            f.Controller(1,"scripted_forage")
        self.assertEqual(f.NOVELTY_COEFFICIENT,1.0)
        self.assertEqual(f.HORIZON,480)
        self.assertEqual(f.SWITCH_TICK,240)


if __name__=="__main__":
    unittest.main()
