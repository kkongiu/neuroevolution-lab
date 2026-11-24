"""FlyLoop M2 prospective integrity tests, not post-hoc science thresholds."""
import unittest
from neuroevolution import flyloop_m0 as m0
from neuroevolution import flyloop_m1 as m1
from neuroevolution import flyloop_m2 as m2


class FlyLoopM2Tests(unittest.TestCase):
    def test_prospective_cells_count_and_fixed_old_m0_topology(self):
        self.assertEqual(m2.GEOMETRIES,("baseline","procedural"))
        self.assertEqual(m2.SENSORS,("full","no_scent","contact_food","blind"))
        self.assertEqual(m2.ARMS,
                         ("frozen","random_motion","plastic","plastic_novelty"))
        self.assertEqual(m2.SEEDS,tuple(range(7600,7624)))
        self.assertEqual(m2.HORIZON,m0.HORIZON)
        self.assertEqual(m2.SWITCH,240)
        self.assertEqual(m0.N_ACTOR+m0.N_CRITIC,14)
        self.assertEqual(2*4*4*len(m2.SEEDS),768)
        self.assertEqual(m2.World(7600,"baseline","full").openings,
                         m0.OLD_OPENINGS)

    def test_procedural_gen_is_reproducible_connected_and_separate(self):
        signatures=set()
        for seed in m2.SEEDS:
            gen=m2.procedural_map(seed)
            gen_repeat=m2.procedural_map(seed)
            self.assertEqual(gen,gen_repeat)
            signatures.add(m2._digest({
                k:sorted(v) for k,v in gen.items()
            }))
            self.assertEqual(len(gen["posts"]),m2.PROCEDURAL_POSTS)
            self.assertEqual(len(gen["food"]),16)
            self.assertEqual(len(gen["hazards"]),9)
            self.assertFalse(gen["old"]&gen["new"])
            self.assertFalse(set(gen["food"])&set(gen["hazards"]))
            for openings in (gen["old"],gen["new"]):
                walls=m0.World._walls(openings)|gen["posts"]
                self.assertTrue(m2._reachable(walls))
                self.assertFalse(set(gen["food"])&walls)
                self.assertFalse(set(gen["hazards"])&walls)
            self.assertNotEqual(
                m0.World._walls(gen["old"])|gen["posts"],
                m0.World._walls(m0.OLD_OPENINGS)
            )
        self.assertGreaterEqual(len(signatures),20)

    def test_sensory_mask_is_applied_to_actor_and_critic(self):
        for geometry in m2.GEOMETRIES:
            for sense in m2.SENSORS:
                world=m2.World(7602,geometry,sense)
                opts,critic=world.observe()
                self.assertEqual(len(opts),5)
                self.assertEqual(len(critic),6)
                self.assertTrue(all(len(o)==8 for o in opts))
                if sense!="full":
                    self.assertTrue(all(o[1]==o[4]==0 for o in opts))
                    self.assertEqual(critic[4],0)
                if sense in ("contact_food","blind"):
                    self.assertTrue(all(o[3]==0 for o in opts))
                    self.assertEqual(critic[3],0)
                if sense=="blind":
                    self.assertTrue(all(o[2]==o[6]==0 for o in opts))
                self.assertTrue(any(o[5] for o in opts) or
                                not any(o[5] for o in opts))
        # Blind does not mean blind to the physical WORLD AFTER contact.
        w=m2.World(7602,"baseline","blind")
        f=next(iter(w.food_cooldown))
        w.body.x,w.body.y=f
        w.body.energy=30.
        reward,done=w.step(4)
        self.assertGreater(reward,10)
        self.assertEqual(w.body.food_eaten,1)
        self.assertFalse(done)

    def test_procedural_gate_changes_during_continuous_life(self):
        world=m2.World(7600,"procedural","blind")
        before=world.openings
        body_id=id(world.body)
        world.tick=240
        self.assertEqual(world.switch_if_due(),"change_gate_openings")
        self.assertEqual(world.openings,world.procedural_new)
        self.assertNotEqual(before,world.openings)
        self.assertEqual(id(world.body),body_id)
        self.assertEqual([r["step"] for r in world.events],[240])
        self.assertEqual(world.events[0]["openings"],
                         sorted(world.procedural_new))

    def test_initial_world_and_14_weights_are_paired_across_arms(self):
        for geom in m2.GEOMETRIES:
            rows=[m2.simulate(7600,geom,"blind",arm) for arm in m2.ARMS]
            self.assertEqual(
                len({m2._digest(r["start_map"]) for r in rows}),1
            )
            self.assertEqual(
                len({m2._digest(r["initial_weights"]) for r in rows}),1
            )
            for r in rows:
                self.assertTrue(m2._verify_masks([r]))
                self.assertTrue(m2._verify_physics([r]))
                self.assertEqual(
                    sum(t["food"] for t in r["trajectory"] if "action" in t),
                    r["food_consumed"]
                )
            self.assertEqual(rows[0]["initial_weights"],
                             rows[0]["final_weights"])
            self.assertEqual(rows[1]["initial_weights"],
                             rows[1]["final_weights"])
            self.assertEqual(rows[0]["online_weight_update_events"],0)
            self.assertEqual(rows[1]["online_weight_update_events"],0)
            self.assertTrue(all(tr["action"]!=4
                            for tr in rows[1]["trajectory"]
                            if "action" in tr))
            self.assertGreater(rows[2]["online_weight_update_events"],0)
            self.assertNotEqual(rows[2]["initial_weights"],
                                rows[2]["final_weights"])

    def test_new_worlds_and_agents_repeat_identical(self):
        for sensor in m2.SENSORS:
            a=m2.simulate(7601,"procedural",sensor,"plastic_novelty")
            b=m2.simulate(7601,"procedural",sensor,"plastic_novelty")
            self.assertEqual(m2._digest(a),m2._digest(b))
            self.assertTrue(m2._verify_masks([a]))
            self.assertTrue(m2._verify_physics([a]))
            self.assertLessEqual(a["steps_alive"],480)
        with self.assertRaises(ValueError):
            m2.simulate(7601,"fake","blind","plastic")
        with self.assertRaises(ValueError):
            m2.simulate(7601,"baseline","clairvoyant","plastic")
        with self.assertRaises(ValueError):
            m2.simulate(7601,"baseline","blind","scripted_forage")
        with self.assertRaises(ValueError):
            m2.simulate(7300,"baseline","blind","plastic")


if __name__=="__main__":
    unittest.main()
