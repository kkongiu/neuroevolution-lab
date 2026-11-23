"""FlyLoop M1: physics, sensory privilege, new topology, reset and reproducibility."""
import unittest

from neuroevolution import flyloop_m0 as m0
from neuroevolution import flyloop_m1 as m1


class FlyLoopM1Tests(unittest.TestCase):
    def test_four_scenarios_and_four_arms_frozen(self):
        self.assertEqual(len(m1.SCENARIOS),4)
        self.assertEqual(len(m1.ARMS),4)
        self.assertEqual(m1.SEEDS,tuple(range(7300,7332)))
        self.assertEqual(m1.HORIZON,480)
        self.assertEqual(m1.SWITCH,240)
        self.assertEqual(m1.RESTORE,360)
        self.assertEqual(m0.N_ACTOR+m0.N_CRITIC,14)
        with self.assertRaises(ValueError):
            m1.FlyWorld(7300,"not_allowed")
        with self.assertRaises(ValueError):
            m1.simulate(7300,"no_scent_gate","scripted_forager")

    def test_unscented_no_privileged_attraction_but_adjacent_visible(self):
        w=m1.FlyWorld(7300,"no_scent_gate")
        self.assertEqual(len(w.observe()[0]),5)
        self.assertTrue(all(f[1]==0 and f[4]==0
                            for f in w.observe()[0]))
        # Place body adjacent to a food cell, without moving resources.
        food=next(iter(w.food_cooldown))
        for i,(dx,dy) in enumerate(m0.DIRECTIONS[:4]):
            pos=(food[0]-dx,food[1]-dy)
            if pos not in w.walls:
                w.body.x,w.body.y=pos
                opts,_=w.observe()
                self.assertEqual(opts[i][3],1.)
                self.assertEqual(opts[i][4],0.)
                break
        else:
            self.fail("No accessible neighboring food sensor")

    def test_scent_occlusion_and_bounded_noise(self):
        w=m1.FlyWorld(7300,"occluded_noisy_gate")
        w.body.x,w.body.y=5,6
        w.food_cooldown={(7,6):0}
        self.assertEqual(w._path_only_scent()[1],0.0)
        observed=w.observe()[0][1][4]
        self.assertGreaterEqual(observed,0.)
        self.assertLessEqual(observed,m1.SENSOR_NOISE_BOUND+1e-12)
        self.assertAlmostEqual(w.observe()[0][1][4],observed)
        w.food_cooldown={(5,5):0}
        self.assertGreater(w._path_only_scent()[0],0.)
        self.assertLessEqual(abs(w._sensor_noise(0)),m1.SENSOR_NOISE_BOUND+1e-12)

    def test_novel_topology_and_resource_no_wall_overlap(self):
        a=m1.FlyWorld(7300,"novel_layout")
        b=m1.FlyWorld(7300,"occluded_noisy_gate")
        self.assertNotEqual(a.walls,b.walls)
        self.assertEqual(a.openings,m1.NOVEL_OLD_GATES)
        self.assertFalse(set(a.food_cooldown)&a.walls)
        self.assertFalse(set(a.hazards)&a.walls)
        self.assertFalse(set(a.food_cooldown)&set(a.hazards))
        self.assertEqual(sum(p[0]<6 for p in a.food_cooldown),8)
        self.assertEqual(sum(p[0]>6 for p in a.food_cooldown),8)
        a.tick=240
        event=a.switch_if_due()
        self.assertEqual(event,"change_gate_openings")
        self.assertEqual(a.openings,m1.NOVEL_NEW_GATES)
        self.assertEqual(a.walls & {(6,4),(6,11)},set())

    def test_food_move_then_return_same_sites_with_no_map_message(self):
        w=m1.FlyWorld(7301,"food_relocation")
        before=set(w.food_cooldown)
        hazards=set(w.hazards)
        w.tick=240
        self.assertEqual(w.switch_if_due(),"relocate_food")
        moved=set(w.food_cooldown)
        self.assertEqual(len(moved),16)
        self.assertFalse(before&moved)
        self.assertFalse(moved&hazards)
        self.assertFalse(moved&w.walls)
        self.assertEqual(sum(p[0]<6 for p in moved),8)
        self.assertEqual(sum(p[0]>6 for p in moved),8)
        self.assertEqual(w.observe()[0][0][0],1.0)
        self.assertEqual(w.openings,m0.OLD_OPENINGS)  # no gate confound
        self.assertEqual([e["step"] for e in w.events],[240])
        w.tick=360
        self.assertEqual(w.switch_if_due(),"restore_food")
        self.assertEqual(set(w.food_cooldown),before)
        self.assertEqual([e["step"] for e in w.events],[240,360])

    def test_relocation_is_exogenous_same_if_agent_moves_elsewhere(self):
        a=m1.FlyWorld(7303,"food_relocation")
        b=m1.FlyWorld(7303,"food_relocation")
        a.body.x,a.body.y=2,5
        b.body.x,b.body.y=10,9
        a.tick=b.tick=240
        a.switch_if_due()
        b.switch_if_due()
        self.assertEqual(a.relocated_sites,b.relocated_sites)
        self.assertEqual(a.food_cooldown,b.food_cooldown)

    def test_paired_scent_worlds_and_identical_initial_parameters(self):
        for scenario in m1.SCENARIOS:
            rows=[m1.simulate(7300,scenario,arm) for arm in m1.ARMS]
            initial={m1.canonical_digest(r["initial_weights"]) for r in rows}
            maps={m1.canonical_digest(r["start_map"]) for r in rows}
            self.assertEqual(len(initial),1)
            self.assertEqual(len(maps),1)
            self.assertEqual(rows[0]["initial_weights"],rows[2]["initial_weights"])
            self.assertEqual(rows[0]["final_weights"],rows[0]["initial_weights"])
            self.assertEqual(rows[0]["online_parameter_updates"],0)
            self.assertGreater(rows[1]["online_parameter_updates"],0)
            self.assertNotEqual(rows[1]["final_weights"],rows[1]["initial_weights"])
            self.assertEqual(
                rows[1]["trajectory"][:240],
                rows[3]["trajectory"][:240],
            )  # reset arm identical before first reset
            self.assertTrue(all(e["after_reset_equal_initial"]
                                for e in rows[3]["resets"]))
            self.assertEqual(
                rows[1]["steps_alive"],
                sum(p["steps"] for p in rows[1]["by_phase"].values()),
            )

    def test_reproducibility_and_legal_step_physics(self):
        for scenario in m1.SCENARIOS:
            a=m1.simulate(7302,scenario,"plastic_novelty")
            b=m1.simulate(7302,scenario,"plastic_novelty")
            self.assertEqual(m1.canonical_digest(a),m1.canonical_digest(b))
            self.assertTrue(m1._integrity([a]))
            self.assertLessEqual(a["steps_alive"],480)
            self.assertEqual(a["food_consumed"],
                             sum(x["food"] for x in a["by_phase"].values()))
            self.assertEqual(len(a["trajectory"]),
                             a["steps_alive"]+len(a["events"]))

    def test_physical_reward_remains_actual_energy_and_health(self):
        w=m1.FlyWorld(7300,"no_scent_gate")
        orig_energy=w.body.energy
        orig_health=w.body.health
        reward,death=w.step(4)
        self.assertEqual(reward,-.25)
        self.assertEqual(w.body.energy,orig_energy-.25)
        self.assertEqual(w.body.health,orig_health)
        self.assertFalse(death)


if __name__=="__main__":
    unittest.main()
