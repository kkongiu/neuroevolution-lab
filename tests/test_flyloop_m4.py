"""FlyLoop M4 local-only chemo sense, causal feedback and immutable M3 controls."""
import unittest

from neuroevolution import flyloop_m0 as m0
from neuroevolution import flyloop_m3 as m3
from neuroevolution import flyloop_m4 as m4


class M4Tests(unittest.TestCase):
    def test_locked_design_and_invalid_conditions(self):
        self.assertEqual(len(m4.SEEDS),32)
        self.assertEqual(m4.SEEDS,tuple(range(8300,8332)))
        self.assertEqual(m4.SCENARIOS,("baseline","procedural","relocation"))
        self.assertEqual(len(m4.ARMS),7)
        self.assertEqual(32*3*7,672)
        self.assertEqual(m0.N_ACTOR+m0.N_CRITIC,14)
        self.assertEqual(m4.DISTANCE_CAP,7)
        self.assertEqual(m4.NOISE_BOUND,.015)
        with self.assertRaises(ValueError):
            m4.simulate(7900,"baseline","chemo_true")
        with self.assertRaises(ValueError):
            m4.simulate(8300,"not_world","chemo_true")
        with self.assertRaises(ValueError):
            m4.simulate(8300,"baseline","scripted_follow_odor")

    def test_only_current_cell_concentration_and_wall_occlusion(self):
        w=m4.World(8300,"baseline")
        w.body.x,w.body.y=5,6
        w.food_cooldown={(7,6):0}
        # An impassable wall separates these cells; openings far away.
        self.assertNotIn((7,6),w._distances_from((5,6)))
        odor,noise=w.odor_here()
        self.assertLessEqual(odor,m4.NOISE_BOUND+1e-9)
        self.assertLessEqual(abs(noise),m4.NOISE_BOUND+1e-9)
        self.assertEqual(w.odor_here(),(odor,noise))
        w.food_cooldown={(5,5):0}
        near,_=w.odor_here()
        self.assertGreater(near,odor)
        self.assertGreater(near,0.05)
        self.assertLessEqual(near,1.)
        # Source is at current location; it cannot reveal a vector.
        w.food_cooldown={(5,6):0}
        on_source,_=w.odor_here()
        self.assertGreater(on_source,near)

    def test_lagged_causal_delta_uses_only_current_and_previous_odor(self):
        w=m4.World(8300,"baseline")
        raw,critic=w.observe()
        m=m4.SensorMemory(8300)
        f0,c0,meta0=m.encode(raw,critic,"chemo_true",0,.25)
        self.assertEqual(meta0["odor_previous"],None)
        self.assertEqual(meta0["true_delta"],0.)
        self.assertEqual(meta0["source_tick"],None)
        self.assertTrue(all(v[1]==0 and v[2]==0 for v in f0))
        self.assertEqual(c0[3],.25)
        m.commit_action(1,0,.25)
        f1,c1,meta1=m.encode(raw,critic,"chemo_true",1,.35)
        self.assertAlmostEqual(meta1["true_delta"],.6)
        self.assertEqual(meta1["source_tick"],1)
        self.assertAlmostEqual(f1[1][1],.6)
        self.assertAlmostEqual(f1[3][1],-.6)
        self.assertAlmostEqual(f1[1][2],.15)
        self.assertEqual(f1[1][3],1.)
        self.assertEqual(f1[3][3],-1.)
        self.assertEqual(f1[4][3],0.)
        self.assertAlmostEqual(c1[4],.6)

    def test_sham_uses_older_scalar_change_and_private_rng(self):
        w=m4.World(8301,"baseline")
        raw,critic=w.observe()
        m=m4.SensorMemory(8301)
        sequence=(.2,.35,.22,.5,.1,.7,.43,.33)
        private_init=m.sham_rng.getstate()
        for t,odor in enumerate(sequence):
            f,c,info=m.encode(raw,critic,"chemo_sham",t,odor)
            self.assertEqual(info["odor_now"],odor)
            if t<3:
                self.assertIsNone(info["source_tick"])
                self.assertEqual(info["used_delta"],0.)
            else:
                self.assertLessEqual(info["source_tick"],t-2)
                old=next(s for s in m.samples
                         if s["tick"]==info["source_tick"])
                self.assertEqual(info["used_delta"],old["delta"])
                self.assertEqual(info["used_ema"],old["ema"])
            m.commit_action(1,0,odor)
        self.assertNotEqual(private_init,m.sham_rng.getstate())

    def test_no_future_food_or_hazard_in_actor_and_critic(self):
        for scenario in m4.SCENARIOS:
            for arm in m4.ARMS:
                r=m4.simulate(8300,scenario,arm)
                self.assertTrue(r["trajectory"])
                for t in r["trajectory"]:
                    if "action" not in t:
                        continue
                    self.assertTrue(all(
                        f[i]==0. for f in t["raw_blind_actor_inputs"]
                        for i in (1,2,3,4,6)
                    ))
                    self.assertEqual(t["raw_blind_critic_inputs"][3],0.)
                    self.assertEqual(t["raw_blind_critic_inputs"][4],0.)
                    self.assertTrue(0<=t["measured_odor_scalar"]<=1)
                    self.assertLessEqual(abs(t["odor_measurement_noise"]),.01501)
                    self.assertEqual(t["memory"]["odor_now"],
                                     t["measured_odor_scalar"])
                phys,blind,causal,weights,field=m4._validate([r])
                self.assertTrue(all((phys,blind,causal,weights,field)),
                                (scenario,arm,(phys,blind,causal,weights,field)))
                self.assertLessEqual(r["steps_alive"],480)

    def test_pairing_and_relocation_not_controlled_by_agent(self):
        for scenario in m4.SCENARIOS:
            lives=[m4.simulate(8301,scenario,a) for a in m4.ARMS]
            self.assertEqual(len({m4.digest(r["start_map"])
                                  for r in lives}),1)
            self.assertEqual(len({m4.digest(r["initial_weights"])
                                  for r in lives}),1)
            self.assertTrue(all(
                r["initial_weights"]==r["final_weights"]
                for r in lives if r["arm"] in
                ("frozen","random_motion","chemo_frozen")
            ))
            if scenario=="relocation":
                changes={
                    m4.digest(r["events"][0]["food_sites"])
                    for r in lives if r["events"]
                }
                self.assertEqual(len(changes),1)
                self.assertTrue(all(
                    r["start_map"]["future_openings"]==
                    r["start_map"]["initial_openings"]
                    for r in lives
                ))

    def test_unchanged_m2_m3_controls_reproduce_exact_trajectories(self):
        self.assertTrue(m4.reference_history_compatibility())

    def test_repeatability_and_real_rewards_not_fabricated(self):
        for arm in ("plastic_plain","motor_only","chemo_true",
                    "chemo_sham","chemo_frozen"):
            a=m4.simulate(8302,"procedural",arm)
            b=m4.simulate(8302,"procedural",arm)
            self.assertEqual(m4.digest(a),m4.digest(b))
            self.assertEqual(a["food_consumed"],sum(
                t["food"] for t in a["trajectory"] if "action" in t
            ))
            self.assertEqual(a["steps_alive"],sum(
                "action" in t for t in a["trajectory"]
            ))


if __name__=="__main__":
    unittest.main()
