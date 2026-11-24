"""M3 predeclared causal one-step-history, same-14-weights, blinded-ecology tests."""
import unittest

from neuroevolution import flyloop_m0 as m0
from neuroevolution import flyloop_m2 as m2
from neuroevolution import flyloop_m3 as m3


class FlyLoopM3Tests(unittest.TestCase):

    def test_protocol_exact_new_seeds_14_parameters_7_arms(self):
        self.assertEqual(m3.SCENARIOS,("baseline","procedural","relocation"))
        self.assertEqual(len(m3.ARMS),7)
        self.assertEqual(m3.SEEDS,tuple(range(7900,7932)))
        self.assertTrue(set(m3.SEEDS).isdisjoint(m2.SEEDS))
        self.assertEqual(len(m3.SEEDS)*len(m3.ARMS)*len(m3.SCENARIOS),672)
        self.assertEqual(m0.N_ACTOR+m0.N_CRITIC,14)
        self.assertEqual(m3.RESTORE,360)
        self.assertEqual(m3.EMA_DECAY,.75)
        with self.assertRaises(ValueError):
            m3.simulate(7300,"baseline","plastic_plain")
        with self.assertRaises(ValueError):
            m3.simulate(7900,"unknown","temporal_feedback")
        with self.assertRaises(ValueError):
            m3.simulate(7900,"baseline","scripted_food_search")

    def test_zero_history_first_step_and_one_step_true_outcome(self):
        w=m3.World(7900,"baseline")
        raw,critic=w.observe()
        m=m3.Memory(7900)
        a,c,h=m.encode(raw,critic,"temporal_feedback",0)
        self.assertEqual(h["last_action"],None)
        self.assertIsNone(h["source_tick"])
        self.assertTrue(all(a==raw for a,raw in zip(a,raw)))
        self.assertEqual(c,critic)

        m.observe_actual(0,1,16.75,50.,100.,{
            "collision":0,"energy":66.75,"health":100.0
        })
        aug,newcritic,h=m.encode(raw,critic,"temporal_feedback",1)
        self.assertEqual(h["source_tick"],0)
        self.assertEqual(h["last_action"],1)
        self.assertGreater(aug[1][1],0)  # candidate EAST reward memory
        self.assertEqual(aug[1][3],1)
        self.assertEqual(aug[3][3],-1)
        self.assertEqual(aug[4][3],0) # WAIT direction
        self.assertGreater(newcritic[3],0)
        self.assertTrue(all(aug[i][5]==raw[i][5] for i in range(5)))
        self.assertTrue(all(aug[i][0]==raw[i][0] for i in range(5)))

    def test_motor_only_has_no_reward_history(self):
        w=m3.World(7901,"baseline")
        raw,critic=w.observe()
        m=m3.Memory(7901)
        m.observe_actual(0,0,-5.25,55.,100.,{
            "collision":0,"energy":54.75,"health":96.
        })
        a,c,h=m.encode(raw,critic,"motor_memory",1)
        self.assertIsNone(h["source_tick"])
        self.assertEqual(h["feedback"],[0.,0.,0.])
        self.assertTrue(any(f[3]!=0 for f in a[:4]))
        self.assertTrue(all(f[1]==f[2]==f[6]==0 for f in a))
        self.assertEqual(c,critic)
        true,_,meta=m.encode(raw,critic,"temporal_feedback",1)
        self.assertEqual(meta["source_tick"],0)
        self.assertNotEqual(true[0][2],0.)

    def test_sham_never_uses_last_or_future_event(self):
        w=m3.World(7902,"baseline")
        raw,critic=w.observe()
        mem=m3.Memory(7902)
        mem.observe_actual(0,1,16.75,50.,100.,{
            "collision":0,"energy":66.75,"health":100.
        })
        _,_,h=mem.encode(raw,critic,"temporal_sham",1)
        self.assertIsNone(h["source_tick"])
        mem.observe_actual(1,3,-5.25,66.75,100.,{
            "collision":0,"energy":65.75,"health":96.
        })
        _,_,h=mem.encode(raw,critic,"temporal_sham",2)
        self.assertEqual(h["source_tick"],0)
        self.assertEqual(h["feedback"][0],mem.events[0]["positive"])
        for t in range(2,15):
            mem.observe_actual(t, t%4,-.25,65.,96.,{
                "collision":int(t%3==0),"energy":64.75,"health":96.
            })
            _,_,h=mem.encode(raw,critic,"temporal_sham",t+1)
            self.assertLessEqual(h["source_tick"],t-1)

    def test_same_initial_world_and_weights_for_all_seven_arms(self):
        for sc in m3.SCENARIOS:
            trajectories=[m3.simulate(7900,sc,arm) for arm in m3.ARMS]
            self.assertEqual(len({m3.digest(r["start_map"])
                                  for r in trajectories}),1)
            self.assertEqual(len({m3.digest(r["initial_weights"])
                                  for r in trajectories}),1)
            self.assertTrue(all(r["steps_alive"]<=480 for r in trajectories))
            self.assertTrue(all(r["steps_alive"]>0 for r in trajectories))
            self.assertTrue(all(r["initial_weights"]==r["final_weights"]
                            for r in trajectories if r["arm"] in (
                                "frozen","random_motion","feedback_frozen"
                            )))
            self.assertTrue(all(r["initial_weights"]!=r["final_weights"]
                            for r in trajectories if r["arm"] in (
                                "plastic_plain","motor_memory","temporal_feedback",
                                "temporal_sham",
                            )))
            valid,blind,causal,weights=m3._verify(trajectories)
            self.assertTrue(valid)
            self.assertTrue(blind)
            self.assertTrue(causal)
            self.assertTrue(weights)

    def test_real_relocation_has_exogenous_map_and_restores_original(self):
        a=m3.World(7905,"relocation")
        b=m3.World(7905,"relocation")
        original=set(a.food_cooldown)
        a.body.x,a.body.y=1,8
        b.body.x,b.body.y=9,5
        self.assertEqual(a.openings,m0.OLD_OPENINGS)
        a.tick=b.tick=240
        a.switch_if_due()
        b.switch_if_due()
        self.assertEqual(a.food_cooldown,b.food_cooldown)
        self.assertFalse(original & set(a.food_cooldown))
        self.assertEqual(len(a.food_cooldown),16)
        self.assertEqual(a.openings,m0.OLD_OPENINGS)
        a.tick=b.tick=360
        a.switch_if_due()
        b.switch_if_due()
        self.assertEqual(set(a.food_cooldown),original)
        self.assertEqual([e["event"] for e in a.events],
                         ["relocate_food","restore_food"])

    def test_blind_inputs_not_leaking_represented_contact(self):
        for scenario in m3.SCENARIOS:
            row=m3.simulate(7903,scenario,"temporal_feedback")
            self.assertTrue(row["trajectory"])
            for step in row["trajectory"]:
                if "action" not in step:
                    continue
                self.assertTrue(all(
                    f[i]==0 for f in step["raw_blind_action_inputs"]
                    for i in (1,2,3,4,6)
                ))
                self.assertEqual(step["raw_blind_critic_inputs"][3],0.)
                self.assertEqual(step["raw_blind_critic_inputs"][4],0.)
                t=step["step"]
                src=step["memory_used"]["source_tick"]
                self.assertTrue(src is None or src<t)
                self.assertEqual(len(step["effective_actor_inputs"]),5)
                self.assertTrue(all(len(x)==8 for x in
                                    step["effective_actor_inputs"]))

    def test_baseline_three_legacy_arms_exactly_m2_on_old_seeds(self):
        self.assertTrue(m3.reference_m2_baseline_check())

    def test_deterministic_full_episode(self):
        for arm in m3.ARMS:
            a=m3.simulate(7902,"procedural",arm)
            b=m3.simulate(7902,"procedural",arm)
            self.assertEqual(m3.digest(a),m3.digest(b))
            if arm=="random_motion":
                self.assertTrue(all(
                    t["action"]!=4 for t in a["trajectory"]
                    if "action" in t
                ))


if __name__=="__main__":
    unittest.main()
