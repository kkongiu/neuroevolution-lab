"""R9 tests: actual physics substeps, event response, sham provenance and controls."""
import copy
import math
import unittest

from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r2 as r2
from neuroevolution import flyloop_r8 as r8
from neuroevolution import flyloop_r9 as r9


class R9Tests(unittest.TestCase):
    def test_preregistered_dimensions_and_genuine_independent_time_scale(self):
        self.assertEqual(r9.SEEDS,tuple(range(11500,11512)))
        self.assertTrue(set(r9.SEEDS).isdisjoint(r8.SEEDS))
        self.assertEqual(r9.GEOMETRIES,("obstacles","nearfield"))
        self.assertEqual(r9.REGIMES,("static","removal"))
        self.assertEqual(len(r9.ARMS),5)
        self.assertEqual(12*2*2*5,240)
        self.assertEqual(r9.SUBSTEPS,4)
        self.assertAlmostEqual(r9.MICRO_DT,.05)
        self.assertAlmostEqual(1-(1-r9.MICRO_ALPHA)**4,r1.MOTOR_ALPHA)
        self.assertEqual(r9.FAST_NEURONS,4)
        self.assertEqual(r9.FAST_INPUTS,7)
        self.assertEqual(r9.WARMUP_MACROS,20)
        self.assertEqual(r9.SWITCH_MACRO,140)
        self.assertEqual(r9.SHAM_MIN_AGE,12)
        self.assertEqual(r1.N,16)
        self.assertEqual(r1.PARAMS,434)
        with self.assertRaises(ValueError):
            r9.World(11200,"nearfield","removal")
        with self.assertRaises(ValueError):
            r9.World(11500,"food","removal")
        with self.assertRaises(ValueError):
            r9.simulate(11500,"nearfield","removal","attack")

    def test_true_micro_integration_has_proportional_real_injury_energy(self):
        w=r9.World(11500,"obstacles","static")
        w.body.x=.301
        w.body.y=10.
        w.body.heading=math.pi
        w.body.speed=1.
        last=w.body.integrity
        last_energy=w.body.energy
        f=w.micro_integrate((.5,0.))
        self.assertEqual(w.frame,1)
        self.assertEqual(f["collision"],1)
        self.assertEqual(f["position"],[.301,10.])
        self.assertAlmostEqual(last-f["integrity"],r1.COLLISION_INJURY/4)
        self.assertAlmostEqual(last_energy-f["energy"],r1.ENERGY_PER_TICK/4)
        self.assertTrue(w.perceive()[6]==1)
        self.assertEqual(r9.local_fast_sensors(w.perceive())[3],1.)
        self.assertTrue(r9.is_triggered(r9.local_fast_sensors(w.perceive())))

    def test_fast_network_has_four_units_and_ten_adaptable_outputs(self):
        f=r9.FastCircuit(11500)
        self.assertEqual(len(f.h),4)
        self.assertEqual(len(f.input),4)
        self.assertTrue(all(len(x)==7 for x in f.input))
        self.assertEqual(len(f.recurrent),4)
        self.assertTrue(all(len(x)==4 for x in f.recurrent))
        self.assertEqual(len(f.output),2)
        self.assertTrue(all(len(x)==5 for x in f.output))
        before=copy.deepcopy(f.output)
        state=f.observe((.9,.5,.2,1.,.25,-.1,.1))
        self.assertEqual(len(state),4)
        motor,noise,over=f.act((.2,.1),"fast_real",True)
        self.assertTrue(over)
        self.assertTrue(all(-1<=v<=1 for v in motor))
        self.assertEqual(f.output,before) # never learns before actual feedback
        source,train=f.learn(0,"fast_real",True,noise,-1.2)
        self.assertIsNone(source)
        self.assertEqual(train,-1.2)
        self.assertEqual(f.update_events,1)
        self.assertNotEqual(f.output,before)
        self.assertEqual(f.output_initial,before)

    def test_sham_uses_genuinely_older_real_physical_reward(self):
        f=r9.FastCircuit(11501)
        f.observe((.2,.3,.4,1.,.2,.0,.2))
        actual=[]
        for i in range(40):
            reward=float(i)/50.
            source,train=f.learn(
                i,"fast_old_sham",True,(.1,-.15),reward
            )
            if i<12:
                self.assertIsNone(source)
                self.assertIsNone(train)
            else:
                self.assertTrue(0<=source<=i-12)
                self.assertEqual(train,actual[source])
            actual.append(reward)
        self.assertEqual(f.update_events,28)
        self.assertEqual(f.actual_reward_history,actual)

    def test_slow_only_does_not_actuate_fast_and_original_neurons_frozen(self):
        a=r9.simulate(11500,"nearfield","removal","slow_only")
        self.assertEqual(a["micro_steps_executed"],1120)
        self.assertEqual(a["fast_learning_update_events"],0)
        self.assertEqual(a["initial_fast_output"],a["final_fast_output"])
        self.assertEqual(
            a["original_434_slow_neural_coefficients_initial"],
            a["original_434_slow_neural_coefficients_final"],
        )
        self.assertTrue(all(
            not f["fast_actually_overrode_motor"] for f in a["raw_micro_trace"]
        ))

    def test_all_five_controls_have_exact_same_initial_warmup_microphysics(self):
        rows=[
            r9.simulate(11501,"nearfield","static",arm)
            for arm in r9.ARMS
        ]
        baseline=[
            (f["actual_executed_micro_motor"],f["real_physics"])
            for f in rows[0]["raw_micro_trace"][:20*4]
        ]
        for row in rows:
            self.assertEqual(
                [
                    (f["actual_executed_micro_motor"],f["real_physics"])
                    for f in row["raw_micro_trace"][:80]
                ],baseline
            )
            self.assertEqual(row["initial_fast_output"],
                             rows[0]["initial_fast_output"])
            self.assertEqual(
                row["original_434_slow_neural_coefficients_initial"],
                rows[0]["original_434_slow_neural_coefficients_initial"],
            )

    def test_exact_environment_match_pre_switch_and_fixed_removal(self):
        for arm in r9.ARMS:
            a=r9.simulate(11502,"nearfield","static",arm)
            b=r9.simulate(11502,"nearfield","removal",arm)
            self.assertEqual(
                a["raw_micro_trace"][:140*4],
                b["raw_micro_trace"][:140*4],
            )
            self.assertEqual(a["final_rectangles"],a["initial_ten_rectangles"])
            self.assertEqual(
                b["final_rectangles"],
                b["initial_ten_rectangles"][::2],
            )

    def test_bypass_starts_at_next_real_microstep_without_macro_recompute(self):
        life=r9.simulate(11503,"nearfield","removal","fast_real")
        raw=life["raw_micro_trace"]
        self.assertEqual(len(raw),1120)
        same_macro_contacts=[
            (a,b) for a,b in zip(raw,raw[1:])
            if a["macro_tick"]>=20 and a["real_physics"]["collision"]
            and b["macro_tick"]==a["macro_tick"]
        ]
        self.assertTrue(same_macro_contacts)
        for a,b in same_macro_contacts:
            self.assertTrue(b["fast_salience_detected"])
            self.assertTrue(b["fast_actually_overrode_motor"])
            self.assertEqual(b["micro_tick"],a["micro_tick"]+1)
            self.assertEqual(b["slow_sixteen_neural_state"],
                             a["slow_sixteen_neural_state"])
        self.assertGreater(life["fast_learning_update_events"],0)

    def test_reruns_reproduce_actual_lives_and_internal_physics(self):
        sample=[
            r9.simulate(11504,"obstacles","removal","fast_real"),
            r9.simulate(11504,"nearfield","static","fast_old_sham"),
            r9.simulate(11504,"nearfield","removal","fast_fixed"),
        ]
        self.assertEqual(
            r9.sha(sample[0]),
            r9.sha(r9.simulate(11504,"obstacles","removal","fast_real")),
        )
        report=r9._structural_audit(sample)
        self.assertTrue(all(report.values()),report)
        for r in sample:
            self.assertEqual(len(r["raw_micro_trace"]),r["micro_steps_executed"])
            self.assertTrue(r["postchange_micro_steps"]<=140*4)


if __name__=="__main__":
    unittest.main()
