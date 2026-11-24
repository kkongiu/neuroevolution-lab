"""Regression and causal tests for FlyLoop R2 (prospectively frozen)."""
import copy
import math
import unittest

from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r2 as r2


class R2Tests(unittest.TestCase):

    def test_protocol_has_288_lifetimes_and_thirtyfour_motor_weights(self):
        self.assertEqual(r2.SEEDS, tuple(range(9400,9424)))
        self.assertTrue(set(r2.SEEDS).isdisjoint(r1.SEEDS))
        self.assertEqual(r2.SCENES,("obstacles","nearfield"))
        self.assertEqual(len(r2.ARMS),6)
        self.assertEqual(len(r2.SEEDS)*len(r2.SCENES)*len(r2.ARMS),288)
        self.assertEqual(r2.N_MOTOR_COEFF*2,34)
        self.assertEqual(r1.PARAMS,434)
        self.assertEqual(r2.WINDOW,12)
        self.assertEqual(r2.EXPLORATION_SD,.12)
        with self.assertRaises(ValueError):
            r2.World(9100,"nearfield")
        with self.assertRaises(ValueError):
            r2.World(9400,"food")
        with self.assertRaises(ValueError):
            r2.MotorLearner(9400,"planned_avoid")

    def test_world_geometry_exogenous_and_continuous_physics_R1(self):
        for seed in r2.SEEDS:
            a=r2.World(seed,"nearfield")
            b=r2.World(seed,"nearfield")
            self.assertEqual(a.rectangles,b.rectangles)
            self.assertEqual(len(a.rectangles),10)
            self.assertFalse(a.blocked(a.body.x,a.body.y))
            for x1,y1,x2,y2 in a.rectangles:
                self.assertTrue(a.blocked((x1+x2)/2,(y1+y2)/2))
            self.assertEqual(a.perceive(),b.perceive())
        self.assertEqual(len({
            r2.sha(r2.nearfield_rectangles(seed))
            for seed in r2.SEEDS
        }),24)

    def test_noise_matched_prefix_and_neural_coefficients_equal(self):
        brains=[
            r2.MotorLearner(9400,arm) for arm in
            ("recurrent_frozen","recurrent_plastic","recurrent_sham",
             "collision_only","leaky_plastic")
        ]
        self.assertEqual(len({r2.sha(c.initial_coefficients)
                              for c in brains}),1)
        sample=(1.,.4,.5,.6,0.,0.,0.,0.)
        sequences=[]
        for c in brains:
            seq=[]
            for tick in range(25):
                raw,h,inp,noise,command=c.command(sample)
                seq.append(noise)
            sequences.append(seq)
        self.assertTrue(all(seq==sequences[0] for seq in sequences))
        self.assertTrue(all(len(c.eligibility)==2 for c in brains))

    def test_feedback_after_actual_physics_and_exact_lag(self):
        brain=r2.MotorLearner(9400,"recurrent_plastic")
        initial=copy.deepcopy(brain.initial_coefficients)
        pose=[(2.,2.)]
        for t in range(15):
            pose.append((2.+(.1*(t+1)),2.))
            fb=brain.adapt(
                t,.1,0,pose,[.5]*16,(.1,-.1),
            )
            self.assertLessEqual(fb["physical_progress"],1.)
            self.assertAlmostEqual(
                fb["physical_progress"],
                (.1 if t>=12 else .1)/(r1.V_MAX*r1.DT),
                places=7,
            )
            self.assertEqual(fb["feedback_source_tick"],t)
        final=brain.circuit.weights
        self.assertNotEqual(initial["w_motor"],final["w_motor"])
        self.assertEqual(initial["w_input"],final["w_input"])
        self.assertEqual(initial["w_recurrent"],final["w_recurrent"])
        self.assertEqual(initial["bias"],final["bias"])
        self.assertEqual(brain.updates,15)

    def test_collision_feedback_and_sham_never_uses_current(self):
        brain=r2.MotorLearner(9401,"recurrent_sham")
        locations=[(1.,1.)]
        for t in range(15):
            locations.append((1.+.1*(t+1),1.))
            report=brain.adapt(
                t,.1,int(t%4==0),locations,[.4]*16,(.1,.05)
            )
            source=report["feedback_source_tick"]
            self.assertTrue(source is None or source<=t-5)
            if t>=5:
                self.assertIsNotNone(source)
            if source is not None:
                self.assertAlmostEqual(
                    report["training_feedback"],
                    brain.real_feedback_history[source],places=7,
                )
        noc=r2.MotorLearner(9401,"collision_only")
        places=[(1.,1.),(1.,1.)]
        p=noc.adapt(0,0.,1,places,[0.]*16,(.1,.2))
        self.assertEqual(p["training_feedback"],-1.2)
        self.assertEqual(p["actual_feedback"],-1.2)

    def test_exactly_recurrent_or_leaky_structure_without_extra_input(self):
        for arm in r2.ARMS:
            c=r2.MotorLearner(9402,arm)
            self.assertEqual(len(r1.coefficient_flat(c.initial_coefficients)),434)
            w=r2.World(9402,"obstacles")
            signal=w.perceive()
            raw,state,sensors,noise,command=c.command(signal)
            self.assertEqual(len(state),16)
            self.assertEqual(len(sensors),8)
            self.assertEqual(len(command),2)
            self.assertEqual(len(raw),2)
            self.assertTrue(all(-1<=v<=1 for v in command))
            if arm=="inertial_random":
                self.assertTrue(all(x==0 for x in state))
                self.assertEqual(noise,(0.,0.))
            else:
                self.assertNotEqual(state,tuple([0.]*16))

    def test_each_arm_deterministic_and_valid_trajectory(self):
        for arm in r2.ARMS:
            for scene in r2.SCENES:
                a=r2.simulate(9403,scene,arm)
                b=r2.simulate(9403,scene,arm)
                self.assertEqual(r2.sha(a),r2.sha(b))
                self.assertGreater(a["steps_alive"],0)
                self.assertLessEqual(a["steps_alive"],600)
                checks=r2._audit([a])
                self.assertTrue(all(checks.values()),(arm,scene,checks))
                self.assertEqual(a["update_events"],
                                 0 if arm in ("recurrent_frozen",
                                               "inertial_random")
                                 else a["steps_alive"])
                self.assertEqual(a["initial_coefficients"]["w_input"],
                                 a["final_coefficients"]["w_input"])
                self.assertEqual(a["initial_coefficients"]["w_recurrent"],
                                 a["final_coefficients"]["w_recurrent"])


if __name__=="__main__":
    unittest.main()
