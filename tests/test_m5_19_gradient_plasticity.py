"""M5.19: gradients, explicit outcome-only memories, and stop/go discipline."""
import math
import unittest
from neuroevolution.m5_19_gradient_plasticity import (
    ARMS,SEEDS,PHASES,PARAMS,TRAIN_STEPS,MEMORY_CAP,
    REFERENCE_PATTERNS,Brain,valid,trials,inputs,environment_reward,
    dot,projected_gradient,frozen_exam,run,experiment
)


class M519Tests(unittest.TestCase):
    def test_first_order_projection_and_no_op_when_compatible(self):
        new=[-2.,1.,3.]
        old=[[1.,0.,0.],[0.,1.,0.]]
        projected,count=projected_gradient(new,old)
        self.assertGreater(count,0)
        self.assertGreaterEqual(dot(projected,old[0]),-1e-12)
        self.assertGreaterEqual(dot(projected,old[1]),-1e-12)
        self.assertEqual(projected[2],3.)
        compat=[1.,2.,3.]
        g,c=projected_gradient(compat,old)
        self.assertEqual(g,compat)
        self.assertEqual(c,0)

    def test_same_founders_sensory_cues_and_context_conflict(self):
        founders=[Brain(4200,a) for a in ARMS]
        self.assertEqual(len(founders),5)
        self.assertEqual(len(founders[0].weights),PARAMS)
        for b in founders:
            self.assertEqual(tuple(b.weights),tuple(founders[0].weights))
        for phase,(task,n) in enumerate(PHASES):
            pairs=trials(4200,phase,n)
            self.assertEqual(len(pairs),n)
            for k in range(0,n,40):
                block=pairs[k:k+40]
                self.assertEqual(set(block),set(REFERENCE_PATTERNS))
                self.assertEqual(block.count((0,0)),10)
        for x in (0,1):
            for y in (0,1):
                self.assertNotEqual(environment_reward(0,x,y,0),
                                    environment_reward(1,x,y,0))
                self.assertEqual(len(inputs(x,y,0,0)),8)
        self.assertEqual(MEMORY_CAP,128)

    def test_reference_gradient_from_only_experienced_actions(self):
        for arm in ARMS:
            brain=Brain(4200,arm)
            for tick in range(84):
                context=0 if tick<44 else 1
                x,y=REFERENCE_PATTERNS[tick%4]
                a=brain.choose(x,y,context,tick,True)
                result=environment_reward(context,x,y,a)
                brain.experience(x,y,context,a,result)
            self.assertEqual(brain.updates,84)
            self.assertEqual(brain.lived,84)
            self.assertEqual(brain.memory_used_for_weight_replay,0)
            self.assertGreater(brain.reference_gradients_evaluated,0)
            self.assertLessEqual(sum(len(x) for x in brain.memory),128)
            self.assertTrue(all(
                record[4] in (0,1)
                for group in brain.memory for record in group))
            before=brain.state()
            p=frozen_exam(brain,4200,0,20)
            self.assertTrue(0<=p<=1)
            self.assertEqual(before,brain.state())

    def test_local_actually_suppresses_and_no_extra_updates(self):
        projected=Brain(4200,"gradient_projection")
        local=Brain(4200,"projection_local")
        assert projected.weights==local.weights
        for tick in range(90):
            context=0 if tick<40 else 1
            x,y=REFERENCE_PATTERNS[tick%4]
            for b in (projected,local):
                a=b.choose(x,y,context,tick,True)
                reward=environment_reward(context,x,y,a)
                b.experience(x,y,context,a,reward)
        self.assertEqual(projected.updates,local.updates)
        self.assertGreater(local.local_opportunities,0)
        self.assertGreater(local.local_conflicting_coordinates,0)
        self.assertEqual(projected.local_conflicting_coordinates,0)

    def test_reproducible_results_and_sealed_cohort_firewall(self):
        for arm in ARMS:
            one=run(4200,arm)
            self.assertEqual(one,run(4200,arm))
            self.assertTrue(one["frozen_exams"])
            self.assertEqual(one["parameter_updates"],TRAIN_STEPS)
            self.assertEqual(one["reference_buffer_capacity"],MEMORY_CAP)
            self.assertEqual(one["replay_weight_updates"],0)
        result=experiment((4200,4201))
        self.assertEqual(result["development_screen_pass"],
                         all(result["screening_gate"].values()))
        self.assertFalse(result["independent_replication_complete"])
        self.assertFalse(result["evolution_authorized"])
        self.assertFalse(result["advance_to_evolution"])
        self.assertTrue(result["same_gradient_update_and_reference_compute_budget"])
        for invalid in (4100,4224,161000,162000,163000,4200.0):
            with self.assertRaises(ValueError):valid(invalid)
        with self.assertRaises(ValueError):experiment((4200,4200))

if __name__=="__main__":
    unittest.main()
