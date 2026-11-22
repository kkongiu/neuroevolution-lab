"""M5.22: two neural populations, memory/director and alternation controls."""
import unittest

from neuroevolution.m5_22_two_populations import (
    SEEDS, ARMS, PHASES, BLOCK, EXAM, TRAIN_STEPS,
    PARAMS, GATE_OFFSET, MEMORY_CAP, TwoPopulationBrain,
    cue_sequence, reward_world, sensory, frozen_exam,
    schedules, run, check_seed
)


class TwoPopulationTests(unittest.TestCase):
    def test_expert_architecture_and_balanced_conflicting_tasks(self):
        self.assertEqual(PARAMS,154)
        self.assertEqual(GATE_OFFSET,146)
        self.assertEqual(len(SEEDS),24)
        self.assertEqual(set(ARMS),{
            "joint","memory_director","memory_phase_bound",
            "swapped_phase_bound","alternating",
            "alternating_yoked_random","replay_reference"})
        for phase,(_,n) in enumerate(PHASES):
            pairs=cue_sequence(4600,phase,n)
            self.assertEqual(len(pairs),n)
            for i in range(0,n,BLOCK):
                self.assertEqual(set(pairs[i:i+BLOCK]),
                                 {(0,0),(0,1),(1,0),(1,1)})
                self.assertEqual(pairs[i:i+BLOCK].count((0,0)),10)
        for x in (0,1):
            for y in (0,1):
                self.assertNotEqual(reward_world(0,x,y,0),
                                    reward_world(1,x,y,0))
                self.assertEqual(len(sensory(x,y,2,1)),8)

    def test_same_founders_reward_learned_gate_and_frozen_exams(self):
        founders=[TwoPopulationBrain(4600,a) for a in ARMS]
        self.assertTrue(all(b.founder==founders[0].founder
                            for b in founders))
        for brain in founders:
            p0=brain._predict(0,1,0,0)[0]
            a=brain.choose(0,1,0,0,False)
            received=reward_world(0,0,1,a)
            brain.lived_action(0,1,0,a,received,(True,True))
            self.assertEqual(brain.memory[0][0],(0,1,0,a,received))
            self.assertEqual(brain.lived,1)
            self.assertTrue(0<=p0<=1)
            self.assertEqual(brain.gate_update_calls,
                             5 if brain.arm=="replay_reference" else 1)
            self.assertEqual(brain.population_update_calls,
                             [5,5] if brain.arm=="replay_reference" else [1,1])
            state=brain.snapshot()
            s,g=frozen_exam(brain,4600,0,20)
            self.assertTrue(0<=s<=1)
            self.assertTrue(0<=g<=1)
            self.assertEqual(state,brain.snapshot())

    def test_autonomous_freezing_is_success_driven_not_phase_oracle(self):
        brain=TwoPopulationBrain(4600,"memory_director")
        for _ in range(5):
            brain.lived=160
            brain.update_autonomous_freeze(.95)
        self.assertIsNone(brain.freeze_step)
        brain.lived=200
        brain.update_autonomous_freeze(.95)
        self.assertEqual(brain.freeze_step,200)
        self.assertEqual(brain.frozen_group,0)
        state=brain.snapshot()
        brain.update_autonomous_freeze(.20)
        self.assertEqual(brain.freeze_step,200)
        self.assertEqual(brain._allowed(200,0,5,schedules(4600)),
                         (False,True))
        self.assertNotEqual(brain.snapshot(),state)
        other=TwoPopulationBrain(4600,"memory_director")
        other.lived=400
        other.update_autonomous_freeze(.90)
        other.update_autonomous_freeze(.89)
        self.assertIsNone(other.freeze_step)

    def test_alternation_exact_population_budget_and_yoked_random(self):
        for seed in (4600,4601,4602):
            choices=schedules(seed)
            for phase in range(1,len(PHASES)):
                self.assertEqual(choices[phase].count(0),
                                 choices[phase].count(1))
            for arm in ("alternating","alternating_yoked_random"):
                b=TwoPopulationBrain(seed,arm)
                exposure=[0,0]
                for phase,(_,n) in enumerate(PHASES):
                    for block in range(n//BLOCK):
                        tick=sum(size for _,size in PHASES[:phase])+block*BLOCK
                        allowed=b._allowed(tick,phase,block,choices)
                        for i,active in enumerate(allowed):
                            if active:exposure[i]+=BLOCK
                self.assertEqual(exposure,[960,960])

    def test_full_reproducible_trajectory_and_stop_gate(self):
        for arm in ARMS:
            r=run(4600,arm)
            self.assertEqual(r,run(4600,arm))
            self.assertEqual(r["lived_experiences"],TRAIN_STEPS)
            self.assertEqual(r["memory_slots"],MEMORY_CAP)
            self.assertTrue(r["frozen_probes"])
            self.assertTrue(r["correct_action_labels_not_seen"])
            if arm=="replay_reference":
                self.assertEqual(r["gradient_update_calls"],TRAIN_STEPS*5)
                self.assertEqual(r["historical_replay_updates"],TRAIN_STEPS*4)
            else:
                self.assertEqual(r["gradient_update_calls"],TRAIN_STEPS)
                self.assertEqual(r["historical_replay_updates"],0)
            if arm in ("alternating","alternating_yoked_random"):
                self.assertEqual(r["population_update_calls"],[960,960])
        for invalid in (4500,4624,4200,161000,163000,4600.0):
            with self.assertRaises(ValueError):check_seed(invalid)
        with self.assertRaises(ValueError):
            TwoPopulationBrain(4600,"supervised_labels")


if __name__=="__main__":
    unittest.main()
