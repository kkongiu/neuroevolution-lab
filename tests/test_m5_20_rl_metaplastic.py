"""M5.20 scientific tests: meta-RL learns only from lived consequences."""
import unittest

from neuroevolution.m5_20_rl_metaplastic import (
    SEEDS,ARMS,PHASES,TRAIN_STEPS,EXAM,MEMORY_LIMIT,SLOTS,STRENGTHS,
    GROUPS,PARAMS,Controller,Brain,valid,trials,inputs,group,
    environment_reward,frozen_exam,run,experiment
)


class MetaRLTests(unittest.TestCase):
    def test_training_curriculum_and_group_membership(self):
        self.assertEqual(SEEDS,tuple(range(4300,4324)))
        self.assertEqual(sum(n for _,n in PHASES),TRAIN_STEPS)
        self.assertEqual(sum(SLOTS),MEMORY_LIMIT)
        self.assertEqual({group(i) for i in range(PARAMS)},set(range(GROUPS)))
        for phase,(_,n) in enumerate(PHASES):
            sequence=trials(4300,phase,n)
            self.assertEqual(len(sequence),n)
            for start in range(0,n,40):
                b=sequence[start:start+40]
                self.assertEqual(set(b),{(0,0),(0,1),(1,0),(1,1)})
                self.assertEqual(b.count((0,0)),10)
        for x in (0,1):
            for y in (0,1):
                self.assertNotEqual(
                    environment_reward(0,x,y,0),
                    environment_reward(1,x,y,0))
        self.assertEqual(len(inputs(1,0,2,0)),8)

    def test_policy_has_no_target_labels_and_uses_only_chosen_actions(self):
        b=Brain(4300)
        c=Controller(4300,"q_learning_controller")
        st=c.state(.5,None)
        factor,decisions=c.choose(st,None)
        self.assertEqual(len(factor),GROUPS)
        self.assertTrue(all(x in STRENGTHS for x in factor))
        x,y=0,1
        a=b.choose(x,y,0,0,True)
        reward=environment_reward(0,x,y,a)
        b.live(x,y,0,a,reward,factor)
        self.assertEqual(b.lived,1)
        self.assertEqual(b.updates,1)
        self.assertEqual(len(b.memory[0]),1)
        self.assertEqual(b.memory[0][0],(x,y,0,a,reward))
        self.assertIsNone(b.old_brier(0))
        before=b.snapshot()
        frozen_exam(b,4300,0,20)
        self.assertEqual(before,b.snapshot())

    def test_Q_bootstrapping_versus_random_negative_control(self):
        for arm in ("q_learning_controller","scrambled_reward_q",
                    "random_controller","heuristic_controller"):
            c=Controller(4300,arm)
            actions=c.choose(1,None)[1]
            self.assertEqual(len(actions),GROUPS)
            old=c.q_checksum()
            c.feedback(1,actions,2,1.0)
            if arm in ("q_learning_controller","scrambled_reward_q"):
                self.assertEqual(c.q_updates,GROUPS)
                self.assertNotEqual(c.q_checksum(),old)
            else:
                self.assertEqual(c.q_updates,0)
                self.assertEqual(c.q_checksum(),old)
        real=Controller(4300,"q_learning_controller")
        shuffled=Controller(4300,"scrambled_reward_q")
        for i in range(8):
            a=(1,2,0,3)
            real.feedback(1,a,2,.75)
            shuffled.feedback(1,a,2,.75)
        self.assertNotEqual(real.q_checksum(),shuffled.q_checksum())

    def test_budget_reproducible_and_no_evolution_gate(self):
        founders=[Brain(4300) for _ in ARMS]
        self.assertTrue(all(b.original==founders[0].original
                            for b in founders))
        rows=[]
        for arm in ARMS:
            r=run(4300,arm)
            self.assertEqual(r,run(4300,arm))
            self.assertEqual(r["lived_experiences"],TRAIN_STEPS)
            self.assertEqual(r["synaptic_updates"],TRAIN_STEPS)
            self.assertEqual(r["meta_decision_blocks"],TRAIN_STEPS//40)
            self.assertEqual(r["memory_capacity"],MEMORY_LIMIT)
            self.assertEqual(r["weight_replay_updates"],0)
            self.assertTrue(r["frozen_exams"])
            self.assertFalse(r["meta_reward_uses_exam"])
            rows.append(r)
        self.assertEqual(len({r["reference_forward_evaluations"]
                              for r in rows}),1)
        report=experiment((4300,4301))
        self.assertEqual(report["development_screen_pass"],
                         all(report["screening_gate"].values()))
        self.assertFalse(report["evolution_authorized"])
        self.assertFalse(report["independent_replication_complete"])
        self.assertFalse(report["advance_to_evolution"])

    def test_seed_firewall_and_invalid_inputs(self):
        for invalid in (4200,4324,161000,162000,163000,4300.0):
            with self.assertRaises(ValueError):valid(invalid)
        with self.assertRaises(ValueError):
            experiment((4300,4300))
        with self.assertRaises(ValueError):
            Controller(4300,"oracle")
        with self.assertRaises(ValueError):
            Brain(4300).live(0,0,0,0,2,(.1,)*4)
        with self.assertRaises(ValueError):
            Brain(4300).live(0,0,0,0,1,(2.,)*4)


if __name__=="__main__":
    unittest.main()
