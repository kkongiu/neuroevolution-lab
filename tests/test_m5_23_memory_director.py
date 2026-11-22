"""M5.23: audit separable memory write/read/director and leakage controls."""
import unittest

from neuroevolution.m5_23_memory_director import (
    SEEDS,ARMS,PHASES,TRAIN_STEPS,FAST_PARAMETERS,
    ADDRESS_PARAMETERS,DIRECTOR_DIM,TOTAL_PARAMETERS,MEMORY_CAP,
    SpecializedBrain,cues,observations,environment_reward,
    exam,run,validate_seed,
)


class M523Tests(unittest.TestCase):
    def test_common_founder_and_balanced_cues(self):
        self.assertEqual(len(SEEDS),24)
        self.assertEqual(FAST_PARAMETERS,145)
        self.assertEqual(ADDRESS_PARAMETERS,4)
        self.assertEqual(DIRECTOR_DIM,10)
        self.assertEqual(TOTAL_PARAMETERS,159)
        self.assertEqual(MEMORY_CAP,128)
        brains=[SpecializedBrain(4700,arm) for arm in ARMS]
        self.assertTrue(all(b.founder==brains[0].founder for b in brains))
        for phase,(_,n) in enumerate(PHASES):
            pairs=cues(4700,phase,n)
            self.assertEqual(len(pairs),n)
            for start in range(0,n,40):
                block=pairs[start:start+40]
                self.assertEqual(set(block),{(0,0),(0,1),(1,0),(1,1)})
                self.assertEqual(block.count((0,0)),10)
        for x in (0,1):
            for y in (0,1):
                self.assertNotEqual(
                    environment_reward(0,x,y,0),
                    environment_reward(1,x,y,0))
                self.assertEqual(len(observations(x,y,2,1)),8)

    def test_independent_archive_can_retrieve_opposite_context_rules(self):
        b=SpecializedBrain(4700,"learned_director")
        # Deliberately store only actually compatible action/outcome records.
        # These are test fixtures, not part of training and not a teacher
        # data source available to the normal learner.
        for _ in range(10):
            b.memory[0].append((0,1,0,1,1))
            b.memory[0].append((0,1,0,0,0))
            b.memory[1].append((0,1,1,0,1))
            b.memory[1].append((0,1,1,1,0))
        p_A_right=b._reader(0,1,0,1,False)[0]
        p_A_wrong=b._reader(0,1,0,0,False)[0]
        p_B_right=b._reader(0,1,1,0,False)[0]
        p_B_wrong=b._reader(0,1,1,1,False)[0]
        self.assertGreater(p_A_right,p_A_wrong)
        self.assertGreater(p_B_right,p_B_wrong)
        self.assertGreater(p_A_right,.8)
        self.assertGreater(p_B_right,.8)

    def test_explicit_context_ablated_has_conflicting_address(self):
        full=SpecializedBrain(4700,"learned_director")
        blind=SpecializedBrain(4700,"context_blind_director")
        for brain in (full,blind):
            for _ in range(13):
                brain.memory[0].append((0,1,0,1,1))
                brain.memory[0].append((0,1,0,0,0))
                brain.memory[1].append((0,1,1,0,1))
                brain.memory[1].append((0,1,1,1,0))
        full_diff=(full._reader(0,1,0,1,False)[0]
                   -full._reader(0,1,1,1,False)[0])
        blind_diff=(blind._reader(0,1,0,1,False)[0]
                    -blind._reader(0,1,1,1,False)[0])
        self.assertGreater(full_diff,.5)
        self.assertLess(abs(blind_diff),.05)
        self.assertEqual(blind.founder,full.founder)

    def test_live_write_from_own_action_after_learning_and_frozen(self):
        for arm in ARMS:
            brain=SpecializedBrain(4700,arm)
            x,y,c=0,1,0
            action=brain.choose(x,y,c,0,True)
            real_reward=environment_reward(c,x,y,action)
            self.assertEqual(len(brain.memory[0]),0)
            brain.live(x,y,c,action,real_reward)
            self.assertEqual(brain.lived,1)
            self.assertEqual(brain.memory[0][0][:4],(x,y,c,action))
            if arm!="corrupted_memory_director":
                self.assertEqual(brain.memory[0][0][4],real_reward)
            self.assertEqual(brain.gradient_update_calls,
                             5 if arm=="replay_reference" else 1)
            old=brain.frozen_state()
            for mode in ("policy","archive","fast"):
                result=exam(brain,4700,0,20,readout=mode)
                self.assertTrue(0<=result<=1)
            self.assertEqual(old,brain.frozen_state())
            self.assertEqual(brain.read_record_evaluations_exam,480)
        for arm in ("learned_director","fixed_address_director",
                    "context_blind_director"):
            b=SpecializedBrain(4700,arm)
            a=b.choose(0,1,0,0,True)
            b.live(0,1,0,a,environment_reward(0,0,1,a))
            self.assertEqual(b.director_gradient_update_calls,1)
            self.assertEqual(b.address_gradient_update_calls,
                             0 if arm=="fixed_address_director" else 1)
        with self.assertRaises(ValueError):
            SpecializedBrain(4700,"fast_only").live(0,0,0,0,2)

    def test_trajectory_is_reproducible_and_exposure_matched(self):
        for arm in ARMS:
            row=run(4700,arm)
            self.assertEqual(row,run(4700,arm))
            self.assertEqual(row["lived_actions"],TRAIN_STEPS)
            self.assertEqual(row["memory_slots"],MEMORY_CAP)
            self.assertTrue(row["frozen_exams"])
            self.assertTrue(row["no_oracle_correct_action_targets"])
            expected=TRAIN_STEPS*(5 if arm=="replay_reference" else 1)
            self.assertEqual(row["gradient_update_calls"],expected)
            self.assertEqual(row["old_episode_weight_replay_updates"],
                             4*TRAIN_STEPS if arm=="replay_reference" else 0)
        no_replay=[run(4701,arm) for arm in ARMS
                   if arm!="replay_reference"]
        self.assertEqual(len({
            r["training_read_record_evals"] for r in no_replay}),1)
        self.assertEqual(len({
            r["frozen_read_record_evals"] for r in no_replay}),1)
        self.assertEqual(len({
            r["memory_stored_records"] for r in no_replay}),1)

    def test_seed_firewall(self):
        for bad in (4600,4724,4800,161000,162000,163000,4700.0):
            with self.assertRaises(ValueError):validate_seed(bad)
        with self.assertRaises(ValueError):
            SpecializedBrain(4700,"supervised_oracle")


if __name__=="__main__":
    unittest.main()
