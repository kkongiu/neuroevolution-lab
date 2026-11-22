"""M5.27 invariants: unseen cues, no oracle leakage, bit-level compression."""
import unittest

from neuroevolution.m5_27_rule_compression import (
    SEEDS,STRESS_SEEDS,ARMS,STRESS_ARMS,CONTEXT,
    PHASES,TRAIN_STEPS,NEURAL_PARAMS,
    AffineCompiler,SingleCuePrior,EpisodeStore,OnlineNeural,
    Learner,rank_gf2,parity,affine_answer,environment_success,
    training_split,hidden_rule_triplet,generate_phase_cues,
    frozen_exam,simulate,check_seed,
)


class RuleCompressionTests(unittest.TestCase):
    def test_seed_firewall_rank_and_true_unseen_split(self):
        self.assertEqual(SEEDS,tuple(range(5100,5124)))
        self.assertEqual(STRESS_SEEDS,tuple(range(5200,5212)))
        self.assertFalse(set(SEEDS)&set(STRESS_SEEDS))
        self.assertEqual(NEURAL_PARAMS,265)
        self.assertEqual(sum(n for _,n in PHASES),TRAIN_STEPS)
        for seed in (5100,5101,5109,5123):
            for ctx in range(3):
                seen,unseen=training_split(seed,ctx,False)
                self.assertEqual(len(seen),8)
                self.assertEqual(len(unseen),8)
                self.assertFalse(set(seen)&set(unseen))
                self.assertEqual(set(seen)|set(unseen),set(range(16)))
                self.assertEqual(
                    rank_gf2(1|(cue<<1) for cue in seen),5)
                stream=generate_phase_cues(seed,ctx,ctx,320,False)
                self.assertEqual(set(stream),set(seen))
                self.assertEqual(len(stream),320)
                self.assertFalse(set(stream)&set(unseen))
        for bad in (5000,5124,5200,161000,5100.0):
            with self.assertRaises(ValueError):check_seed(bad)
        for bad in (5100,5200.0,5212,162000):
            with self.assertRaises(ValueError):check_seed(bad,True)

    def test_streaming_compiler_learns_true_function_from_own_action_only(self):
        for seed in (5100,5101,5110,5123):
            targets=hidden_rule_triplet(seed,False)
            self.assertEqual(targets[1],targets[0]^1)
            for ctx,truth in enumerate(targets):
                model=AffineCompiler()
                seen,heldout=training_split(seed,ctx,False)
                for index,cue in enumerate(seen):
                    chosen=index%2
                    observed=environment_success(truth,cue,chosen,False)
                    model.observe(cue,chosen,observed)
                self.assertEqual(model.code,truth)
                self.assertEqual(model.independent_equations,5)
                self.assertEqual(model.persistent_rule_bits(),5)
                self.assertEqual(model.transient_basis_bits(),0)
                self.assertEqual(model.violations,0)
                for cue in seen+heldout:
                    self.assertEqual(model.predict(cue),
                                     affine_answer(truth,cue))
                for cue in seen:
                    model.observe(cue,1,
                                  environment_success(truth,cue,1,False))
                self.assertEqual(model.code,truth)
                self.assertEqual(model.violations,0)

    def test_inductive_bias_is_a_prior_not_general_learning(self):
        model=SingleCuePrior()
        self.assertEqual(len(model.hypotheses),10)
        self.assertTrue(all((r>>1).bit_count()<=1
                            for r in model.hypotheses))
        # Real environment has interaction outside affine class.
        for cue in range(16):
            # Uses only a chosen action and the environment's feedback.
            chosen=cue%2
            reward=environment_success(0,cue,chosen,True)
            model.observe(cue,chosen,reward)
        self.assertEqual(model.events,16)
        self.assertTrue(any(model.predict(c)!=
                            (environment_success(0,c,1,True))
                            for c in range(16)))

    def test_literal_memories_cannot_recover_unseen_cue_without_prior(self):
        memory=EpisodeStore(8)
        truth=hidden_rule_triplet(5100,False)[0]
        seen,held=training_split(5100,0,False)
        for cue in seen:
            action=0
            reward=environment_success(truth,cue,action,False)
            memory.observe(0,cue,action,reward)
        self.assertEqual(len(memory.records),8)
        for cue in seen:
            self.assertEqual(memory.predict(0,cue),
                             affine_answer(truth,cue))
        for cue in held:
            self.assertEqual(memory.predict(0,cue),0)
        # Real new observations are evicted once global budget is full.
        memory.observe(1,held[0],1,1)
        self.assertEqual(len(memory.records),8)
        self.assertEqual(memory.evictions,1)

    def test_neural_uses_action_conditioned_reward_only(self):
        brain=OnlineNeural(5100)
        self.assertEqual(len(brain.weights),NEURAL_PARAMS)
        before=tuple(brain.weights)
        brain.observe(0,5,1,0)
        self.assertEqual(brain.updates,1)
        self.assertNotEqual(tuple(brain.weights),before)
        for cue in range(16):
            self.assertIn(brain.predict(0,cue),(0,1))

    def test_frozen_exam_and_single_complete_life(self):
        for arm in ARMS:
            learner=Learner(5100,arm,False)
            before=learner.state()
            score=frozen_exam(
                learner,0,5100,"heldout",False,
                oracle_rule=hidden_rule_triplet(5100,False)[0])
            self.assertTrue(0<=score<=1)
            self.assertEqual(before,learner.state())
            run=simulate(5100,arm,False)
            self.assertEqual(run["lived_actions"],TRAIN_STEPS)
            self.assertTrue(run[
                "environment_truth_not_passed_to_nonoracle_learner"])
            self.assertTrue(run["frozen_exams_include_no_feedback_or_updates"])
            if arm=="oracle_rule":
                self.assertEqual(run["online_updates"],0)
                self.assertEqual(run["joint_unseen_pre_refresher"],1)
            else:
                self.assertEqual(run["online_updates"],TRAIN_STEPS)
            if arm=="concept_compressor":
                self.assertEqual(run["code_bits_final"],15)
                self.assertEqual(run["transient_equation_bits_final"],0)
                self.assertEqual(run["joint_unseen_pre_refresher"],1)
            if arm=="episodic8":
                self.assertEqual(run["episodic_capacity"],8)
                self.assertLessEqual(run["episodic_stored_records"],8)
        with self.assertRaises(ValueError):
            Learner(5100,"fantasy_oracle",False)
        with self.assertRaises(ValueError):
            AffineCompiler().observe(0,0,3)
        # Out-of-family counterexample may be outside the built-in prior.
        off=simulate(5200,"concept_compressor",True)
        self.assertTrue(off["stress_nonaffine"])
        self.assertEqual(off["code_bits_final"],15)


if __name__=="__main__":
    unittest.main()
