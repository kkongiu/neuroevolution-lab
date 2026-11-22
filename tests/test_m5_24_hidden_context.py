"""M5.24: hidden world identity, lived-feedback causal inference and frozen probes."""
import unittest

from neuroevolution.m5_24_hidden_context import (
    SEEDS,ARMS,PHASE_NAMES,BLOCK,PROBE,EXAM,
    MEMORY_LIMIT,MAX_BANKS,LatentBank,Organism,check_seed,schedule,
    observations,environmental_success,frozen_exam,run,
)


class HiddenContextTests(unittest.TestCase):
    def test_world_schedule_is_hidden_and_three_rules_conflict(self):
        self.assertEqual(len(SEEDS),24)
        self.assertEqual(PHASE_NAMES,("A","B","C","A","B"))
        self.assertEqual(len(ARMS),8)
        self.assertEqual(MEMORY_LIMIT,128)
        self.assertEqual(MAX_BANKS,3)
        for seed in SEEDS[:4]:
            phases=schedule(seed)
            self.assertEqual(tuple(x for x,_ in phases),PHASE_NAMES)
            self.assertEqual(len(phases),5)
            for phase,(name,n) in enumerate(phases):
                self.assertEqual(n%BLOCK,0)
                samples=observations(seed,phase,n)
                self.assertEqual(len(samples),n)
                for start in range(0,n,BLOCK):
                    block=samples[start:start+BLOCK]
                    self.assertEqual(set(block),{
                        (0,0),(0,1),(1,0),(1,1)})
                    self.assertEqual(block.count((0,0)),10)
        for x in (0,1):
            for y in (0,1):
                self.assertNotEqual(
                    environmental_success("A",x,y,0),
                    environmental_success("B",x,y,0))

    def test_memory_estimate_from_real_chosen_action_only(self):
        memory=LatentBank(42)
        self.assertEqual(memory.response(1,1,0),(.5,0))
        for _ in range(9):
            # Observed success of action 1 logically implies action 0
            # would have failed under the explicitly deterministic
            # two-action complementary-success assumption.
            memory.append((1,1,1,1))
        p_zero,n=memory.response(1,1,0)
        p_one,_=memory.response(1,1,1)
        self.assertLess(p_zero,.1)
        self.assertGreater(p_one,.9)
        self.assertEqual(n,9)
        for _ in range(100):
            memory.append((0,0,0,1))
        self.assertEqual(len(memory.records),42)

    def test_no_hidden_agent_accepts_world_label(self):
        for arm in ARMS:
            model=Organism(4800,arm)
            if arm=="oracle_context":
                self.assertEqual(model.current_bank(1),1)
                with self.assertRaises(ValueError):
                    model.current_bank()
            else:
                with self.assertRaises(AssertionError):
                    model.current_bank(ctx_for_oracle=0)
                with self.assertRaises(AssertionError):
                    model.choose(
                        0,1,0,training=True,context_for_oracle=2)
                with self.assertRaises(AssertionError):
                    model.experience(
                        0,1,0,1,context_for_oracle=2)

    def test_evidence_router_uses_recent_actions_not_labels(self):
        brain=Organism(4800,"latent_director")
        first=brain.banks[0]
        # Fully remembered rule A based only on actually experienced
        # reward records. Fixture constrains desired bank readout,
        # never passed into the real training routine.
        for x in (0,1):
            for y in (0,1):
                for _ in range(7):
                    first.append((x,y,0,environmental_success("A",x,y,0)))
        known=[
            (x,y,0,environmental_success("A",x,y,0))
            for x in (0,1) for y in (0,1) for _ in range(4)]
        foreign=[
            (x,y,0,environmental_success("B",x,y,0))
            for x in (0,1) for y in (0,1) for _ in range(4)]
        self.assertEqual(brain._bank_from_history(
            known,novelty_allowed=True),0)
        self.assertEqual(brain._bank_from_history(
            foreign,novelty_allowed=True),1)
        self.assertEqual(brain.select_from_probe(known),0)

    def test_real_experience_archive_and_frozen_working_context(self):
        for arm in ARMS:
            brain=Organism(4800,arm)
            x,y=0,1
            chosen=brain.choose(
                x,y,0,training=True,
                context_for_oracle=0 if arm=="oracle_context" else None)
            feedback=environmental_success("A",x,y,chosen)
            brain.experience(
                x,y,chosen,feedback,
                context_for_oracle=0 if arm=="oracle_context" else None)
            self.assertEqual(brain.lived,1)
            self.assertEqual(brain.update_calls,1)
            self.assertEqual(brain.fast_updates,1)
            self.assertEqual(len(brain.banks[0].records),1)
            self.assertEqual(brain.banks[0].records[0],
                             (x,y,chosen,feedback))
            before=brain.snapshot()
            acc,selected=frozen_exam(
                brain,4800,"A",20,priming=PROBE)
            cold,_=frozen_exam(
                brain,4800,"A",20,priming=0)
            self.assertTrue(0<=acc<=1)
            self.assertTrue(0<=cold<=1)
            self.assertEqual(before,brain.snapshot())
            with self.assertRaises(ValueError):
                frozen_exam(brain,4800,"A",20,priming=1)

    def test_small_life_reproducible_and_no_evolution(self):
        for arm in ARMS:
            row=run(4800,arm)
            self.assertEqual(row,run(4800,arm))
            self.assertEqual(row["lived_experiences"],
                             sum(n for _,n in schedule(4800)))
            self.assertEqual(row["gradient_update_calls"],
                             row["lived_experiences"])
            self.assertLessEqual(row["final_memory_size"],MEMORY_LIMIT)
            self.assertLessEqual(row["final_bank_count"],MAX_BANKS)
            self.assertEqual(row["exam_priming_real_feedback_count"],16)
            self.assertTrue(row["exam_only_ephemeral_working_state"])
            self.assertTrue(row["no_teacher_oracle_action_labels"])
            if arm!="oracle_context":
                self.assertTrue(row["no_world_id_passed_to_hidden_arm"])

    def test_original_and_other_development_cohorts_blocked(self):
        for invalid in (4700,4824,161000,163000,4800.0,4300):
            with self.assertRaises(ValueError):check_seed(invalid)
        with self.assertRaises(ValueError):
            Organism(4800,"answer_oracle")


if __name__=="__main__":
    unittest.main()
