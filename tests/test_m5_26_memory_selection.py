"""M5.26: audit bounded long-term storage, novelty decisions, and controls."""
import unittest

from neuroevolution.m5_26_memory_selection import (
    ARMS, SEEDS, SLOTS12, SLOTS8, CAPACITY_BY_ARM, MEMORY_LIMIT,
    MAX_PENDING, BLOCK, BudgetOrganism, check_seed,
    schedule, observations, environmental_success,
    make_organism, frozen_exam, run
)


class MemorySelectionTests(unittest.TestCase):
    def test_fresh_firewall_and_equal_founders(self):
        self.assertEqual(SEEDS,tuple(range(5000,5024)))
        self.assertEqual(len(ARMS),10)
        self.assertEqual(SLOTS12,(4,4,4))
        self.assertEqual(SLOTS8,(3,3,2))
        self.assertEqual(CAPACITY_BY_ARM["reference128"],(43,43,42))
        self.assertEqual(sum(CAPACITY_BY_ARM["reference128"]),128)
        self.assertEqual(CAPACITY_BY_ARM["novelty8"],(3,3,2))
        self.assertEqual(CAPACITY_BY_ARM["novelty12"],(4,4,4))
        self.assertEqual(MAX_PENDING,12)
        for invalid in (4900,5024,4800,161000,163000,5000.0):
            with self.assertRaises(ValueError):check_seed(invalid)
        brains=[make_organism(5000,arm) for arm in ARMS]
        self.assertEqual(len({brain.founder for brain in brains}),1)
        for arm,brain in zip(ARMS,brains):
            self.assertIsInstance(brain,BudgetOrganism)
            self.assertEqual(brain.slots,CAPACITY_BY_ARM[arm])
            self.assertEqual(len(brain.banks[0].records),0)
            with self.assertRaises(AssertionError):
                brain.experience(0,0,0,1,context_for_oracle=0)
        for phase,(_,n) in enumerate(schedule(5000)):
            block=observations(5000,phase,n)
            for i in range(0,n,BLOCK):
                self.assertEqual(block[i:i+BLOCK].count((0,0)),10)

    def test_deduplicate_novelty_retains_four_actual_cue_patterns(self):
        brain=BudgetOrganism(5000,"novelty12")
        own_action=0
        for x,y in ((0,0),(0,1),(1,0),(1,1)):
            outcome=environmental_success("A",x,y,own_action)
            brain.experience(x,y,own_action,outcome)
        bank=brain.banks[0]
        initial=tuple(bank.records)
        self.assertEqual(len(initial),4)
        self.assertEqual({(x,y) for x,y,_,_ in initial},
                         {(0,0),(0,1),(1,0),(1,1)})
        for _ in range(70):
            for x,y in ((0,0),(0,1),(1,0),(1,1)):
                brain.experience(
                    x,y,own_action,
                    environmental_success("A",x,y,own_action))
        self.assertEqual(tuple(bank.records),initial)
        self.assertEqual(brain.new_cue_admissions,4)
        self.assertGreater(brain.duplicate_refusals,100)
        self.assertEqual(brain.evictions,0)

    def test_fifo_and_reservoir_evict_under_same_capacity(self):
        examples=[(x,y,0,environmental_success("A",x,y,0))
                  for x,y in ((0,0),(0,1),(1,0),(1,1))]
        a=BudgetOrganism(5000,"fifo12")
        b=BudgetOrganism(5000,"reservoir12")
        for model in (a,b):
            for e in examples:
                model._write_to_bank(0,e)
            assert len(model.banks[0].records)==4
            for _ in range(64):
                model._write_to_bank(0,examples[0])
            self.assertEqual(len(model.banks[0].records),4)
            self.assertGreater(model.per_bank_seen[0],4)
            self.assertGreater(model.evictions,0)
        self.assertEqual({(x,y) for x,y,_,_ in a.banks[0].records},
                         {(0,0)})
        self.assertEqual(b.banks[0].records.maxlen,4)

    def test_8_slot_stress_banks_can_hold_only_8_distinct_cues(self):
        brain=BudgetOrganism(5000,"novelty8")
        self.assertEqual(brain.slots,(3,3,2))
        for x,y in ((0,0),(0,1),(1,0),(1,1)):
            brain._write_to_bank(
                0,(x,y,0,environmental_success("A",x,y,0)))
        self.assertEqual(len(brain.banks[0].records),3)
        self.assertEqual(brain.new_cue_admissions,3)
        self.assertEqual(brain.denied_stable_writes,1)
        self.assertEqual(sum(brain.slots),8)

    def test_phases_and_exam_do_not_read_teacher_world_id(self):
        for arm in ARMS:
            brain=BudgetOrganism(5000,arm)
            for x,y in ((0,0),(0,1),(1,0),(1,1)):
                a=brain.choose(x,y,0,training=True)
                reward=environmental_success("A",x,y,a)
                brain.experience(x,y,a,reward)
            original=brain.snapshot()
            score,bank=frozen_exam(
                brain,5000,"A",20,priming=16)
            score_zero,bank_zero=frozen_exam(
                brain,5000,"A",20,priming=0)
            self.assertEqual(original,brain.snapshot())
            self.assertTrue(0<=score<=1)
            self.assertTrue(0<=score_zero<=1)
            self.assertTrue(0<=bank<len(brain.banks))

    def test_reproducible_full_lifetime_and_real_memory_ledger(self):
        for arm in ARMS:
            row=run(5000,arm)
            self.assertEqual(row,run(5000,arm))
            self.assertEqual(row["memory_capacity"],sum(CAPACITY_BY_ARM[arm]))
            self.assertLessEqual(row["final_memory_size"],sum(CAPACITY_BY_ARM[arm]))
            self.assertLessEqual(row["guarded_pending_size"],MAX_PENDING)
            self.assertEqual(row["quarantine_working_memory_capacity"],MAX_PENDING)
            self.assertEqual(row["gradient_update_calls"],
                             row["lived_experiences"])
            self.assertTrue(row["no_world_id_passed_to_hidden_arm"])
            self.assertEqual(row["exam_priming_real_feedback_count"],16)
            self.assertTrue(row["exam_only_ephemeral_working_state"])
            if arm=="novelty12":
                self.assertEqual(row["final_distinct_cues_per_bank"],[4,4,4])
                self.assertEqual(row["final_memory_size"],12)


if __name__=="__main__":
    unittest.main()
