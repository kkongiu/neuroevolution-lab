"""M5.25: writer quarantine, bank isolation and causal controls."""
import unittest
from neuroevolution.m5_25_guarded_memory import (
    ARMS, SEEDS, BLOCK, MEMORY_LIMIT, LOCK_AFTER, PENDING_EVIDENCE,
    GuardedOrganism, Organism, check_seed, schedule, observations,
    environmental_success, frozen_exam, make_organism, run, CONTEXT
)


class GuardedMemoryTests(unittest.TestCase):
    def test_fresh_development_firewall_and_same_worlds(self):
        self.assertEqual(SEEDS,tuple(range(4900,4924)))
        self.assertEqual(len(ARMS),9)
        self.assertEqual(MEMORY_LIMIT,128)
        for bad in (4800,4924,4700,163000,161000,4900.0):
            with self.assertRaises(ValueError):check_seed(bad)
        for phase,(world,n) in enumerate(schedule(4900)):
            seq=observations(4900,phase,n)
            for i in range(0,n,BLOCK):
                self.assertEqual(set(seq[i:i+BLOCK]),
                                 {(0,0),(0,1),(1,0),(1,1)})
        for x in (0,1):
            for y in (0,1):
                self.assertNotEqual(
                    environmental_success("A",x,y,0),
                    environmental_success("B",x,y,0))

    def test_all_founders_identical_and_oracle_excluded(self):
        brains=[make_organism(4900,a) for a in ARMS]
        self.assertEqual(len({b.founder for b in brains}),1)
        for b in brains:
            if b.arm!="oracle_context":
                with self.assertRaises(AssertionError):
                    b.current_bank(ctx_for_oracle=1)
        for arm in ARMS:
            if arm.startswith("guarded_"):
                self.assertIsInstance(make_organism(4900,arm),
                                      GuardedOrganism)
        with self.assertRaises(ValueError):
            make_organism(4900,"teacher_correct_action")

    def test_verified_memory_immutable_and_switch_from_own_rewards(self):
        brain=GuardedOrganism(4900,"guarded_director")
        def feed(world, n):
            # Every action is a physically taken action 0. Success is
            # observed from that action, never a teacher-correct label.
            patterns=[(0,0),(0,1),(1,0),(1,1)]
            for j in range(n):
                x,y=patterns[j%4]
                r=environmental_success(world,x,y,0)
                brain.experience(x,y,0,r)
        feed("A",40)
        original=tuple(brain.banks[0].records)
        self.assertEqual(len(original),LOCK_AFTER)
        self.assertEqual(brain.verified_archive_freezes,1)
        # New opposite world: wait for 12 lived outcomes. No old-bank
        # writes are authorized while those observations are ambiguous.
        feed("B",PENDING_EVIDENCE-1)
        self.assertEqual(len(brain.banks),1)
        self.assertEqual(tuple(brain.banks[0].records),original)
        self.assertEqual(len(brain.pending),PENDING_EVIDENCE-1)
        x,y=(1,1)
        brain.experience(x,y,0,environmental_success("B",x,y,0))
        self.assertEqual(len(brain.banks),2)
        self.assertEqual(brain.active,1)
        self.assertEqual(len(brain.banks[1].records),PENDING_EVIDENCE)
        feed("B",LOCK_AFTER-PENDING_EVIDENCE)
        bankB=tuple(brain.banks[1].records)
        self.assertEqual(len(bankB),LOCK_AFTER)
        # Third rule is compatible with A/B on some cues, but differs
        # across the full observation set and merits a separate bank.
        feed("C",LOCK_AFTER)
        self.assertEqual(len(brain.banks),3)
        self.assertEqual(len(brain.banks[2].records),LOCK_AFTER)
        bankC=tuple(brain.banks[2].records)
        self.assertEqual(tuple(brain.banks[0].records),original)
        self.assertEqual(tuple(brain.banks[1].records),bankB)
        # Revisits do not corrupt any of the three committed banks.
        feed("A",32)
        feed("B",32)
        self.assertEqual(tuple(brain.banks[0].records),original)
        self.assertEqual(tuple(brain.banks[1].records),bankB)
        self.assertEqual(tuple(brain.banks[2].records),bankC)
        self.assertGreaterEqual(brain.denied_stable_writes,1)
        self.assertLessEqual(sum(len(b.records) for b in brain.banks),
                             MEMORY_LIMIT)

    def test_mutable_control_writes_in_uncertain_regime(self):
        stable=GuardedOrganism(4900,"guarded_director")
        mutable=GuardedOrganism(4900,"guarded_mutable")
        for model in (stable,mutable):
            for _ in range(10):
                for x,y in ((0,0),(0,1),(1,0),(1,1)):
                    model.experience(
                        x,y,0,environmental_success("A",x,y,0))
        old=tuple(stable.banks[0].records)
        for _ in range(3):
            for x,y in ((0,0),(0,1),(1,0),(1,1)):
                for m in (stable,mutable):
                    m.experience(
                        x,y,0,environmental_success("B",x,y,0))
        self.assertEqual(tuple(stable.banks[0].records),old)
        self.assertNotEqual(tuple(mutable.banks[0].records),old)
        self.assertGreater(mutable.tentative_bad_writes,0)

    def test_exam_cannot_train_and_premature_cue_free_exams(self):
        for arm in ARMS:
            model=make_organism(4900,arm)
            for x,y in ((0,0),(0,1),(1,0),(1,1)):
                model.experience(
                    x,y,0,environmental_success("A",x,y,0),
                    **({"context_for_oracle":0}
                       if arm=="oracle_context" else {}))
            before=model.snapshot()
            grade,bank=frozen_exam(
                model,4900,"A",20,priming=16)
            cold,_=frozen_exam(
                model,4900,"A",20,priming=0)
            self.assertEqual(before,model.snapshot())
            self.assertTrue(0<=grade<=1)
            self.assertTrue(0<=cold<=1)

    def test_full_reproducible_life_and_blocked_evolution(self):
        for arm in ARMS:
            a=run(4900,arm)
            self.assertEqual(a,run(4900,arm))
            self.assertEqual(a["gradient_update_calls"],
                             a["lived_experiences"])
            self.assertLessEqual(a["final_memory_size"],MEMORY_LIMIT)
            self.assertTrue(a["exam_only_ephemeral_working_state"])
            if arm!="oracle_context":
                self.assertTrue(a["no_world_id_passed_to_hidden_arm"])


if __name__=="__main__":
    unittest.main()
