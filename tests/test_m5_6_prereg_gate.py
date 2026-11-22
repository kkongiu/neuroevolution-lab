"""M5.6 safety checks: blind world cohorts and fail-closed final run."""
import unittest

from neuroevolution.m5_6_prereg_gate import (
    manifest, assert_ready_for_holdout, REQUIRED_ARMS, SOURCE, HOLDOUT,
    REPLICATION, DEVELOPMENT,
)


class PreregistrationSafetyTests(unittest.TestCase):
    def test_cohorts_are_blind_independent_and_manifest_stable(self):
        cohorts = (SOURCE, HOLDOUT, REPLICATION, DEVELOPMENT)
        self.assertTrue(all(not (set(a) & set(b))
                            for i, a in enumerate(cohorts)
                            for b in cohorts[i + 1:]))
        self.assertGreaterEqual(len(SOURCE), 32)
        self.assertGreaterEqual(len(HOLDOUT), 32)
        self.assertGreaterEqual(len(REPLICATION), 32)
        self.assertEqual(manifest(), manifest())
        self.assertEqual(len(manifest()["sha256"]), 64)
        self.assertEqual(manifest()["status"],
                         "provisional_seed_manifest_not_executable_freeze")

    def test_final_gate_refuses_incomplete_factorial(self):
        with self.assertRaisesRegex(RuntimeError, "BLOCKED"):
            assert_ready_for_holdout(
                executable_frozen=False, source_weights_frozen=False,
                world_difficulty_frozen=False, matched_budget_verified=False,
                retention_test_implemented=False, implemented_arms=set())

    def test_final_gate_requires_all_independent_integrity_guards(self):
        flags = dict(executable_frozen=True, source_weights_frozen=True,
                     world_difficulty_frozen=True,
                     matched_budget_verified=True,
                     retention_test_implemented=True,
                     implemented_arms=set(REQUIRED_ARMS))
        assert_ready_for_holdout(**flags)
        for key in list(flags):
            partial = flags.copy()
            partial[key] = (set(REQUIRED_ARMS) - {"curiosity_memory_reset_predators"}
                            if key == "implemented_arms" else False)
            with self.assertRaisesRegex(RuntimeError, "BLOCKED"):
                assert_ready_for_holdout(**partial)


if __name__ == "__main__":
    unittest.main()
