"""M5.34: function-level exclusion, legitimate meta-supervision and frozen inference."""
import unittest
import numpy as np

import neuroevolution.m5_34_learned_representation as m
from neuroevolution.m5_33_active_cue import CATALOG


class M534Tests(unittest.TestCase):
    def test_split_complete_function_firewall_and_shape(self):
        train, test = m.entire_function_split()
        self.assertEqual(len(train), 85)
        self.assertEqual(len(test), 27)
        self.assertEqual(len({f["signature"] for f in train + test}), 112)
        self.assertFalse({f["signature"] for f in train} &
                         {f["signature"] for f in test})
        self.assertEqual(set(m.HOLDOUT), set(m.COHORT_EFFECTS))
        for family, n in m.HOLDOUT.items():
            self.assertEqual(sum(f["cohort"] == family for f in test), n)
        self.assertEqual(m.CUE_COUNT, 16)
        self.assertEqual(m.SUPPORT, 8)
        self.assertEqual(m.HIDDEN, 64)
        self.assertEqual(m.PRETRAIN_STEPS, 3000)
        self.assertEqual(m.BATCH, 32)
        self.assertEqual(m.OPTIMIZER_SEEDS, (1301, 1303))
        self.assertEqual(len(m.ARMS), 5)
        self.assertEqual(len(m.POLICIES), 2)
        for f in test:
            self.assertTrue(any(np.array_equal(CATALOG[j], f["truth"])
                                for j in range(len(CATALOG))))

    def test_record_encoding_no_oracle_and_reward_complement(self):
        train, held = m.entire_function_split()
        f = held[-1]
        a, exam, trace = m.heldout_support(f, 1, "random_cues")
        feat = m.input_from_records(a)
        self.assertEqual(feat.shape, (32,))
        self.assertEqual(feat.dtype, np.float32)
        self.assertEqual(int(np.count_nonzero(feat[16:])), 8)
        self.assertEqual(len(set(x[0] for x in a)), 8)
        self.assertEqual(set(exam) & {x[0] for x in a}, set())
        for cue, action, reward in a:
            self.assertEqual(feat[cue], 1-2*(action ^ (1-reward)))
            self.assertEqual(int(reward), int(action == f["truth"][cue]))
        self.assertEqual(len(trace), 8)
        with self.assertRaises(ValueError):
            m.input_from_records(a[:-1])
        with self.assertRaises(ValueError):
            m.heldout_support(f, 17, "random_cues")
        with self.assertRaises(ValueError):
            m.heldout_support(f, 0, "imaginary-policy")

    def test_random_and_active_shared_first_cues_but_distinct_selection(self):
        _, test = m.entire_function_split()
        choices_differ = False
        for f in test[:5]:
            for episode in (0,1,2):
                randomized, exam_r, _ = m.heldout_support(f, episode, "random_cues")
                active, exam_a, _ = m.heldout_support(
                    f, episode, "family_version_space_active"
                )
                self.assertEqual(
                    tuple(x[0] for x in randomized[:2]),
                    tuple(x[0] for x in active[:2])
                )
                self.assertEqual(len(exam_r), 8)
                self.assertEqual(len(exam_a), 8)
                for cue, action, reward in randomized:
                    observed = next((x for x in active if x[0] == cue), None)
                    if observed is not None:
                        self.assertEqual(observed, (cue, action, reward))
                choices_differ |= set(x[0] for x in randomized) != set(
                    x[0] for x in active
                )
        self.assertTrue(choices_differ)

    def test_shared_representation_real_float32_and_frozen_support(self):
        train, test = m.entire_function_split()
        f = test[0]
        records, unseen, _ = m.heldout_support(f, 0, "random_cues")
        model = m.SharedRepresentation(1301)
        self.assertEqual(sum(v.nbytes for v in model.params.values()), 3152*4)
        before = model.checkpoint()["sha256"]
        prediction_before, norm = model.predictions(records)
        self.assertEqual(len(prediction_before), 16)
        self.assertTrue(np.isfinite(norm))
        for cue, action, reward in records:
            self.assertEqual(prediction_before[cue], action ^ (1-reward))
        self.assertEqual(
            set(unseen) & {x[0] for x in records}, set()
        )
        self.assertEqual(model.checkpoint()["sha256"], before)
        # Pretraining tests use just three updates and restore the
        # full 3000-step preregistration immediately afterwards.
        old = m.PRETRAIN_STEPS
        m.PRETRAIN_STEPS = 3
        try:
            genuine = m.SharedRepresentation(1301).pretrain(
                train, corrupted=False
            )
            null = m.SharedRepresentation(1301).pretrain(
                train, corrupted=True
            )
            self.assertEqual(genuine.steps, 3)
            self.assertEqual(null.steps, 3)
            self.assertNotEqual(genuine.checkpoint()["sha256"], before)
            self.assertNotEqual(
                genuine.checkpoint()["sha256"],
                null.checkpoint()["sha256"]
            )
            self.assertTrue(genuine.pretraining_pairs_correct)
            self.assertFalse(null.pretraining_pairs_correct)
            self.assertEqual(genuine.training_source_signatures,
                             null.training_source_signatures)
            self.assertNotIn(f["signature"], genuine.training_source_signatures)
            self.assertEqual(len(genuine.training_source_signatures), 85)
            a, _ = genuine.predictions(records)
            b, _ = null.predictions(records)
            self.assertEqual(len(a), 16)
            self.assertEqual(len(b), 16)
        finally:
            m.PRETRAIN_STEPS = old

    def test_one_trial_all_controls_and_unseen_grading(self):
        train, test = m.entire_function_split()
        old = m.PRETRAIN_STEPS
        m.PRETRAIN_STEPS = 2
        try:
            trained = m.SharedRepresentation(1301).pretrain(train)
            corrupt = m.SharedRepresentation(1301).pretrain(
                train, corrupted=True
            )
            scratch = m.SharedRepresentation(1301)
            row = m.one_trial(
                test[-1], 0, "family_version_space_active", 1301,
                trained, corrupt, scratch
            )
            self.assertEqual(set(row["arm_accuracies"]), set(m.ARMS))
            self.assertEqual(set(row["arm_all16_frozen_predictions"]),
                             set(m.ARMS))
            self.assertEqual(len(row["eight_unseen_exam_ids"]), 8)
            self.assertTrue(row["same_support_across_all_learners"])
            self.assertTrue(row["no_heldout_function_used_to_pretrain"])
            self.assertTrue(row["no_query_seen_in_any_training_support"])
            self.assertEqual(row["dual_gradient_updates"], 2400)
            for arm in m.ARMS:
                self.assertEqual(
                    len(row["arm_all16_frozen_predictions"][arm]), 16
                )
                for k, value in row["arm_accuracies"][arm].items():
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, 1)
                self.assertEqual(
                    row["arm_accuracies"][arm]["support8_accuracy"], 1.
                )
        finally:
            m.PRETRAIN_STEPS = old


if __name__ == "__main__":
    unittest.main()
