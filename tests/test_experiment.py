import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

from neuroevolution.experiment import Config, Genome, evaluate, mutate, run


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            seed=7, generations=3, population=6, episodes=2,
            steps=40, hidden=4, flip_interval=10, mutations=2
        )

    def test_deterministic_evaluation(self):
        genome = Genome.fresh(random.Random(11), self.config.hidden)
        first = evaluate(genome, [13, 19], self.config)
        self.assertEqual(first, evaluate(genome, [13, 19], self.config))
        self.assertGreaterEqual(first["accuracy"], 0.0)
        self.assertLessEqual(first["accuracy"], 1.0)

    def test_weight_mutation_keeps_topology(self):
        genome = Genome.fresh(random.Random(11), self.config.hidden)
        child = mutate(genome, random.Random(2), topology=False, count=30)
        self.assertEqual((genome.mi, genome.mr, genome.mo, genome.active),
                         (child.mi, child.mr, child.mo, child.active))
        self.assertNotEqual(genome.wi + genome.wr + genome.wo + genome.bh + genome.bo,
                            child.wi + child.wr + child.wo + child.bh + child.bo)

    def test_structural_mutation_is_possible(self):
        genome = Genome.fresh(random.Random(11), self.config.hidden)
        child = mutate(genome, random.Random(2), topology=True, count=100)
        self.assertNotEqual((genome.mi, genome.mr, genome.mo, genome.active),
                            (child.mi, child.mr, child.mo, child.active))
        self.assertGreaterEqual(sum(child.active), 1)

    def test_repeatable_campaign(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a = run(self.config, root / "a")
            b = run(self.config, root / "b")
            self.assertEqual(a, b)
            self.assertTrue((root / "a" / "scores.svg").exists())
            self.assertTrue((root / "a" / "checkpoint.json").exists())
            self.assertEqual(len(a), 3)

    def test_resuming_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full = run(self.config, root / "full")
            short = Config(**{**self.config.__dict__, "generations": 1})
            run(short, root / "short")
            resumed = run(self.config, root / "short",
                          resume=root / "short" / "checkpoint.json")
            self.assertEqual(full, resumed)
            checkpoint = json.loads((root / "short" / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["next_generation"], self.config.generations)

    def test_mismatched_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run(self.config, root / "first")
            other = Config(**{**self.config.__dict__, "population": 8})
            with self.assertRaises(ValueError):
                run(other, root / "second", resume=root / "first" / "checkpoint.json")


if __name__ == "__main__":
    unittest.main()
