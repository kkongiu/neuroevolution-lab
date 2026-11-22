"""Genotype vs lifetime memory strict inheritance, selection controls and seed seals."""
import math
import unittest

from neuroevolution.m5_6_online import Learner
from neuroevolution.m5_9_selection import (
    ARMS, POPULATION, TRAIN_EPISODES, Puzzle, evaluate_individual,
    experiment, founder_genomes, generation_step, mutate, puzzles, run
)


class HeritablePredispositionTests(unittest.TestCase):
    def test_fresh_child_receives_genetic_initial_weights_not_memory(self):
        g=founder_genomes(3000)[0]
        parent=Learner.fresh(g,(5,5))
        parent.hidden=[.5]*8
        parent.weights[3]+=1.
        parent.curiosity.visits[(7,7)]=5
        parent.predictor.learned_transitions=13
        child_genome=mutate(g,__import__("random").Random(17))
        child=Learner.fresh(child_genome,(5,5))
        self.assertEqual(child.weights,list(child_genome))
        self.assertNotEqual(child.weights,parent.weights)
        self.assertEqual(child.hidden,[0.]*8)
        self.assertEqual(child.predictor.learned_transitions,0)
        self.assertEqual(set(child.curiosity.visits),{(5,5)})

    def test_puzzle_answer_never_revealed_in_sensors(self):
        a=Puzzle(0,0,(0,1,1,0))
        b=Puzzle(0,1,(0,1,1,0))
        self.assertEqual(a.observation(1,0),b.observation(1,0))
        self.assertNotEqual(a.successful(1,0),b.successful(1,0))
        self.assertEqual(len(a.observation(0,0)),19)

    def test_plasticity_is_private_and_frozen_control(self):
        g=founder_genomes(3000)[0]
        tasks=puzzles(3000,0,TRAIN_EPISODES)
        a=evaluate_individual(g,tasks,12345,plastic=True)
        b=evaluate_individual(g,tasks,12345,plastic=False)
        self.assertGreater(a["learn_updates"],0)
        self.assertGreater(a["learned_weight_drift"],0)
        self.assertEqual(b["learned_weight_drift"],0.)
        self.assertEqual(b["learn_updates"],0)
        self.assertEqual(a["personal_predictor_updates"],0)

    def test_parent_selection_does_not_use_memory_as_genome(self):
        g=founder_genomes(3000)
        tasks=puzzles(3000,0,TRAIN_EPISODES)
        for arm in ARMS:
            children,report=generation_step(g,tasks,3000,0,arm)
            self.assertEqual(len(children),POPULATION)
            self.assertTrue(report["all_children_inherit_initial_genotypes"])
            self.assertEqual(len(report["parent_indices"]),POPULATION)
            self.assertTrue(all(len(c)==260 for c in children))
        self.assertEqual(run(3000,"solution_plastic",generations=1),
                         run(3000,"solution_plastic",generations=1))

    def test_guards_and_isolated_check_worlds(self):
        for seed in (2000,2064,3050,161000,162000,163000):
            with self.assertRaises(ValueError):founder_genomes(seed)
        for seed in (3000,2000,162000):
            with self.assertRaises(ValueError):
                puzzles(seed,777,5,evaluation=True)
        with self.assertRaises(ValueError):
            experiment((3000,3000))
        with self.assertRaises(ValueError):
            run(3000,"solution_plastic",generations=999)


if __name__=="__main__":
    unittest.main()
