"""M5.6 strict compatibility tests: same recurrence, legal local decisions."""
import random
import unittest

from neuroevolution import m4_2, m4_4a, m5_0, m5_5b
from neuroevolution.m5_6_bridge import (EcologicalRNN,
                                         ecological_observation,
                                         legal_cardinal_moves)


class EcologicalBridgeTests(unittest.TestCase):
    def test_original_architecture_is_preserved_exactly(self):
        settings = m4_2.Settings()
        g = tuple(m4_4a._initial_genome(m4_4a.INITIALS[0], "rnn", settings))
        self.assertEqual(len(g), 260)
        bridge = EcologicalRNN.create(g)
        observations = [
            tuple(random.Random(17 + i).uniform(-1, 1)
                  for _ in range(m4_2.INPUTS))
            for i in range(4)
        ]
        for k in range(1, 5):
            choice = bridge.decide(observations[k - 1], (0, 1, 2, 3))
            original_logits, original_cache = m4_4a._forward(
                g, "rnn", observations[:k], settings)
            self.assertEqual(choice, max(range(4), key=lambda a:
                                        (original_logits[a], -a)))
            for x, y in zip(bridge.hidden, original_cache[-1][2]):
                self.assertAlmostEqual(x, y, places=12)
        bridge.reset()
        self.assertEqual(bridge.hidden, [0.] * settings.hidden)

    def test_legal_motor_and_no_hidden_predator_map(self):
        settings = m5_5b.Settings(
            seed_start=300, seeds=1, shifted_start=400, shifted_seeds=1,
            ticks=12, size=11, initial_food=9, food_cap=14,
            initial_population=8, max_population=18,
            checkpoint_every=4, no_food_controls=0)
        eco = m5_0.create_ecology(300, settings, "heritable_no_mutation")
        agent = next(iter(eco.agents.values()))
        predator = (agent.position[0] + 1, agent.position[1])
        obs = ecological_observation(eco, agent, (predator,))
        self.assertEqual(len(obs), 19)
        self.assertEqual(obs[-4:], tuple(float(
            (agent.position[0] + dx, agent.position[1] + dy) == predator
        ) for dx, dy in m5_0.DIRECTIONS))
        legal = legal_cardinal_moves(eco, agent)
        genome = tuple(m4_4a._initial_genome(
            m4_4a.INITIALS[0], "rnn", m4_2.Settings()))
        self.assertIn(EcologicalRNN.create(genome).decide(obs, legal),
                      legal if legal else (None,))

    def test_invalid_weight_and_sensory_contract(self):
        with self.assertRaises(ValueError):
            EcologicalRNN.create((0.,))
        g = (0.,) * 260
        rnn = EcologicalRNN.create(g)
        with self.assertRaises(ValueError):
            rnn.decide((0.,) * 18, (0,))
        with self.assertRaises(ValueError):
            rnn.decide((0.,) * 19, (4,))
        self.assertIsNone(rnn.decide((0.,) * 19, ()))
