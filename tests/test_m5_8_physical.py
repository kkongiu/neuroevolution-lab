"""M5.8 physically enacted exogenous shocks (development-only)."""
import unittest
from dataclasses import replace

from neuroevolution.m5_6_balance_pilot import settings as pilot_settings
from neuroevolution.m5_6_online import simulate_development
from neuroevolution.m5_8_worlds import make_world, reachable
from neuroevolution.m5_8_adapter import ActiveShocks, attach
from neuroevolution import m5_0 as base


class PhysicalWorldTests(unittest.TestCase):
    @staticmethod
    def settings():
        return replace(pilot_settings(), ticks=60, initial_food=9,
                       regeneration_probability=.55)

    def test_attached_world_is_real_and_conserves_energy(self):
        s = self.settings()
        eco = base.create_ecology(2000, s, "heritable_no_mutation")
        world = make_world(2000)
        attach(eco, world)
        self.assertEqual({tuple(p) for p in world["food"]}, eco.food)
        self.assertEqual({tuple(p) for p in world["walls"]},
                         set(eco.world.walls) - {
                             (x,y) for x in range(11) for y in range(11)
                             if x in (0,10) or y in (0,10)})
        self.assertEqual(len(eco.agents), 8)
        self.assertEqual(len(set(a.position for a in eco.agents.values())),8)
        self.assertFalse(any(a.position in eco.world.walls
                             for a in eco.agents.values()))
        self.assertAlmostEqual(base.energy_balance(eco), 0.)

    def test_same_world_shocks_change_physical_process_not_energy(self):
        s = self.settings()
        normal = simulate_development(
            2000, s, "learning_progress", predator_count=1,
            curiosity_gain=4., learning_cutoff_tick=30,
            model_decision_mode="model", progressive_world=True,
            stochastic_challenges=False)
        stressed = simulate_development(
            2000, s, "learning_progress", predator_count=1,
            curiosity_gain=4., learning_cutoff_tick=30,
            model_decision_mode="model", progressive_world=True,
            stochastic_challenges=True)
        self.assertEqual(normal["world_sha256"], stressed["world_sha256"])
        self.assertEqual(normal["actual_shocks_proposed"],0)
        self.assertGreater(stressed["actual_shocks_proposed"],0)
        self.assertGreaterEqual(stressed["predator_extra_steps"],0)
        self.assertLess(abs(stressed["energy_residual"]),1e-6)
        self.assertLess(abs(normal["energy_residual"]),1e-6)
        self.assertEqual(stressed, simulate_development(
            2000, s, "learning_progress", predator_count=1,
            curiosity_gain=4., learning_cutoff_tick=30,
            model_decision_mode="model", progressive_world=True,
            stochastic_challenges=True))

    def test_explicit_world_fences(self):
        with self.assertRaises(ValueError):
            simulate_development(162000,self.settings(),"novelty",
                                 progressive_world=True)
        with self.assertRaises(ValueError):
            simulate_development(2000,self.settings(),"novelty")
        with self.assertRaises(ValueError):
            simulate_development(2000,self.settings(),"novelty",
                                 stochastic_challenges=True)


if __name__=="__main__":
    unittest.main()
