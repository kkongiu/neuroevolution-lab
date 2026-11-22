"""M5.11 anatomical growth/pruning experiment invariants."""
import random
import unittest

from neuroevolution.m5_11_synapses import (
    SEEDS,ARMS,CAPACITY,INITIAL_ACTIVE,STRUCTURAL_INTERVAL,MIN_ACTIVE,
    Synapses,genotype,initial_active_set,observed,rules,schedule,run,experiment,
)

class SynapticRemodelingTests(unittest.TestCase):
    def test_equal_initial_sparsity_no_correct_action_leak(self):
        for seed in (3400,3401):
            for arm in ARMS:
                syn=Synapses(seed,arm)
                expected=CAPACITY if arm=="full_dense" else INITIAL_ACTIVE
                self.assertEqual(len(syn.active),expected)
                self.assertEqual(syn.initial_count,expected)
                self.assertEqual(
                    {i for i,w in enumerate(syn.agent.weights) if w!=0},
                    syn.active)
                self.assertEqual(syn.agent.predictor.learned_transitions,0)
            self.assertEqual(len(schedule(seed)),800)
            self.assertEqual(len(rules(seed)),3)
            self.assertEqual(len(observed(1,0,0)),19)
            self.assertEqual(Synapses(seed,"fixed_sparse").active,
                             initial_active_set(seed))

    def test_dormant_edges_cannot_learn_without_synapse_birth(self):
        syn=Synapses(3400,"fixed_sparse")
        initial=set(syn.active)
        for step in range(15):
            obs=observed(0,step%2,0.)
            action,cache=syn.agent.action(
                obs,(0,1),random.Random(771+step))
            syn.masked_update(cache,int(step%3==0))
        self.assertEqual(set(syn.active),initial)
        self.assertEqual(
            {i for i,w in enumerate(syn.agent.weights) if w!=0},initial)
        self.assertEqual(len(syn.events),0)
        self.assertTrue(all(syn.agent.weights[i]==0 for i in
                            set(range(CAPACITY))-initial))

    def test_structural_change_only_from_observations(self):
        for arm in ("prune_only","grow_only","grow_prune","turnover_random"):
            syn=Synapses(3400,arm)
            for tick in range(STRUCTURAL_INTERVAL*4):
                obs=observed(0,tick%2,0.)
                _,cache=syn.agent.action(obs,(0,1),random.Random(1100+tick))
                syn.masked_update(cache,0)
                syn.remodel(tick)
            self.assertGreaterEqual(len(syn.active),MIN_ACTIVE)
            self.assertLessEqual(len(syn.active),CAPACITY)
            self.assertTrue(all(0<=v["active"]<=CAPACITY
                                for v in syn.events))
            self.assertEqual(
                {i for i,w in enumerate(syn.agent.weights) if w!=0},
                syn.active)
            if arm=="prune_only":
                self.assertEqual(syn.total_new,0)
                self.assertGreater(syn.total_removed,0)
            if arm=="grow_only":
                self.assertEqual(syn.total_removed,0)
                self.assertGreater(syn.total_new,0)
            if arm in ("grow_prune","turnover_random"):
                self.assertGreater(syn.total_removed,0)
                self.assertGreater(syn.total_new,0)

    def test_three_controls_and_guard(self):
        for seed in (3400,3401):
            rows={arm:run(seed,arm) for arm in ARMS}
            self.assertEqual(rows["fixed_sparse"]["new_synapses"],0)
            self.assertEqual(rows["fixed_sparse"]["pruned_synapses"],0)
            self.assertEqual(rows["full_dense"]["mean_synapses"],CAPACITY)
            self.assertEqual(rows["full_dense"]["new_synapses"],0)
            self.assertEqual(rows["full_dense"]["pruned_synapses"],0)
            for r in rows.values():
                self.assertFalse(r["policy_received_correct_action"])
                self.assertFalse(r["reproduction"])
                self.assertTrue(0<=r["overall_accuracy"]<=1)
            self.assertEqual(rows["grow_prune"],run(seed,"grow_prune"))
        for seed in (3300,3424,161000,162000,163000):
            with self.assertRaises(ValueError):run(seed,"grow_prune")
        with self.assertRaises(ValueError):experiment((3400,3400))
        result=experiment((3400,3401))
        self.assertFalse(result["advance_to_evolution"] or
                         not all(result["screening_gate"].values()))
        self.assertTrue(result["real_synapse_birth_death"])

if __name__=="__main__":
    unittest.main()
