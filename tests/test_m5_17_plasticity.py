"""M5.17: importance protection, finite episodic memory and yoking."""
import unittest
from neuroevolution.m5_17_plasticity import (
    ARMS,SEEDS,PHASES,PARAMS,INITIAL_SPARSE,MEMORY_LIMIT,UPDATES,
    TRAIN_STEPS,PROBE_STEPS,Learner,require_seed,curriculum,sensory,
    environmental_reward,frozen_exam,simulate,experiment,
)

class M517PlasticityTests(unittest.TestCase):
    def test_balanced_cues_and_opposing_rules(self):
        for phase,(name,n) in enumerate(PHASES):
            pair=curriculum(4000,phase,n)
            self.assertEqual(len(pair),n)
            for j in range(0,n,40):
                self.assertEqual(pair[j:j+40].count((0,0)),10)
                self.assertEqual(set(pair[j:j+40]),{
                    (0,0),(0,1),(1,0),(1,1)})
        for x in (0,1):
            for y in (0,1):
                self.assertNotEqual(environmental_reward(0,x,y,0),
                                    environmental_reward(1,x,y,0))
                self.assertEqual(len(sensory(x,y,2,1)),8)

    def test_controls_share_founder_and_memory_budgets(self):
        all_brains=[Learner(4000,a,yoked_events=[] if a==
                    "replay_plasticity_yoked" else None) for a in ARMS]
        expected=all_brains[0].original_weights
        self.assertEqual(len(expected),PARAMS)
        for brain in all_brains:
            self.assertEqual(brain.original_weights,expected)
        sparse=[b for b in all_brains if b.arm.startswith(
            "replay_plasticity_") and b.arm!="replay_plasticity"]
        self.assertEqual(len(sparse),3)
        self.assertTrue(all(len(b.active)==INITIAL_SPARSE for b in sparse))
        self.assertEqual(sparse[0].active,sparse[1].active)
        for arm in ("replay","replay_plasticity","replay_plasticity_sparse",
                    "replay_uncoupled"):
            b=Learner(4000,arm)
            for tick in range(135):
                ctx=tick//45
                x,y=tick%2,(tick//2)%2
                act=b.choose(x,y,ctx,tick,True)
                reward=environmental_reward(ctx,x,y,act)
                b.observe(x,y,ctx,act,reward,tick)
            self.assertLessEqual(
                sum(len(v) for v in b.context_memory.values()),MEMORY_LIMIT)
            self.assertEqual(b.updates,135*UPDATES)

    def test_no_cross_action_teacher_labels_and_frozen_exam(self):
        for arm in ARMS:
            brain=Learner(4000,arm,yoked_events=[] if arm==
                          "replay_plasticity_yoked" else None)
            for tick in range(48):
                x,y=tick%2,(tick//2)%2
                a=brain.choose(x,y,0,tick,True)
                r=environmental_reward(0,x,y,a)
                brain.observe(x,y,0,a,r,tick)
            before=brain.snapshot()
            x=frozen_exam(brain,4000,0,20)
            self.assertTrue(0<=x<=1)
            self.assertEqual(before,brain.snapshot())
            if arm=="plasticity_only":
                brain.consolidate(0)
                self.assertEqual(len(brain.anchors),1)
                self.assertGreater(max(brain.anchors[0][1]),0)

    def test_yoked_growth_death_and_equal_update_control(self):
        grown=simulate(4000,"replay_plasticity_growth")
        plan=[{k:e[k] for k in ("tick","kind","count")}
              for e in grown["events"]]
        yoked=simulate(4000,"replay_plasticity_yoked",yoked_events=plan)
        self.assertEqual(
            [(e["tick"],e["kind"],e["count"],e["active"])
             for e in grown["events"]],
            [(e["tick"],e["kind"],e["count"],e["active"])
             for e in yoked["events"]])
        self.assertEqual(grown["mean_active_parameters"],
                         yoked["mean_active_parameters"])
        self.assertEqual(grown["parameter_updates"],TRAIN_STEPS*UPDATES)
        self.assertEqual(yoked["parameter_updates"],TRAIN_STEPS*UPDATES)
        self.assertEqual(grown,simulate(4000,"replay_plasticity_growth"))
        sample=experiment((4000,4001))
        self.assertEqual(sample["development_screen_pass"],
                         all(sample["screening_gate"].values()))
        self.assertFalse(sample["advance_to_evolution"])
        self.assertFalse(sample["independent_replication_complete"])
        self.assertFalse(sample["evolution_authorized"])

    def test_cohort_isolation(self):
        for bad in (3900,4024,161000,162000,163000,4000.0):
            with self.assertRaises(ValueError):require_seed(bad)
        with self.assertRaises(ValueError):experiment((4000,4000))
        with self.assertRaises(ValueError):
            Learner(4000,"replay_plasticity_yoked")
        with self.assertRaises(ValueError):
            Learner(4000,"replay",yoked_events=[])

if __name__=="__main__":unittest.main()
