"""M4.4A strictly non-heldout tests of genuine online synaptic causal memory."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from neuroevolution import m4_2 as old
from neuroevolution import m4_3 as previous
from neuroevolution import m4_4a as m


class OnlineRnnPhysicalLearning(unittest.TestCase):
    def setUp(self):
        self.settings=old.Settings()
        self.settings.validate()

    def test_fixed_protocol_disjoint_and_original_network_architectures(self):
        self.assertEqual(m.PROTOCOL_SHA,
                         "af9e64e9e2ec3a5f47d27f963c27467661ee1681")
        self.assertEqual((old.genome_size("rnn",self.settings),
                          old.genome_size("ff",self.settings)),(260,268))
        self.assertEqual(m.UPDATES,288)
        self.assertEqual(len(m.INITIALS),8)
        for a in (m.TRAIN_WORLDS,m.MAIN_WORLDS,
                  m.OOD_WORLDS,m.AUTONOMOUS_WORLDS):
            for bb in (m.TRAIN_WORLDS,m.MAIN_WORLDS,
                       m.OOD_WORLDS,m.AUTONOMOUS_WORLDS):
                if a is not bb:
                    self.assertFalse(set(a)&set(bb))
        self.assertFalse(set(m.DEVELOPMENT_WORLDS)&set(m.MAIN_WORLDS))
        self.assertEqual(m.BOOTSTRAP,20000)

    def test_network_forward_equivalent_to_unmodified_m4_2_controller(self):
        for architecture in ("rnn","ff"):
            g=m._initial_genome(m.INITIALS[0],architecture,self.settings)
            world=old.make_pair(m.DEVELOPMENT_WORLDS[0],self.settings)[0]
            sensory=[]
            pos=world.start
            for tick in range(world.stem_length):
                obs=old.observation(world,pos,tick,24.-tick,0,0,self.settings)
                sensory.append(obs)
                dx,dy=old.DIRECTIONS[world.forward]
                pos=pos[0]+dx,pos[1]+dy
            sensory.append(old.observation(
                world,world.fork,world.stem_length,
                24.-world.stem_length,0,0,self.settings))
            controller=old.Controller(tuple(g),architecture,self.settings)
            for x in sensory:
                controller.choose(x)
            from_old=previous._logits(controller)
            ours, caches=m._forward(g,architecture,sensory,self.settings)
            for a,b in zip(from_old,ours):
                self.assertAlmostEqual(a,b,places=12)
            self.assertEqual(caches[-1][2],controller.hidden)

    def test_exact_reinforce_bptt_gradient_against_numeric_differences(self):
        world=old.make_pair(m.DEVELOPMENT_WORLDS[1],self.settings)[0]
        inputs=[
            old.observation(world,world.start,0,24.,0,0,self.settings),
            old.observation(world,world.start,1,23.,0,0,self.settings),
            previous.neutral_observation(),
            previous.fork_observation(world),
        ]
        legal=tuple(sorted(old.direction(world.fork,g) for g in world.gates))
        for arch in ("rnn","ff"):
            g=m._initial_genome(m.INITIALS[0],arch,self.settings)
            logits, cache=m._forward(g,arch,inputs,self.settings)
            grad,norm=m._reward_gradient(
                g,arch,cache,legal,legal[0],1,self.settings)
            self.assertGreater(norm,0)
            chosen=legal[0]
            def target():
                val,_=m._forward(g,arch,inputs,self.settings)
                ps=m._softmax_two(val,legal)
                return math.log(ps[0])
            for index in (0,15,18,19,31,
                          len(g)//2,len(g)-2,len(g)-1):
                saved=g[index]
                eps=1e-6
                g[index]=saved+eps
                up=target()
                g[index]=saved-eps
                down=target()
                g[index]=saved
                self.assertAlmostEqual((up-down)/(2*eps),grad[index],
                                       delta=5e-6,msg=f"{arch}:{index}")

    def test_original_physical_corridor_reward_and_no_goal_leak(self):
        a,b=old.make_pair(m.DEVELOPMENT_WORLDS[2],self.settings)
        self.assertEqual(m.physical_world_hash(a),m.physical_world_hash(b))
        genome=m._initial_genome(m.INITIALS[0],"rnn",self.settings)
        for world in (a,b):
            record,caches,legal=m.physical_episode(
                world,self.settings,genome=genome)
            self.assertEqual(record["fork_reached"],1)
            self.assertEqual(record["physical_collisions"],0)
            self.assertEqual(record["cue_observed_at_start"],1)
            self.assertEqual(record["cue_absent_at_fork"],1)
            self.assertEqual(record["guided_forward_moves"],world.stem_length)
            self.assertEqual(record["reward_from_food"],
                             2*record["physical_food_eaten"]-1)
            self.assertTrue(caches)
            self.assertEqual(record["real_cardinal_moves"],world.stem_length+1)
            self.assertAlmostEqual(record["actual_final_energy"],
                self.settings.initial_energy-record["real_cardinal_moves"]
                *self.settings.movement_cost
                +record["physical_food_eaten"]*self.settings.food_energy)
            self.assertEqual(record["fork_sensors_sha"],
                 m.digest(old.observation(world,world.fork,world.stem_length,
                       self.settings.initial_energy-
                       world.stem_length*self.settings.movement_cost,
                       0,0,self.settings)))
        aa,_,_=m.physical_episode(a,self.settings,genome=genome)
        bb,_,_=m.physical_episode(b,self.settings,genome=genome)
        self.assertEqual(aa["fork_sensors_sha"],bb["fork_sensors_sha"])
        self.assertEqual(aa["chosen_action"],bb["chosen_action"])
        self.assertEqual(aa["physical_food_eaten"]+bb["physical_food_eaten"],1)
        for world in (a,b):
            ref,_,_=m.physical_episode(world,self.settings,genome=None,symbolic=True)
            self.assertEqual(ref["physical_food_eaten"],1)

    def test_online_training_modifies_true_recurrent_synapses(self):
        bank,curves=m.train_source(
            initials=(m.INITIALS[0],),
            train_worlds=m.DEVELOPMENT_WORLDS[:2],
            epochs=2)
        self.assertEqual(len(curves),24)
        self.assertEqual(bank["heldout_worlds_examined"],0)
        self.assertEqual(bank["episodes_per_training_arm"],8)
        family=bank["networks"][0]
        for value in family["online_rnn_l2_drift_per_module"].values():
            self.assertGreater(value,0)
        self.assertNotEqual(family["online_rnn"],family["initial_rnn"])
        self.assertNotEqual(family["shuffled_reward_rnn"],
                            family["initial_rnn"])
        self.assertEqual(family["initial_rnn"],
                         m._initial_genome(m.INITIALS[0],"rnn",self.settings))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"bank.json"
            path.write_text(json.dumps(bank),encoding="utf-8")
            self.assertEqual(m.load_bank(path,allow_development=True)
                             ["bank_sha256"],bank["bank_sha256"])
            bad=json.loads(path.read_text())
            bad["networks"][0]["online_rnn"][0]+=0.01
            path.write_text(json.dumps(bad),encoding="utf-8")
            with self.assertRaises(AssertionError):
                m.load_bank(path,allow_development=True)
        with self.assertRaises(ValueError):
            m.train_source(initials=(m.INITIALS[0],),
                           train_worlds=m.MAIN_WORLDS[:1],
                           epochs=1)

    def test_balanced_new_worlds_and_cluster_inference(self):
        bank,curves=m.train_source(
            initials=(m.INITIALS[0],m.INITIALS[1]),
            train_worlds=m.DEVELOPMENT_WORLDS[:1],epochs=1)
        rows=[]
        for split,seeds,harder in (
            ("main",m.DEVELOPMENT_WORLDS[2:4],False),
            ("ood",m.DEVELOPMENT_WORLDS[4:6],True)):
            for family in bank["networks"]:
                for seed in seeds:
                    for w in old.make_pair(seed,self.settings,shifted=harder):
                        for arm in m.ARMS:
                            row=m._evaluate_arm(w,arm,family,self.settings)
                            row["split"]=split
                            rows.append(row)
        ans=m.evaluate(
            rows,bank,main=m.DEVELOPMENT_WORLDS[2:4],
            ood=m.DEVELOPMENT_WORLDS[4:6],reps=100)
        self.assertEqual(ans["main"]["independent_network_initializations"],2)
        for split in ("main","ood"):
            self.assertEqual(ans[split]["arms"]["scripted_one_bit"]
                             ["collected_food_rate"],1.)
            self.assertEqual(ans[split]["arms"]["random_binary"]
                             ["collected_food_rate"],.5)
            self.assertEqual(ans[split]["arms"]["online_ff"]
                             ["collected_food_rate"],.5)
            self.assertEqual(ans[split]["arms"]["online_rnn_reset"]
                             ["collected_food_rate"],.5)
        self.assertFalse(ans["gates"]
                         ["unassisted_autonomous_lifelong_learning_proven"])
        rows.pop()
        with self.assertRaises(AssertionError):
            m.evaluate(rows,bank,main=m.DEVELOPMENT_WORLDS[2:4],
                       ood=m.DEVELOPMENT_WORLDS[4:6],reps=100)


if __name__=="__main__":
    unittest.main()
