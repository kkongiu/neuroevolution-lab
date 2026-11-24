"""FlyLoop R1 physical / neuroscience negative-control validation."""
import math
import unittest

from neuroevolution import flyloop_r1 as r1


class FlyLoopR1Tests(unittest.TestCase):
    def test_frozen_protocol_dimensions_new_seeds_and_controllers(self):
        self.assertEqual(r1.SEEDS,tuple(range(9100,9124)))
        self.assertEqual(r1.SCENARIOS,("empty","obstacles"))
        self.assertEqual(r1.ARMS,(
            "recurrent","leaky_no_recurrence","instantaneous",
            "recurrent_blind","inertial_random",
        ))
        self.assertEqual(r1.N,16)
        self.assertEqual(r1.INPUTS,8)
        self.assertEqual(r1.OUTPUTS,2)
        self.assertEqual(r1.PARAMS,434)
        self.assertEqual(r1.HORIZON,600)
        self.assertEqual(len(r1.SEEDS)*len(r1.ARMS)*len(r1.SCENARIOS),240)
        self.assertEqual(len(r1.coefficient_flat(r1.generated_coefficients(9100))),434)
        with self.assertRaises(ValueError):
            r1.Circuit(9100,"scripted_wall_follow")
        with self.assertRaises(ValueError):
            r1.World(9100,"food_stage")
        with self.assertRaises(ValueError):
            r1.World(7300,"obstacles")

    def test_physical_world_impenetrable_and_respects_radius(self):
        world=r1.World(9100,"obstacles")
        self.assertEqual(len(world.rectangles),10)
        self.assertFalse(world.blocked(world.body.x,world.body.y))
        for x1,y1,x2,y2 in world.rectangles:
            self.assertTrue(world.blocked((x1+x2)/2,(y1+y2)/2))
        self.assertTrue(world.blocked(0.,0.))
        self.assertTrue(world.blocked(21.9,21.9))
        with self.assertRaises(ValueError):
            world.integrate((3.,0.))
        # Force a collision with a wall while holding a bounded motor command.
        world.body.x,world.body.y=.301,10.
        world.body.heading=math.pi
        world.body.speed=1.
        before=(world.body.x,world.body.y)
        record=world.integrate((1.,0.))
        self.assertEqual(record["collision"],1)
        self.assertEqual(record["position"],list(before))
        self.assertAlmostEqual(world.body.integrity,99.75)
        self.assertAlmostEqual(world.body.energy,99.98)

    def test_ray_distance_never_passes_solid_wall(self):
        world=r1.World(9101,"empty")
        world.body.x,world.body.y=1.,10.
        self.assertAlmostEqual(world.ray_distance(math.pi),.7,places=7)
        self.assertAlmostEqual(world.ray_distance(0.),r1.RAY_LENGTH,places=7)
        # Sensor can't magically give global map or route: eight numbers only.
        measurements=world.perceive()
        self.assertEqual(len(measurements),8)
        self.assertTrue(all(-1.-1e-9<=x<=1.+1e-9 for x in measurements))
        self.assertGreater(measurements[1],0.)

    def test_identical_weights_and_initial_pose_across_all_arms(self):
        initial_hash=set()
        pose=set()
        world_map=set()
        for arm in r1.ARMS:
            r=r1.simulate(9100,"obstacles",arm)
            initial_hash.add(r["initial_coefficients_sha256"])
            pose.add(tuple(r["initial_pose"]))
            world_map.add(r1.sha(r["rectangles"]))
            self.assertFalse(r["weights_updated"])
            self.assertEqual(r["initial_coefficients_sha256"],
                             r["final_coefficients_sha256"])
        self.assertEqual(len(initial_hash),1)
        self.assertEqual(len(pose),1)
        self.assertEqual(len(world_map),1)

    def test_recurrent_equals_written_evolution_from_same_input(self):
        c=r1.Circuit(9102,"recurrent")
        s=(1.,.1,.2,.3,.4,.0,.0,.0)
        _,a,_=c.forward(s)
        assert all(len(row)==8 for row in c.weights["w_input"])
        self.assertEqual(len(a),16)
        _,b,_=c.forward(s)
        self.assertNotEqual(a,b)
        first=[]
        for j in range(r1.N):
            z=(r1.dot(c.weights["w_input"][j],s)
               +c.weights["bias"][j])
            first.append(r1.ALPHA*math.tanh(z))
        self.assertTrue(all(abs(x-y)<1e-12 for x,y in zip(first,a)))
        second=[]
        for j in range(r1.N):
            z=(r1.dot(c.weights["w_input"][j],s)
               +r1.dot(c.weights["w_recurrent"][j],a)
               +c.weights["bias"][j])
            second.append((1-r1.ALPHA)*a[j]+r1.ALPHA*math.tanh(z))
        self.assertTrue(all(abs(x-y)<1e-12 for x,y in zip(second,b)))
        leak=r1.Circuit(9102,"leaky_no_recurrence")
        leak.forward(s)
        _,l,_=leak.forward(s)
        self.assertNotEqual(tuple(l),tuple(b))
        instantaneous=r1.Circuit(9102,"instantaneous")
        _,one,_=instantaneous.forward(s)
        _,two,_=instantaneous.forward(s)
        self.assertEqual(one,two)

    def test_true_sensor_blind_and_inertial_controls(self):
        s=(1.,.6,.8,.2,.4,-.2,1.,.3)
        c=r1.Circuit(9103,"recurrent_blind")
        motor,latent,inp=c.forward(s)
        self.assertTrue(all(inp[k]==0 for k in (1,2,3,6)))
        self.assertEqual(inp[4],s[4])
        self.assertEqual(len(latent),16)
        random_c=r1.Circuit(9103,"inertial_random")
        m,n,obs=random_c.forward(s)
        self.assertEqual(n,tuple([0.]*16))
        self.assertEqual(len(m),2)
        self.assertTrue(all(-1<=x<=1 for x in m))
        self.assertEqual(random_c.initial_hash,r1.Circuit(9103,"recurrent").initial_hash)

    def test_independent_maps_at_least_twenty(self):
        maps=set()
        for seed in r1.SEEDS:
            rects=r1.procedural_rectangles(seed)
            maps.add(r1.sha(rects))
            self.assertEqual(len(rects),10)
            for x1,y1,x2,y2 in rects:
                self.assertGreater(x2,x1)
                self.assertGreater(y2,y1)
                self.assertFalse(
                    x1-1.1<2.2<x2+1.1 and y1-1.1<2.2<y2+1.1
                )
        self.assertGreaterEqual(len(maps),20)

    def test_individual_repeatability_and_full_integrity(self):
        cases=[("empty","recurrent"),("obstacles","recurrent"),
               ("obstacles","leaky_no_recurrence"),
               ("obstacles","instantaneous"),
               ("obstacles","recurrent_blind"),
               ("obstacles","inertial_random")]
        for sc,arm in cases:
            a=r1.simulate(9104,sc,arm)
            b=r1.simulate(9104,sc,arm)
            self.assertEqual(r1.sha(a),r1.sha(b))
            results=r1._integrity([a])
            self.assertTrue(all(results.values()),(sc,arm,results))
            self.assertEqual(a["steps_alive"],len(a["trace"]))
            self.assertTrue(0<=a["steps_alive"]<=600)
            self.assertTrue(0<=a["terminal_energy"]<=100)
            self.assertTrue(0<=a["terminal_integrity"]<=100)


if __name__=="__main__":
    unittest.main()
