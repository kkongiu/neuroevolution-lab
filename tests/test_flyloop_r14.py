import unittest
from neuroevolution import flyloop_r14 as r

class MetabolicTests(unittest.TestCase):
    def test_food_and_energy(self):
        w=r.World(15100,"nearfield")
        w.rectangles=[]
        w.food=[(w.body.x,w.body.y)]
        w.body.speed=0.
        before=w.body.energy
        s=w.micro_integrate((0.,0.))
        self.assertAlmostEqual(s["food_gain"],12.)
        self.assertAlmostEqual(w.body.energy,before-s["metabolic_cost"]+s["food_gain"])
        self.assertEqual(w.food,[])
        self.assertEqual(w.micro_integrate((0.,0.))["food_gain"],0.)

    def test_energy_cap_and_death(self):
        w=r.World(15100,"nearfield")
        w.rectangles=[]
        w.food=[(w.body.x,w.body.y)]
        w.body.energy=94.
        s=w.micro_integrate((0.,0.))
        self.assertEqual(w.body.energy,100.)
        self.assertAlmostEqual(s["food_gain"]+s["discarded"],12.)
        w.food=[(w.body.x,w.body.y)]
        w.body.energy=.001
        s=w.micro_integrate((0.,0.))
        self.assertEqual(w.body.energy,0.)
        self.assertEqual(s["food_gain"],0.)
        with self.assertRaises(AssertionError):
            w.micro_integrate((0.,0.))

    def test_odor_and_obstacle(self):
        w=r.World(15100,"nearfield")
        w.body.x=w.body.y=5.
        w.body.heading=0.
        w.rectangles=[]
        w.food=[(7.,5.)]
        self.assertGreater(w.odor()[0],0.)
        w.rectangles=[(5.8,4.,6.2,6.)]
        self.assertEqual(w.odor(),(0.,0.,0.))

    def test_matched_food_and_freeze(self):
        results={a:r.simulate(15100,"nearfield",a,train=24,test=8) for a in r.ARMS}
        full=results["metabolic"]
        for a,row in results.items():
            self.assertEqual(row["training"]["metrics"]["updates"],24)
            self.assertEqual(row["test"]["metrics"]["updates"],0)
            if a!="no_food":
                self.assertEqual(row["initial_food"],full["initial_food"])
            for phase in ("training","test"):
                m=row[phase]["metrics"]
                self.assertAlmostEqual(m["initial_energy"]-m["metabolic_cost"]+m["food_gain"],m["final_energy"])
        self.assertEqual(results["no_food"]["training"]["metrics"]["food_gain"],0.)
        self.assertEqual(full["weights"],r.simulate(15100,"nearfield","metabolic",train=24,test=0)["weights"])
        self.assertEqual(full,r.simulate(15100,"nearfield","metabolic",train=24,test=8))
