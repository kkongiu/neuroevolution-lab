import math,unittest
from neuroevolution import flyloop_r25a as t,flyloop_r24c as old
class RetinaTests(unittest.TestCase):
    def test_geometry(self):
        w=t.r.WindowWorld(45100,0,'free');w.body.x=w.body.y=11.;w.body.heading=0.
        eye=t.r.CompoundEye()
        def visual(d,a):
            p=(11+d*math.cos(a),11+d*math.sin(a));w.food=[p]
            return t.retina(w,eye,p)
        sizes=[sum(visual(d,0))*eye.width for d in (.8,1.5,3.)]
        self.assertGreater(sizes[0],sizes[1]);self.assertGreater(sizes[1],sizes[2])
        for d,size in zip((.8,1.5,3.),sizes):
            self.assertAlmostEqual(size,2*math.asin(.25/d))
        a=visual(1.5,.7);b=visual(1.5,-.7)
        for x,y in zip(a,reversed(b)):self.assertAlmostEqual(x,y)
        self.assertAlmostEqual(sum(visual(1.5,math.pi/2))*eye.width,math.asin(.25/1.5))
        self.assertEqual(sum(visual(1.5,math.pi)),0.)
        w.food=[];self.assertEqual(sum(t.retina(w,eye,(12.,11.))),0.)
    def test_odor_regression(self):
        for angle in (0,90,180):
            a=old.episode(35106,angle,'ground',.6,'odor',30)
            b=t.episode(35106,angle,3.,.6,'odor',30)
            for x,y in zip(a['trace'],b['trace']):
                for k in ('position','heading','speed','command','cue','energy'):
                    self.assertEqual(x[k],y[k],k)
            self.assertEqual(a['steps'],b['steps'])
            self.assertEqual(a['meal'],b['meal'])
    def test_replay(self):
        for c in t.CUES:
            a=t.episode(37100,45,.8,.6,c,10)
            self.assertEqual(a,t.episode(37100,45,.8,.6,c,10))
            self.assertTrue(a['first_window_complete'])
