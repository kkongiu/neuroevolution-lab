"""FlyLoop R10: slow-critic teaching of fast automata, never fast direct damage."""
import copy
import unittest
from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r8 as r8
from neuroevolution import flyloop_r9 as r9
from neuroevolution import flyloop_r10 as r10


class R10Tests(unittest.TestCase):
    def test_prospective_seeds_six_arms_and_old_physics_unchanged(self):
        self.assertEqual(r10.SEEDS,tuple(range(11800,11812)))
        self.assertTrue(set(r10.SEEDS).isdisjoint(r9.SEEDS))
        self.assertEqual(len(r10.ARMS),6)
        self.assertEqual(r10.TRAIN_MACROS,120)
        self.assertEqual(r10.MACROS,240)
        self.assertEqual(r10.SUBSTEPS,4)
        self.assertAlmostEqual(r10.MICRO_DT,.05)
        self.assertEqual(r10.TEACHER_SHAM_MIN_AGE,8)
        self.assertEqual(r10.TEACHER_PROGRESS_MIN,.15)
        self.assertEqual(r10.SLOW_LR,.014)
        self.assertEqual(r10.FAST_SUPERVISED_LR,.025)
        self.assertEqual(r1.N,16)
        self.assertEqual(r1.INPUTS,8)
        self.assertEqual(r1.PARAMS,434)
        self.assertEqual(r10.GEOMETRIES,("obstacles","nearfield"))
        self.assertEqual(r10.REGIMES,("static","removal"))
        self.assertEqual(len(r10.SEEDS)*2*2*6,288)
        with self.assertRaises(ValueError):
            r10.World(11500,"nearfield","static")
        with self.assertRaises(ValueError):
            r10.simulate(11800,"nearfield","static","escape")

    def test_teacher_critic_initial_zero_and_no_update_before_real_episode(self):
        teacher=r10.SlowCritic(11800)
        w=r10.World(11800,"nearfield","static")
        self.assertEqual(teacher.weights,[[0.]*28,[0.]*28])
        cmd=(.4,.2)
        choice=teacher.choose(20,[.3]*16,w.perceive()[1:4],cmd)
        self.assertEqual(len(choice["candidates"]),5)
        self.assertEqual(teacher.updates,0)
        self.assertTrue(choice["exploratory"]) # fourth macro
        steps=[w.micro_integrate(choice["command"]) for _ in range(4)]
        outcome=teacher.complete(20,choice,steps)
        self.assertEqual(teacher.updates,1)
        self.assertTrue(all(0<=v<=1 for v in outcome["target"]))
        self.assertEqual(outcome["actual_slow_issued_motor"],
                         list(choice["command"]))
        self.assertFalse(outcome["validated"]) # exploration cannot teach
        snapshot=copy.deepcopy(teacher.weights)
        latest=teacher.choose(120,[.1]*16,w.perceive()[1:4],cmd)
        teacher.complete(120,latest,steps)
        self.assertEqual(teacher.weights,snapshot)
        self.assertEqual(teacher.updates,1)

    def test_teacher_validation_is_postphysical_nonexploration_only(self):
        teacher=r10.SlowCritic(11801)
        w=r10.World(11801,"obstacles","static")
        c=teacher.choose(21,[0.]*16,w.perceive()[1:4],(.7,0.))
        self.assertFalse(c["exploratory"])
        real=[w.micro_integrate(c["command"]) for _ in range(4)]
        judged=teacher.complete(21,c,real)
        self.assertEqual(
            judged["validated"],
            len(real)==4 and not judged["any_real_contact"] and
            judged["target"][0]>=r10.TEACHER_PROGRESS_MIN
        )

    def test_student_receives_teacher_motor_not_physical_reward(self):
        student=r10.TeacherStudent(11800)
        initial=copy.deepcopy(student.output)
        student.observe((.3,.2,.1,1.,.1,0.,.2))
        self.assertEqual(student.output,initial)
        student.fit_teacher(16,student.h,(.9,-.4),True)
        self.assertNotEqual(student.output,initial)
        self.assertEqual(student.supervised_updates,1)
        self.assertEqual(student.validated_supervised_updates,1)
        self.assertEqual(student.update_events,0) # no R9 physical learner
        with self.assertRaises(AssertionError):
            student.fit_teacher(120,student.h,(.9,.1),True)
        m,noise,yes=student.fast_motor(False)
        self.assertIsNone(m)
        self.assertFalse(yes)
        m,noise,yes=student.fast_motor(True)
        self.assertTrue(yes)
        self.assertTrue(all(-1<=v<=1 for v in m))

    def test_all_six_arms_same_real_training_trajectories(self):
        lives=[
            r10.simulate(11800,"nearfield","static",arm)
            for arm in r10.ARMS
        ]
        baseline=[
            (f["actual_executed_motor"],f["actual_micro_physics"],
             f["slow_16_hidden"])
            for f in lives[0]["raw_micro_trace"]
            if f["macro_tick"]<120
        ]
        for row in lives:
            self.assertEqual(
                baseline,
                [(f["actual_executed_motor"],f["actual_micro_physics"],
                  f["slow_16_hidden"])
                 for f in row["raw_micro_trace"] if f["macro_tick"]<120]
            )
            self.assertEqual(row["initial_original_slow_434_weights"],
                             lives[0]["initial_original_slow_434_weights"])
            self.assertEqual(row["final_original_slow_434_weights"],
                             row["initial_original_slow_434_weights"])
            self.assertEqual(row["final_slow_critic_56_weights"],
                             lives[0]["final_slow_critic_56_weights"])
            self.assertEqual(row["slow_critic_update_count"],120)
            self.assertTrue(all(not f["fast_activated"] for f in
                                row["raw_micro_trace"] if
                                f["macro_tick"]<120))
            self.assertTrue(all(r10.audit([row]).values()))

    def test_slow_teacher_and_naive_teacher_lesson_provenance(self):
        for arm in ("fast_naive","fast_teacher","fast_shuffled"):
            row=r10.simulate(11801,"nearfield","removal",arm)
            self.assertEqual(row["slow_critic_update_count"],120)
            self.assertEqual(
                row["fast_supervised_teacher_update_count"],
                len(row["actual_teaching_lessons"])
            )
            for entry in row["actual_teaching_lessons"]:
                t=entry["learned_after_macro"]
                src=entry["actual_original_slow_macro_source"]
                self.assertLess(t,120)
                for x,y in zip(
                    entry["actually_executed_slow_motor_label"],
                    row["macro_trace"][src]["slow_motor_physically_issued"]
                ):
                    self.assertAlmostEqual(x,y,places=8)
                if arm=="fast_teacher":
                    self.assertEqual(src,t)
                    self.assertTrue(
                        row["macro_trace"][t][
                          "slow_critic_actual_physical_macro_outcome"
                        ]["validated"]
                    )
                elif arm=="fast_shuffled":
                    self.assertLessEqual(src,t-8)
                    self.assertTrue(
                        row["macro_trace"][src][
                          "slow_critic_actual_physical_macro_outcome"
                        ]["validated"]
                    )
                else:
                    self.assertEqual(src,t)
            self.assertTrue(all(r10.audit([row]).values()))

    def test_static_and_removal_identical_prechange(self):
        for arm in ("slow_only","fast_teacher","fast_direct"):
            a=r10.simulate(11802,"nearfield","static",arm)
            b=r10.simulate(11802,"nearfield","removal",arm)
            self.assertEqual(
                [f for f in a["raw_micro_trace"] if f["macro_tick"]<120],
                [f for f in b["raw_micro_trace"] if f["macro_tick"]<120]
            )
            self.assertEqual(a["final_rectangles"],a["initial_ten_rectangles"])
            self.assertEqual(b["final_rectangles"],
                             b["initial_ten_rectangles"][::2])
            self.assertTrue(all(r10.audit([a,b]).values()))

    def test_deterministic_real_physical_repeated_life(self):
        one=r10.simulate(11803,"obstacles","removal","fast_teacher")
        two=r10.simulate(11803,"obstacles","removal","fast_teacher")
        self.assertEqual(r10.sha(one),r10.sha(two))
        self.assertTrue(all(r10.audit([one]).values()))


if __name__=="__main__":
    unittest.main()
