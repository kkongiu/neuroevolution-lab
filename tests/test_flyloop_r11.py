"""R11 scientific controls: slow-experienced extended collision recovery teaches fast."""
import copy
import math
import unittest

from neuroevolution import flyloop_r1 as r1
from neuroevolution import flyloop_r9 as r9
from neuroevolution import flyloop_r10 as r10
from neuroevolution import flyloop_r11 as r11


class R11Tests(unittest.TestCase):
    def test_locked_fresh_population_original_same_movement_physics(self):
        self.assertEqual(r11.SEEDS,tuple(range(12100,12112)))
        self.assertTrue(set(r11.SEEDS).isdisjoint(r10.SEEDS))
        self.assertEqual(r11.GEOMETRIES,("obstacles","nearfield"))
        self.assertEqual(r11.REGIMES,("static","removal"))
        self.assertEqual(len(r11.ARMS),6)
        self.assertEqual(12*2*2*6,288)
        self.assertEqual(r11.MACROS,260)
        self.assertEqual(r11.TRAIN_MACROS,160)
        self.assertEqual(r11.CONTACT_EXPLORATION_MACROS,12)
        self.assertEqual(r11.CONTACT_WINDOW_MICRO,32)
        self.assertEqual(r11.RECOVERY_FREE_MICROS,4)
        self.assertEqual(r11.RECOVERY_DISTANCE,.035)
        self.assertEqual(r1.PARAMS,434)
        self.assertEqual(r9.FAST_NEURONS,4)
        self.assertEqual(r11.MICRO_DT,.05)
        with self.assertRaises(ValueError):
            r11.World(11800,"nearfield","static")
        with self.assertRaises(ValueError):
            r11.World(12100,"food","static")
        with self.assertRaises(ValueError):
            r11.simulate(12100,"nearfield","static","attack")

    def test_slow_three_28_heads_zero_real_progress_recovery_delayed(self):
        c=r11.SlowRecoveryCritic(12100)
        w=r11.World(12100,"nearfield","static")
        self.assertEqual(c.weights,[[0.]*28 for _ in range(3)])
        first=c.choose(20,[.0]*16,w.perceive()[1:4],(.3,.2),-1)
        self.assertEqual(len(first["predictions"]),5)
        self.assertEqual(first["exploration"],"real_regular_menu_uniform")
        self.assertEqual(c.progress_updates,0)
        real=[w.micro_integrate(first["command"]) for _ in range(4)]
        judged=c.complete_actual_macro(20,first,real)
        self.assertEqual(c.progress_updates,1)
        self.assertEqual(c.recovery_updates,0)
        self.assertTrue(0<=judged["actual_normalized_distance"]<=1)
        before=copy.deepcopy(c.weights[2])
        success={
            "_slow_onset_phi":first["pre_action_phi"],
            "verified_recovered":True
        }
        c.train_delayed_recovery(20,[success])
        self.assertEqual(c.recovery_updates,1)
        self.assertNotEqual(before,c.weights[2])
        snapshot=copy.deepcopy(c.weights)
        other=c.choose(160,[.0]*16,w.perceive()[1:4],(.1,.1),-1)
        c.complete_actual_macro(160,other,real)
        self.assertEqual(c.weights,snapshot)
        with self.assertRaises(AssertionError):
            c.train_delayed_recovery(160,[success])

    def test_real_collision_to_four_free_steps_only_after_future_experienced(self):
        tracker=r11.RecoveryEpisodes()
        hidden=[.2,.3,.4,.5]
        sensors=(.9,.8,.7,1.,.1,.0,.4)
        phi=[.0]*28
        cmd=(.71,-.81)
        tracker.before_micro(0,hidden,sensors)
        out,start=tracker.after_real_micro(
            0,0,{"collision":1,"moved":0.},cmd,hidden,sensors,phi
        )
        self.assertEqual(len(out),0)
        self.assertTrue(start)
        self.assertEqual(tracker.contact_onsets,1)
        self.assertIsNone(tracker.pending["first_post_contact_fast_hidden"])
        for k in range(1,5):
            tracker.before_micro(k,hidden,sensors)
            done,started=tracker.after_real_micro(
                0,k,{"collision":0,"moved":.010},cmd,hidden,sensors,phi
            )
            self.assertFalse(started)
            if k<4:
                self.assertEqual(done,[])
                self.assertEqual(tracker.verified_lessons(),[])
            else:
                self.assertEqual(len(done),1)
                self.assertTrue(done[0]["verified_recovered"])
                self.assertEqual(
                    done[0]["first_post_contact_micro"],1
                )
                self.assertEqual(
                    done[0]["actual_first_recovery_motor"],list(cmd)
                )
                self.assertEqual(
                    done[0]["first_recovery_motor_micro"],1
                )
        self.assertEqual(len(tracker.verified_lessons()),1)
        self.assertEqual(tracker.verified_lessons()[0][
            "first_post_contact_fast_hidden"],hidden)

    def test_contact_window_failure_is_not_fake_training_success(self):
        tracker=r11.RecoveryEpisodes()
        for tick in range(33):
            tracker.before_micro(tick,[.0]*4,[.0]*7)
            outcome,created=tracker.after_real_micro(
                tick//4,tick,{"collision":1,"moved":0.},
                (0.,0.),[.0]*4,[.0]*7,[.0]*28
            )
            if tick==0:
                self.assertTrue(created)
            if tick==32:
                self.assertEqual(len(outcome),1)
                self.assertFalse(outcome[0]["verified_recovered"])
        self.assertEqual(tracker.verified_lessons(),[])

    def test_matched_lesson_count_shuffled_real_motors_and_random_labels(self):
        three=[
            {
                "collision_micro":i*10,
                "first_post_contact_micro":i*10+1,
                "first_post_contact_fast_hidden":[.1+i*.2,.2,.3,.4],
                "actual_first_recovery_motor":[.5-i*.3,-.8+i*.5],
                "first_recovery_motor_micro":i*10+5
            } for i in range(3)
        ]
        safe=[{"macro":2,"motor":[.6,.7]}]
        all_sets={}
        for arm in r11.SUPERVISED_ARMS:
            f=r11.FastStudent(12100)
            examples=r11.consolidate_student(
                12100,arm,f,three,safe
            )
            self.assertEqual(f.supervised_updates,3)
            self.assertEqual(len(examples),3)
            all_sets[arm]=examples
        snapshots=[
            x["actual_contact_fast_four_hidden"] for x in all_sets[
                "fast_recovery_true"
            ]
        ]
        self.assertTrue(all([
            x["actual_contact_fast_four_hidden"] for x in group]==snapshots
            for group in all_sets.values()
        ))
        true=all_sets["fast_recovery_true"]
        shuffled=all_sets["fast_recovery_shuffled"]
        self.assertNotEqual(
            [x["supervised_motor_label"] for x in true],
            [x["supervised_motor_label"] for x in shuffled]
        )
        self.assertEqual(
            sorted(tuple(x["supervised_motor_label"]) for x in true),
            sorted(tuple(x["supervised_motor_label"]) for x in shuffled)
        )
        self.assertTrue(all(
            x["motor_label_source_real_micro"] is None for x in
            all_sets["fast_recovery_random"]
        ))

    def test_exact_slow_only_training_for_six_arms_and_raw_replay(self):
        lives=[
            r11.simulate(12100,"nearfield","removal",arm)
            for arm in r11.ARMS
        ]
        baseline=[
            (f["actually_executed_analog_motor"],
             f["actual_physical_body_after_micro"],
             f["original_slow_sixteen_hidden"])
            for f in lives[0]["raw_micro_trace"]
            if f["macro_tick"]<160
        ]
        for life in lives:
            got=[
                (f["actually_executed_analog_motor"],
                 f["actual_physical_body_after_micro"],
                 f["original_slow_sixteen_hidden"])
                for f in life["raw_micro_trace"]
                if f["macro_tick"]<160
            ]
            self.assertEqual(got,baseline)
            self.assertEqual(
                life["original_434_slow_coefficients_initial"],
                life["original_434_slow_coefficients_final"]
            )
            self.assertTrue(all(
                not f["fast_actually_overrode_motor"]
                for f in life["raw_micro_trace"] if f["macro_tick"]<160
            ))
        for arm in r11.SUPERVISED_ARMS:
            life=next(x for x in lives if x["arm"]==arm)
            n=len(life["real_successful_recovery_episodes"])
            count=n if life["same_life_real_safe_macro_motors"] else 0
            self.assertEqual(life["supervised_fast_student_updates"],count)
        audit=r11.audit(lives)
        self.assertTrue(all(audit.values()),audit)

    def test_matched_static_removal_until_160_and_obstacles_really_remove(self):
        for arm in ("slow_only","fast_recovery_true","fast_recovery_shuffled"):
            fixed=r11.simulate(12101,"obstacles","static",arm)
            changed=r11.simulate(12101,"obstacles","removal",arm)
            pre=lambda row: [
                (f["actually_executed_analog_motor"],
                 f["actual_physical_body_after_micro"])
                for f in row["raw_micro_trace"] if f["macro_tick"]<160
            ]
            self.assertEqual(pre(fixed),pre(changed))
            self.assertEqual(
                changed["final_rectangles"],
                changed["initial_10_rectangles"][::2]
            )
            self.assertEqual(
                fixed["final_rectangles"],fixed["initial_10_rectangles"]
            )
            self.assertTrue(all(r11.audit([fixed,changed]).values()))

    def test_life_is_byte_deterministic_without_offline_physical_futures(self):
        a=r11.simulate(12102,"nearfield","removal","fast_recovery_true")
        b=r11.simulate(12102,"nearfield","removal","fast_recovery_true")
        self.assertEqual(r11.sha(a),r11.sha(b))
        self.assertTrue(all(r11.audit([a]).values()))


if __name__=="__main__":
    unittest.main()
