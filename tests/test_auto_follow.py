"""Following policy checks with no ROS, USB or robot traffic."""
import copy
import math
from pathlib import Path
import unittest
import threading

from auto_follow import AutoFollow, approach_goal, gaze_angles
from cat_markers import load_config, mount_point
from gimbal_web import ControlError


class Navigation:
    def __init__(self):
        self.state = dict(ready=True, busy=False, foreign_busy=False, manual_active=False,
                          robot=(0, 0, 0, 0), result_seq=0, outcome=None)
        self.goals = []; self.alignments = []; self.cancels = 0; self.following = False
    def snapshot(self): return self.state.copy()
    def set_following(self, value): self.following = value
    def cancel(self): self.cancels += 1
    def send_goal(self, pose, align=False):
        self.goals.append(pose); self.alignments.append(align)
        self.state.update(busy=True, outcome=None); return True
    def finish(self, outcome='canceled'):
        self.state.update(busy=False, outcome=outcome, result_seq=self.state['result_seq']+1)


class Gimbal:
    def __init__(self): self.commands = []; self.fault = False
    def command(self, data, owner='manual'):
        if self.fault: raise ControlError('云台反馈中断')
        self.commands.append((data, owner))
        return dict(token='gimbal-session', yaw=0., pitch=10.)


class FollowTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.config = load_config(Path(__file__).resolve().parents[1]/'camera_mount.json')
        self.observation = dict(position=[3., 0., .15], captured_at=self.now, frame='map', config=self.config)
        self.nav = Navigation(); self.gimbal = Gimbal()
        self.follow = AutoFollow(self, self.gimbal, self.nav, clock=lambda: self.now)
    def follow_observation(self): return copy.deepcopy(self.observation)
    def start(self): self.token = self.follow.command(dict(action='start'))['token']
    def advance(self, seconds=.05, refresh=True):
        # Simulate periodic page heartbeats and fresh camera frames.
        remaining = seconds
        while remaining > 1e-9:
            dt = min(.1, remaining); self.now += dt; remaining -= dt
            if refresh: self.observation['captured_at'] = self.now
            if self.follow.token: self.follow.command(dict(action='heartbeat', token=self.token))
            self.follow.tick()

    def test_standoff_and_gaze(self):
        goal, distance = approach_goal([3, 4, 0], [0, 0, 0, 0], 1.)
        self.assertAlmostEqual(distance, 5.)
        self.assertAlmostEqual(math.hypot(3-goal[0], 4-goal[1]), 1.)
        self.assertIsNone(approach_goal([.5, 0, 0], [0, 0, 0, 0], 1.)[0])
        yaw, secondary = gaze_angles([2, 2, 1], [0, 0, 0, math.pi/4], self.config)
        self.assertAlmostEqual(yaw, 0.)
        self.assertEqual(secondary, 0.)

    def test_gimbal_yaw_catches_up_faster_without_speeding_roll(self):
        self.observation['position'] = [3., 3., .15]
        self.start(); self.advance(.05)
        self.assertAlmostEqual(self.follow.commanded[0], 6.)
        self.assertAlmostEqual(self.follow.commanded[1], 8.)
        self.advance(.35)
        self.assertAlmostEqual(self.follow.commanded[0], 45.)
        self.assertGreaterEqual(self.follow.commanded[1], 0.)
        self.advance(.2)
        self.assertAlmostEqual(self.follow.commanded[0], 45.)
        self.assertAlmostEqual(self.follow.commanded[1], 0.)

    def test_base_centers_gimbal_then_translates_with_hysteresis(self):
        self.nav.state['robot'] = (0., 0., 0., .2)
        self.observation['position'] = [3., 2., .15]
        self.start(); self.advance()
        heading = math.atan2(2., 3.)
        self.assertTrue(self.nav.alignments[-1])
        self.assertEqual(self.nav.goals[-1][:2], (0., 0.))
        self.assertAlmostEqual(self.nav.goals[-1][2], heading)
        self.assertGreater(self.follow.commanded[0], 0.)
        self.nav.state['robot'] = (0., 0., 0., heading-math.radians(12))
        self.advance(.1)
        self.assertTrue(self.follow.snapshot()['centering'])
        self.nav.state['robot'] = (0., 0., 0., heading-math.radians(6))
        self.advance(.1)
        self.assertFalse(self.follow.snapshot()['centering'])
        self.assertGreater(self.nav.cancels, 0)
        self.assertEqual(len(self.nav.goals), 1)  # Still waiting for old action terminal.
        self.nav.finish(); self.advance()
        self.assertFalse(self.nav.alignments[-1])
        self.assertAlmostEqual(math.hypot(3.-self.nav.goals[-1][0], 2.-self.nav.goals[-1][1]), 1.)
        self.nav.state['robot'] = (0., 0., 0., heading-math.radians(12))
        self.advance(1.1)
        self.assertFalse(self.follow.snapshot()['centering'])

    def test_centering_at_standoff_and_motor_sign(self):
        for sign in (1, -1):
            self.setUp(); self.config['yaw_sign'] = sign
            self.observation['position'] = [.8, .4, .15]
            self.start(); self.advance()
            self.assertTrue(self.nav.alignments[-1])
            self.assertAlmostEqual(self.nav.goals[-1][2], math.atan2(.4, .8))
            self.assertEqual(self.nav.goals[-1][:2], (0., 0.))
            self.nav.state['robot'] = (0., 0., 0., math.atan2(.4, .8))
            self.advance(); self.nav.finish(); self.advance()
            self.assertFalse(self.follow.snapshot()['centering'])
            self.assertEqual(len(self.nav.goals), 1)  # Hold distance; no approach.
            self.assertAlmostEqual(self.follow.commanded[0], 0.)

    def test_translation_cancels_before_centering_and_small_offsets_do_not_turn(self):
        self.start(); self.advance()
        self.assertFalse(self.nav.alignments[-1])
        self.observation['position'] = [3., .5, .15]
        self.advance(.1)
        self.assertFalse(self.follow.snapshot()['centering'])
        self.assertEqual(self.nav.cancels, 0)
        self.observation['position'] = [3., 2., .15]
        self.advance(.1)
        self.assertTrue(self.follow.snapshot()['centering'])
        self.assertGreater(self.nav.cancels, 0)
        self.assertEqual(len(self.nav.goals), 1)
        self.advance(.1)
        self.assertEqual(len(self.nav.goals), 1)
        self.nav.finish(); self.advance()
        self.assertTrue(self.nav.alignments[-1])
        self.assertEqual(self.nav.goals[-1][:2], (0., 0.))

    def test_heading_replan_wrap_and_loss_during_centering(self):
        self.nav.state['robot'] = (0., 0., 0., math.radians(150))
        self.observation['position'] = [3*math.cos(math.radians(179)), 3*math.sin(math.radians(179)), .15]
        self.start(); self.advance()
        self.assertTrue(self.nav.alignments[-1])
        self.observation['position'] = [3*math.cos(math.radians(-179)), 3*math.sin(math.radians(-179)), .15]
        self.advance(1.1)
        self.assertEqual(self.nav.cancels, 0)  # Two degrees, not 358.
        self.observation['position'] = [3*math.cos(math.radians(-160)), 3*math.sin(math.radians(-160)), .15]
        self.advance(1.1)
        self.assertGreater(self.nav.cancels, 0)
        self.nav.finish(); self.advance()
        self.assertAlmostEqual(self.nav.goals[-1][2], math.radians(-160))
        self.observation = None; self.advance(1., refresh=False)
        self.assertTrue(self.follow.snapshot()['waiting_for_target'])
        self.assertFalse(self.follow.snapshot()['centering'])

    def test_manual_handoff_keeps_status_and_stop_responsive_and_rejects_reentry(self):
        entered, release, responsive = threading.Event(), threading.Event(), threading.Event()
        errors = []
        def wait_for_ros():
            entered.set()
            if not release.wait(2): raise RuntimeError('mock ROS timeout')
        self.nav.take_manual_control = wait_for_ros
        def takeover():
            try: self.follow.take_manual_control()
            except Exception as exc: errors.append(exc)
        def query_and_stop():
            self.assertTrue(self.follow.snapshot()['stopping'])
            self.follow.stop()
            responsive.set()
        self.start()
        worker = threading.Thread(target=takeover)
        worker.start()
        query = threading.Thread(target=query_and_stop)
        try:
            self.assertTrue(entered.wait(1))
            query.start()
            self.assertTrue(responsive.wait(.5), 'status/stop blocked by ROS cancellation')
            with self.assertRaises(ControlError): self.start()
            with self.assertRaises(ControlError): self.follow.take_manual_control()
        finally:
            release.set(); worker.join(2)
            if query.ident: query.join(2)
        self.assertEqual(errors, [])
        self.assertFalse(self.follow.manual_takeover)
        self.assertFalse(self.follow.snapshot()['active'])

    def test_manual_handoff_failure_releases_transition(self):
        def failed(): raise RuntimeError('navigation unavailable')
        self.nav.take_manual_control = failed
        with self.assertRaises(RuntimeError): self.follow.take_manual_control()
        self.assertFalse(self.follow.manual_takeover)
        self.start()

    def test_roll_compensation_preserves_forward_axis(self):
        result = mount_point([1, 0, 2], self.config, dict(connected=True, fault=0, yaw=0, pitch=30))
        self.assertAlmostEqual(result[0], 2.)
        self.assertAlmostEqual(result[1], -math.cos(math.pi/6))
        self.assertAlmostEqual(result[2], .15-.5)

    def test_start_approach_and_hold_one_metre(self):
        self.start(); self.advance()
        self.assertEqual(self.gimbal.commands[0][1], 'auto')
        self.assertEqual(self.nav.goals[0], (2., 0., 0.))
        self.assertAlmostEqual(self.gimbal.commands[-1][0]['pitch'], 8.)
        self.nav.state['robot'] = (2., 0, 0, 0)
        self.advance(); self.assertGreater(self.nav.cancels, 0)
        self.assertTrue(self.follow.snapshot()['active'])
        self.nav.finish(); self.advance(.4)
        self.assertEqual(self.gimbal.commands[-1][0]['pitch'], 0.)
        self.assertEqual(len(self.nav.goals), 1)
        self.assertEqual(self.follow.snapshot()['distance'], 1.)

    def test_stale_target_pauses_indefinitely_until_manual_stop(self):
        self.start(); self.advance()
        self.advance(1.2, refresh=False)
        self.assertTrue(self.follow.snapshot()['active'])
        self.assertTrue(self.follow.snapshot()['waiting_for_target'])
        self.assertTrue(self.nav.following)
        self.assertGreater(self.nav.cancels, 0)
        self.assertIsNone(self.follow.snapshot()['distance'])
        self.advance(60., refresh=False)
        self.assertTrue(self.follow.snapshot()['active'])
        self.assertTrue(self.follow.snapshot()['waiting_for_target'])
        self.assertEqual(self.follow.token, self.token)
        self.assertEqual(len(self.nav.goals), 1)
        self.follow.command(dict(action='stop', token=self.token))
        self.assertFalse(self.follow.snapshot()['active'])
        self.assertFalse(self.nav.following)
        self.assertGreater(self.nav.cancels, 0)
        self.assertEqual(self.gimbal.commands[-1][0]['action'], 'stop')

    def test_short_flicker_keeps_session_goal_and_gimbal_lease(self):
        self.start(); self.advance()
        saved = copy.deepcopy(self.observation)
        commanded = self.follow.commanded
        self.observation = None
        self.advance(.3, refresh=False)
        self.assertTrue(self.follow.snapshot()['waiting_for_target'])
        self.assertEqual(self.follow.token, self.token)
        self.assertEqual(self.nav.cancels, 0)
        self.assertEqual(len(self.nav.goals), 1)
        self.assertEqual(self.follow.commanded, commanded)
        self.assertEqual(self.gimbal.commands[-1][0]['action'], 'target')
        self.observation = saved
        self.advance()
        self.assertFalse(self.follow.snapshot()['waiting_for_target'])
        self.assertEqual(self.nav.cancels, 0)
        self.assertEqual(len(self.nav.goals), 1)

    def test_pause_waits_for_navigation_terminal_before_resuming(self):
        self.start(); self.advance()
        saved = copy.deepcopy(self.observation)
        self.observation = None
        self.advance(.7, refresh=False)
        self.assertTrue(self.follow.snapshot()['active'])
        self.assertTrue(self.follow.loss_paused)
        self.assertEqual(self.nav.cancels, 1)
        self.observation = saved
        self.observation['position'][0] = 4.
        self.advance()
        self.assertEqual(len(self.nav.goals), 1)
        self.nav.finish()
        self.advance()
        self.assertEqual(self.follow.token, self.token)
        self.assertEqual(self.nav.goals[-1], (3., 0., 0.))
        self.assertEqual(len(self.nav.goals), 2)

    def test_old_capture_cannot_resume_waiting_but_fresh_target_can(self):
        self.start(); self.advance()
        saved = copy.deepcopy(self.observation)
        self.observation = None
        self.advance(.6, refresh=False)
        self.nav.finish()
        self.observation = saved
        self.advance(.2, refresh=False)
        self.assertTrue(self.follow.snapshot()['waiting_for_target'])
        self.assertEqual(len(self.nav.goals), 1)
        self.advance(60., refresh=False)
        self.assertTrue(self.follow.snapshot()['active'])
        self.assertTrue(self.follow.snapshot()['waiting_for_target'])
        self.assertEqual(len(self.nav.goals), 1)
        self.advance(.1)
        self.assertTrue(self.follow.snapshot()['active'])
        self.assertFalse(self.follow.snapshot()['waiting_for_target'])
        self.assertEqual(len(self.nav.goals), 2)

    def test_start_without_target_waits_then_follows_and_manual_stop_cannot_rearm(self):
        saved = copy.deepcopy(self.observation)
        self.observation = None
        self.start(); self.advance(refresh=False)
        self.advance(60., refresh=False)
        self.assertTrue(self.follow.snapshot()['waiting_for_target'])
        self.assertEqual(self.nav.goals, [])
        self.assertEqual(self.follow.commanded, (0., 10.))
        self.observation = saved
        self.advance()
        self.assertEqual(len(self.nav.goals), 1)
        self.assertFalse(self.follow.snapshot()['waiting_for_target'])
        self.follow.command(dict(action='stop', token=self.token))
        self.advance(2.)
        self.assertFalse(self.follow.snapshot()['active'])
        self.assertEqual(len(self.nav.goals), 1)

    def test_faults_and_page_loss_still_stop_during_reacquisition(self):
        for failure in ('gimbal', 'localization', 'navigation', 'page', 'stop'):
            with self.subTest(failure=failure):
                self.setUp(); self.start(); self.advance()
                self.observation = None
                self.advance(.6, refresh=False)
                if failure == 'gimbal': self.gimbal.fault = True
                elif failure == 'localization': self.nav.state['ready'] = False
                elif failure == 'navigation': self.nav.finish('failed')
                elif failure == 'page': self.now += .81
                else: self.follow.stop('手动停止')
                self.follow.tick()
                self.assertFalse(self.follow.snapshot()['active'])
                self.assertFalse(self.nav.following)

    def test_repeated_person_confirmation_flicker_does_not_end_session(self):
        from target_tracking import PersonConfirmation
        selector = PersonConfirmation()
        candidate = dict(box=(0, 0, 80, 200), match_score=.9, match_eligible=True)
        self.start(); self.advance()
        for _ in range(4):
            selector.select([])
            self.observation = None
            self.advance(.08, refresh=False)
            for _ in range(3):
                target = selector.select([candidate])
                self.observation = (dict(position=[3., 0, .15], captured_at=self.now,
                                         frame='map', config=self.config) if target else None)
                self.advance(.08, refresh=False)
                self.assertTrue(self.follow.snapshot()['active'])
        self.assertEqual(self.follow.token, self.token)
        self.assertEqual(len(self.nav.goals), 1)
        self.assertEqual(self.nav.cancels, 0)

    def test_page_lease_expiry_cannot_rearm(self):
        self.start(); self.now += .81
        with self.assertRaises(ControlError): self.follow.command(dict(action='heartbeat', token=self.token))
        self.follow.tick()
        self.assertFalse(self.follow.snapshot()['active'])
        with self.assertRaises(ControlError): self.follow.command(dict(action='heartbeat', token=self.token))

    def test_replan_waits_for_cancel_result(self):
        self.start(); self.advance()
        self.observation['position'][0] = 4.
        self.advance(1.1)
        self.assertGreater(self.nav.cancels, 0)
        self.assertEqual(len(self.nav.goals), 1)
        self.nav.finish(); self.advance()
        self.assertEqual(len(self.nav.goals), 2)
        self.assertEqual(self.nav.goals[-1], (3., 0., 0.))

    def test_foreign_goal_manual_drive_and_no_localization_refuse_start(self):
        for key, value in [('ready', False), ('busy', True), ('foreign_busy', True), ('manual_active', True)]:
            original = self.nav.state[key]; self.nav.state[key] = value
            with self.assertRaises(ControlError): self.start()
            self.nav.state[key] = original
        self.assertFalse(self.gimbal.commands)

    def test_navigation_failure_and_gimbal_fault_stop(self):
        self.start(); self.advance(); self.nav.finish('failed'); self.advance()
        self.assertFalse(self.follow.snapshot()['active'])
        self.start(); self.gimbal.fault = True; self.advance()
        self.assertFalse(self.follow.snapshot()['active'])

    def test_old_stop_cannot_stop_new_owner_and_out_of_view_waits(self):
        self.start(); self.follow.command(dict(action='stop', token='old'))
        self.assertTrue(self.follow.snapshot()['active'])
        self.observation['position'] = [-3., 0, .15]; self.advance()
        self.assertTrue(self.follow.snapshot()['active'])
        self.assertTrue(self.follow.snapshot()['waiting_for_target'])
        self.assertGreater(self.nav.cancels, 0)
        self.advance(5.)
        self.assertEqual(self.nav.goals, [])
        self.observation['position'] = [3., 0, .15]; self.advance()
        self.assertFalse(self.follow.snapshot()['waiting_for_target'])
        self.assertEqual(len(self.nav.goals), 1)

    def test_start_with_stale_target_waits_but_invalid_geometry_is_rejected(self):
        self.observation['captured_at'] = self.now-2
        self.start(); self.advance(.1, refresh=False)
        self.assertTrue(self.follow.snapshot()['waiting_for_target'])
        self.assertEqual(self.nav.goals, [])
        self.follow.stop()
        self.gimbal.commands.clear()
        self.observation['captured_at'] = self.now
        self.config['camera_xyz_m'] = [.1, 0, 0]
        with self.assertRaises(ControlError): self.start()
        self.assertFalse(self.gimbal.commands)


if __name__ == '__main__': unittest.main()
