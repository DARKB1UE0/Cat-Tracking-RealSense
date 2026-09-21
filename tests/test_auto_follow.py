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
        self.goals = []; self.cancels = 0; self.following = False
    def snapshot(self): return self.state.copy()
    def set_following(self, value): self.following = value
    def cancel(self): self.cancels += 1
    def send_goal(self, pose):
        self.goals.append(pose); self.state.update(busy=True, outcome=None); return True
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

    def test_missing_target_and_stale_target_stop(self):
        self.start(); self.advance()
        self.advance(1.2, refresh=False)
        self.assertFalse(self.follow.snapshot()['active'])
        self.assertFalse(self.nav.following)
        self.assertGreater(self.nav.cancels, 0)
        self.assertEqual(self.gimbal.commands[-1][0]['action'], 'stop')

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

    def test_old_stop_cannot_stop_new_owner_and_out_of_view_stops(self):
        self.start(); self.follow.command(dict(action='stop', token='old'))
        self.assertTrue(self.follow.snapshot()['active'])
        self.observation['position'] = [-3., 0, .15]; self.advance()
        self.assertFalse(self.follow.snapshot()['active'])

    def test_reject_unsupported_or_stale_geometry(self):
        self.observation['captured_at'] = self.now-2
        with self.assertRaises(ControlError): self.start()
        self.observation['captured_at'] = self.now
        self.config['camera_xyz_m'] = [.1, 0, 0]
        with self.assertRaises(ControlError): self.start()
        self.assertFalse(self.gimbal.commands)


if __name__ == '__main__': unittest.main()
