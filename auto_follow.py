"""Automatic following policy, independent of ROS and USB implementations."""
import math
import secrets
import threading
import time

from flask import Blueprint, jsonify, request
import numpy as np

from cat_markers import rotation
from gimbal_web import ControlError


class TargetUnavailable(ControlError):
    """A missing/stale observation can recover within the current follow lease."""


def angle_difference(a, b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


def approach_goal(target, robot, distance, goal_heading=None):
    dx, dy = target[0] - robot[0], target[1] - robot[1]
    separation = math.hypot(dx, dy)
    if separation <= distance:
        return None, separation
    heading = math.atan2(dy, dx)
    return (target[0] - distance * math.cos(heading),
            target[1] - distance * math.sin(heading),
            heading if goal_heading is None else goal_heading), separation


def gaze_angles(target, robot, config):
    """Map target -> yaw/pitch joint angles (right-hand rotations)."""
    base = rotation([0, 0, -math.degrees(robot[3])]) @ (
        np.asarray(target) - np.asarray(robot[:3]))
    ray = rotation(config['mount_rpy_deg']).T @ (base - config['mount_xyz_m'])
    yaw = math.degrees(math.atan2(ray[1], ray[0])) / config['yaw_sign']
    pitch = (0.0 if config.get('second_axis') == 'roll' else
             -math.degrees(math.atan2(ray[2], math.hypot(ray[0], ray[1]))) / config['pitch_sign'])
    return yaw, pitch


class AutoFollow:
    LEASE = 0.8
    TARGET_AGE = 0.5
    LOSS_HOLD = TARGET_AGE  # Bound motion on the last confirmed capture, not receipt time.
    REPLAN_PERIOD = 1.0
    REPLAN_DISTANCE = 0.25
    STANDOFF = 1.0
    GIMBAL_YAW_RATE = 120.0  # target degrees / second; motor response is separate
    GIMBAL_SECONDARY_RATE = 40.0
    CENTER_START = 20.0  # motor yaw degrees; hysteresis avoids repeated small turns
    CENTER_STOP = 8.0
    CENTER_REPLAN = math.radians(10.)

    def __init__(self, markers, gimbal, navigation, clock=time.monotonic):
        self.markers, self.gimbal, self.nav, self.clock = markers, gimbal, navigation, clock
        self.lock = threading.RLock()
        self.token = None
        self.gimbal_token = None
        self.sequence = 0
        self.deadline = 0
        self.message = '自动追踪未启用'
        self.distance = None
        self.goal = None
        self.sent_at = 0
        self.last_tick = None
        self.commanded = None
        self.result_seen = 0
        self.stop_event = threading.Event()
        self.thread = None
        self.manual_takeover = False
        self.last_seen_at = None
        self.waiting_for_target = False
        self.loss_paused = False
        self.centering = False
        self.goal_kind = None

    def start_worker(self):
        with self.lock:
            if self.thread is None:
                self.nav.start()
                self.thread = threading.Thread(target=self._run, daemon=True, name='cat-auto-follow')
                self.thread.start()

    def snapshot(self):
        with self.lock:
            nav = self.nav.snapshot()
            return dict(active=self.token is not None,
                        stopping=self.manual_takeover or (nav['busy'] and self.token is None),
                        ready=nav['ready'], message=self.message, distance=self.distance,
                        standoff=self.STANDOFF, waiting_for_target=self.waiting_for_target,
                        centering=self.centering)

    def _observation(self):
        observation = self.markers.follow_observation()
        if not observation:
            raise TargetUnavailable('目标位置已失效，请等待重新识别')
        if not math.isfinite(observation['captured_at']) or self.clock() < observation['captured_at']:
            raise ControlError('目标时间戳无效，已停止自动追踪')
        if self.clock() - observation['captured_at'] > self.TARGET_AGE:
            raise TargetUnavailable('目标位置已失效，请等待重新识别')
        config = observation['config']
        if (config['mode'] != 'gimbal' or observation.get('frame') != 'map'
                or not config.get('follow_secondary', config.get('follow_pitch', True))):
            raise ControlError('自动追踪需要 map 定位与云台姿态补偿')
        # The default centered mount has coincident axes and an aligned optical
        # frame. Refuse an unimplemented inverse kinematics instead of guessing.
        if (any(abs(v) > 1e-6 for v in config['camera_xyz_m'] + config['pitch_xyz_m'])
                or not np.allclose(config['camera_rpy_deg'], [-90, 0, -90])):
            raise ControlError('当前自动瞄准仅支持已配置的同轴、朝前相机近似')
        return observation

    def command(self, data):
        if not isinstance(data, dict):
            raise ValueError('请求必须为 JSON 对象')
        with self.lock:
            action = data.get('action')
            if action == 'stop':
                if data.get('token') is None or data.get('token') == self.token:
                    self._stop('自动追踪已停止')
                return {}
            if action == 'heartbeat':
                if self.token is None or data.get('token') != self.token:
                    raise ControlError('自动追踪已结束')
                if self.clock() >= self.deadline:
                    self._stop('网页失联，已取消自动追踪')
                    raise ControlError('控制已超时，请重新启用')
                self.deadline = self.clock() + self.LEASE
                return {}
            if action != 'start':
                raise ValueError('未知自动追踪操作')
            if self.manual_takeover:
                raise ControlError('键盘正在接管，请等待切换完成')
            if self.token:
                raise ControlError('已有自动追踪会话')
            try:
                observation = self._observation()
            except TargetUnavailable:
                observation = None
            nav = self.nav.snapshot()
            if not nav['ready']:
                raise ControlError(nav.get('error') or '等待导航服务与地图定位', 503)
            if nav['busy'] or nav['foreign_busy'] or nav['manual_active']:
                raise ControlError('请先停止其他导航任务和键盘驾驶')
            if observation is not None:
                yaw, pitch = gaze_angles(observation['position'], nav['robot'], observation['config'])
                if abs(yaw) > 90 or abs(pitch) > 30:
                    observation = None
            session = self.gimbal.command({'action': 'start'}, owner='auto')
            self.gimbal_token = session['token']
            self.commanded = (session['yaw'], session['pitch'])
            self.token = secrets.token_urlsafe(24)
            self.deadline = self.clock() + self.LEASE
            self.sequence = 0
            self.goal = None
            self.sent_at = -float('inf')
            self.last_tick = self.clock()
            self.result_seen = nav['result_seq']
            self.last_seen_at = observation['captured_at'] if observation is not None else None
            self.centering = False
            self.goal_kind = None
            self.waiting_for_target = observation is None
            self.loss_paused = False
            self.nav.set_following(True)
            self.message = '自动追踪已启用，等待可跟随目标' if observation is None else '自动追踪中'
            return {'token': self.token}

    def _stop(self, message):
        owned = self.token is not None or self.gimbal_token is not None
        self.token = None
        if owned:
            self.nav.set_following(False)
            self.nav.cancel()
        if self.gimbal_token:
            try:
                self.gimbal.command({'action': 'stop', 'token': self.gimbal_token})
            except ControlError:
                message += '；云台停止未确认'
        self.gimbal_token = None
        self.goal = None
        self.last_seen_at = None
        self.centering = False
        self.goal_kind = None
        self.waiting_for_target = self.loss_paused = False
        self.distance = None
        self.message = message

    def stop(self, message='自动追踪已停止'):
        with self.lock:
            self._stop(message)

    def take_manual_control(self):
        # Reserve the transition under the policy lock, but never hold it while
        # waiting for ROS. Status/stop must respond; duplicate starts must fail
        # immediately instead of starting unexpectedly after the transition.
        with self.lock:
            if self.manual_takeover:
                raise ControlError('键盘正在接管，请等待切换完成')
            self.manual_takeover = True
        try:
            self.stop('键盘接管，自动追踪已停止')
            self.nav.take_manual_control()
        finally:
            with self.lock:
                self.manual_takeover = False

    def _send_gimbal(self):
        self.gimbal.command(dict(action='target', token=self.gimbal_token,
                                 sequence=self.sequence, yaw=self.commanded[0], pitch=self.commanded[1]))
        self.sequence += 1

    def _wait_for_target(self, force_pause=False, reason=None):
        age = self.clock() - self.last_seen_at if self.last_seen_at is not None else float('inf')
        self.waiting_for_target = True
        self.distance = None
        self.last_tick = self.clock()
        # Hold the existing gaze, renew the USB lease and still detect gimbal faults.
        self._send_gimbal()
        if force_pause or age >= self.LOSS_HOLD:
            if not self.loss_paused:
                self.nav.cancel()
                self.goal = None
                self.goal_kind = None
                self.centering = False
                self.loss_paused = True
            self.message = reason or '自动追踪保持启用，已请求暂停导航，持续等待目标'
        else:
            # No new goal or extrapolated target is created during this short gap.
            self.message = '识别短暂中断，保持当前任务，等待重新确认'

    def tick(self):
        with self.lock:
            if not self.token:
                return
            try:
                if self.clock() >= self.deadline:
                    raise ControlError('网页失联，已取消自动追踪')
                nav = self.nav.snapshot()
                if not nav['ready'] or nav['foreign_busy']:
                    raise ControlError(nav.get('error') or '导航或定位不可用，已停止')
                if nav['result_seq'] != self.result_seen:
                    self.result_seen = nav['result_seq']
                    if nav['outcome'] == 'failed':
                        raise ControlError('导航失败，已停止自动追踪')
                try:
                    observation = self._observation()
                    # Replaying an old frame must not resume a paused task or refresh its age.
                    if (self.waiting_for_target and self.last_seen_at is not None
                            and observation['captured_at'] <= self.last_seen_at):
                        raise TargetUnavailable('等待新目标观测')
                except TargetUnavailable:
                    self._wait_for_target()
                    return
                yaw, pitch = gaze_angles(observation['position'], nav['robot'], observation['config'])
                if abs(yaw) > 90 or abs(pitch) > 30:
                    self._wait_for_target(force_pause=True,
                                          reason='目标超出云台范围，暂停导航并持续等待目标回到视野')
                    return
                self.last_seen_at = observation['captured_at']
                self.waiting_for_target = False
                if self.loss_paused:
                    self.sent_at = -float('inf')
                    self.loss_paused = False
                dt = min(.1, max(0, self.clock() - self.last_tick))
                self.last_tick = self.clock()
                desired = (yaw, pitch)
                self.commanded = tuple(old + max(-rate*dt, min(rate*dt, new-old))
                                       for old, new, rate in zip(self.commanded, desired,
                                           (self.GIMBAL_YAW_RATE, self.GIMBAL_SECONDARY_RATE)))
                self._send_gimbal()
                # Use the geometric yaw required by the fresh target, not the
                # rate-limited command. Convert the motor sign back to base yaw.
                if abs(yaw) >= self.CENTER_START:
                    self.centering = True
                elif abs(yaw) <= self.CENTER_STOP:
                    self.centering = False
                kind = 'align' if self.centering else 'approach'
                if self.goal_kind is not None and self.goal_kind != kind:
                    self.nav.cancel()
                    self.goal = self.goal_kind = None
                    self.sent_at = -float('inf')
                    self.message = '正在切换车体纠偏与平移跟随'
                    return
                goal, self.distance = approach_goal(observation['position'], nav['robot'],
                                                    self.STANDOFF, nav['robot'][3])
                if self.centering:
                    heading = nav['robot'][3] + math.radians(yaw * observation['config']['yaw_sign'])
                    goal = (*nav['robot'][:2], angle_difference(heading, 0.))
                    self.message = '车体低速对准目标，云台 Yaw 回中'
                # Stop approaching in the hold zone; resume with hysteresis.
                elif self.distance <= self.STANDOFF + .1:
                    self.nav.cancel()
                    self.goal = self.goal_kind = None
                    self.message = '已接近目标，保持注视'
                    return
                else:
                    self.message = '自动追踪中'
                if not self.centering and self.distance < self.STANDOFF + .3 and self.goal is None:
                    return
                if self.clock() - self.sent_at < self.REPLAN_PERIOD:
                    return
                moved = self.goal is None or math.hypot(goal[0]-self.goal[0], goal[1]-self.goal[1]) >= self.REPLAN_DISTANCE
                if self.centering and self.goal is not None:
                    moved = moved or abs(angle_difference(goal[2], self.goal[2])) >= self.CENTER_REPLAN
                if nav['busy']:
                    if moved:
                        self.nav.cancel()
                    return
                if moved or nav['outcome'] == 'succeeded':
                    if self.nav.send_goal(goal, align=self.centering):
                        self.goal, self.sent_at = goal, self.clock()
                        self.goal_kind = kind
            except (ControlError, ValueError, KeyError) as exc:
                self._stop(str(exc))

    def _run(self):
        while not self.stop_event.wait(.05):
            try:
                self.tick()
            except Exception as exc:
                with self.lock:
                    self._stop(f'自动追踪异常：{exc}')

    def close(self):
        self.stop('自动追踪服务已关闭')
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        self.nav.close()


def create_follow_blueprint(controller):
    blueprint = Blueprint('auto_follow', __name__)

    @blueprint.post('/api/teleop/enable')
    def enable_teleop():
        controller.start_worker()
        try:
            controller.take_manual_control()
            return jsonify(ok=True)
        except ControlError as exc:
            return jsonify(ok=False, error=str(exc)), exc.code
        except RuntimeError as exc:
            return jsonify(ok=False, error=str(exc)), 503

    @blueprint.get('/api/follow/status')
    def status():
        controller.start_worker()
        response = jsonify(controller.snapshot())
        response.headers['Cache-Control'] = 'no-store'
        return response

    @blueprint.post('/api/follow/control')
    def control():
        controller.start_worker()
        try:
            return jsonify(ok=True, **controller.command(request.get_json(silent=True)))
        except ValueError as exc:
            return jsonify(ok=False, error=str(exc)), 400
        except ControlError as exc:
            return jsonify(ok=False, error=str(exc)), exc.code

    return blueprint
