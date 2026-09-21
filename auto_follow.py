"""Automatic following policy, independent of ROS and USB implementations."""
import math
import secrets
import threading
import time

from flask import Blueprint, jsonify, request
import numpy as np

from cat_markers import rotation
from gimbal_web import ControlError


def approach_goal(target, robot, distance):
    dx, dy = target[0] - robot[0], target[1] - robot[1]
    separation = math.hypot(dx, dy)
    if separation <= distance:
        return None, separation
    heading = math.atan2(dy, dx)
    return (target[0] - distance * math.cos(heading),
            target[1] - distance * math.sin(heading), heading), separation


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
    TARGET_AGE = 1.0
    REPLAN_PERIOD = 1.0
    REPLAN_DISTANCE = 0.25
    STANDOFF = 1.0
    GIMBAL_RATE = 40.0  # degrees / second

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
                        standoff=self.STANDOFF)

    def _observation(self):
        observation = self.markers.follow_observation()
        if (not observation or self.clock() - observation['captured_at'] > self.TARGET_AGE
                or self.clock() < observation['captured_at']):
            raise ControlError('目标位置已失效，请等待重新识别')
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
            observation = self._observation()
            nav = self.nav.snapshot()
            if not nav['ready']:
                raise ControlError(nav.get('error') or '等待导航服务与地图定位', 503)
            if nav['busy'] or nav['foreign_busy'] or nav['manual_active']:
                raise ControlError('请先停止其他导航任务和键盘驾驶')
            yaw, pitch = gaze_angles(observation['position'], nav['robot'], observation['config'])
            if abs(yaw) > 90 or abs(pitch) > 30:
                raise ControlError('猫的位置超出云台视角，请先调整车体或相机')
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
            self.nav.set_following(True)
            self.message = '自动追踪中'
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
                observation = self._observation()
                yaw, pitch = gaze_angles(observation['position'], nav['robot'], observation['config'])
                if abs(yaw) > 90 or abs(pitch) > 30:
                    raise ControlError('目标超出云台范围，已停止')
                dt = min(.1, max(0, self.clock() - self.last_tick))
                self.last_tick = self.clock()
                step = self.GIMBAL_RATE * dt
                desired = (yaw, pitch)
                self.commanded = tuple(old + max(-step, min(step, new-old))
                                       for old, new in zip(self.commanded, desired))
                self.gimbal.command(dict(action='target', token=self.gimbal_token,
                                         sequence=self.sequence, yaw=self.commanded[0], pitch=self.commanded[1]))
                self.sequence += 1
                goal, self.distance = approach_goal(observation['position'], nav['robot'], self.STANDOFF)
                # Stop approaching in the hold zone; resume with hysteresis.
                if self.distance <= self.STANDOFF + .1:
                    self.nav.cancel()
                    self.goal = None
                    self.message = '已接近目标，保持注视'
                    return
                self.message = '自动追踪中'
                if self.distance < self.STANDOFF + .3 and self.goal is None:
                    return
                if self.clock() - self.sent_at < self.REPLAN_PERIOD:
                    return
                moved = self.goal is None or math.hypot(goal[0]-self.goal[0], goal[1]-self.goal[1]) >= self.REPLAN_DISTANCE
                if nav['busy']:
                    if moved:
                        self.nav.cancel()
                    return
                if moved or nav['outcome'] == 'succeeded':
                    if self.nav.send_goal(goal):
                        self.goal, self.sent_at = goal, self.clock()
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
