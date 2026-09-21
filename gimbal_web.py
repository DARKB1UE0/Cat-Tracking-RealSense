"""Single USB owner for web gimbal control; no camera or ROS dependency."""
import atexit
import os
import secrets
import threading
import time

from flask import Blueprint, jsonify, request
from tools.gimbal_usb import GimbalUsb, validate_target


class ControlError(Exception):
    def __init__(self, message, code=409):
        super().__init__(message)
        self.code = code


class GimbalController:
    PERIOD = 0.02  # Firmware target watchdog is 100 ms.
    LEASE = 0.6
    STATUS_TIMEOUT = 0.3

    def __init__(self, port=None, factory=GimbalUsb, clock=time.monotonic):
        # Never guess ttyACM0: it can be the chassis or another peripheral.
        self.port = port or os.environ.get(
            'GIMBAL_USB_PORT',
            '/dev/serial/by-id/usb-YueLuEmbedded_Vision_Comm_port_2070377C5948-if00')
        self.factory, self.clock = factory, clock
        self.lock = threading.RLock()
        self.shutdown = threading.Event()
        self.thread = None
        self.usb = None
        self.feedback = None
        self.feedback_at = None
        self.token = None
        self.owner = None
        self.target = None
        self.deadline = 0
        self.sequence = -1
        self.message = '等待云台连接'
        self.next_connect = 0

    def start(self):
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True,
                                               name='gimbal-usb')
                self.thread.start()

    def _ready(self):
        return (self.usb is not None and self.feedback_at is not None
                and self.clock() - self.feedback_at <= self.STATUS_TIMEOUT)

    def snapshot(self):
        with self.lock:
            ready = self._ready()
            return dict(connected=ready, active=self.token is not None,
                        owner=self.owner,
                        controllable=bool(ready and not self.feedback[3] & ~1),
                        yaw=self.feedback[0] if ready else None,
                        pitch=self.feedback[1] if ready else None,
                        enabled=bool(self.feedback[2]) if ready else False,
                        fault=self.feedback[3] if ready else None,
                        rx_age_ms=self.feedback[5] if ready else None,
                        message=self.message, port=self.port)

    def _drop(self, message):
        self.token = self.target = None
        self.owner = None
        self.message = message

    def _disconnect(self, error):
        self._drop('USB 连接失败，请检查云台串口配置、占用及接线')
        if self.usb is not None:
            try:
                self.usb.close()
            except OSError:
                pass
        self.usb = None
        self.feedback_at = None
        self.next_connect = self.clock() + 2

    def _stop(self, message):
        self._drop(message)  # Invalidate ownership before attempting the write.
        if self.usb is not None:
            try:
                self.usb.stop()
            except OSError as exc:
                self._disconnect(exc)
                raise ControlError('停止指令未送达：USB 已断开', 503) from exc

    def command(self, data, owner='manual'):
        if not isinstance(data, dict):
            raise ValueError('请求必须为 JSON 对象')
        action = data.get('action')
        with self.lock:
            if action == 'stop':
                # A session-specific delayed stop must not kill a new session.
                token = data.get('token')
                if token is None or token == self.token:
                    if self.usb is None or self.feedback is None:
                        self._drop('云台未连接，无法确认停止')
                        raise ControlError(self.message, 503)
                    self._stop('已停止电机输出')
                return {}
            if action not in ('start', 'target'):
                raise ValueError('未知云台操作')
            if not self._ready():
                raise ControlError('未收到云台状态，请检查 USB 连接', 503)
            if self.feedback[3] & ~1:
                raise ControlError('云台报告电机故障，请检查状态')
            if action == 'start':
                if self.token is not None:
                    raise ControlError('云台已由另一控制会话占用')
                # Enabling holds the measured position instead of jumping to 0.
                validate_target(*self.feedback[:2])
                self.target = self.feedback[:2]
                self.token = secrets.token_urlsafe(24)
                self.owner = owner
                self.sequence = -1
            else:
                if self.token is None or data.get('token') != self.token:
                    raise ControlError('控制已结束，请重新启用')
                if self.clock() >= self.deadline:
                    self._stop('网页通信超时，已停止')
                    raise ControlError('控制已超时，请重新启用')
                yaw, pitch = data.get('yaw'), data.get('pitch')
                validate_target(yaw, pitch)
                seq = data.get('sequence')
                if type(seq) is not int or seq <= self.sequence:
                    raise ControlError('已忽略过期的角度指令')
                self.target = (yaw, pitch)
                self.sequence = seq
            self.deadline = self.clock() + self.LEASE
            self.message = '自动追踪瞄准中' if self.owner == 'auto' else '手动控制中'
            return dict(token=self.token, yaw=self.target[0], pitch=self.target[1])

    def tick(self):
        """One serial-worker iteration. Also used with a fake clock in tests."""
        with self.lock:
            if self.usb is None:
                if self.clock() < self.next_connect:
                    return
                self.usb = self.factory(self.port, timeout=0)
                self.feedback = self.feedback_at = None
                self.message = '串口已打开，等待云台反馈'
            status = self.usb.read_status()
            if status is not None:
                self.feedback, self.feedback_at = status, self.clock()
                if self.token is None:
                    self.message = '云台已连接'
            if self.token is not None:
                if self.clock() >= self.deadline:
                    self._stop('网页通信超时，已停止')
                elif not self._ready() or self.feedback[3] & ~1:
                    self._stop('反馈中断或电机故障，已停止')
                else:
                    self.usb.send_setpoint(*self.target)

    def _run(self):
        while not self.shutdown.is_set():
            started = self.clock()
            try:
                self.tick()
            except (OSError, ControlError) as exc:
                with self.lock:
                    self._disconnect(exc)
            self.shutdown.wait(max(0, self.PERIOD - (self.clock() - started)))

    def close(self):
        self.shutdown.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        with self.lock:
            if self.token is not None:
                try:
                    self._stop('控制服务已关闭')
                except ControlError:
                    pass
            if self.usb is not None:
                self.usb.close()
                self.usb = None


def create_gimbal_blueprint(controller):
    blueprint = Blueprint('gimbal', __name__)

    @blueprint.get('/api/gimbal/status')
    def status():
        controller.start()
        response = jsonify(controller.snapshot())
        response.headers['Cache-Control'] = 'no-store'
        return response

    @blueprint.post('/api/gimbal/control')
    def control():
        controller.start()
        try:
            result = controller.command(request.get_json(silent=True))
            return jsonify(ok=True, **result)
        except ValueError as exc:
            return jsonify(ok=False, error=str(exc)), 400
        except ControlError as exc:
            return jsonify(ok=False, error=str(exc)), exc.code

    return blueprint


def install_gimbal(app):
    controller = GimbalController()
    app.register_blueprint(create_gimbal_blueprint(controller))
    atexit.register(controller.close)
    return controller
