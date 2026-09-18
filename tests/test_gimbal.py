"""Protocol and Flask checks with a fake USB wire; never opens a device."""
import struct
import unittest

from flask import Flask
from gimbal_web import GimbalController, ControlError, create_gimbal_blueprint
from tools.gimbal_usb import GimbalUsb, FrameParser, frame, crc16, STATUS, STOP, SETPOINT


class Wire:
    def __init__(self):
        self.rx = bytearray()
        self.tx = []
        self.closed = False

    def read(self, size):
        result = bytes(self.rx[:size])
        del self.rx[:size]
        return result

    def write(self, data):
        self.tx.append(data)
        return len(data)

    def close(self):
        self.closed = True

    def feedback(self, yaw=12.5, pitch=-5, enabled=1, fault=0):
        self.rx.extend(frame(STATUS, 0, struct.pack('<ffBBHI', yaw, pitch, enabled, fault, 0, 20)))


class GimbalTest(unittest.TestCase):
    def setUp(self):
        self.now = 1.0
        self.wire = Wire()
        self.usb = GimbalUsb(serial_port=self.wire)
        self.controller = GimbalController(factory=lambda *a, **kw: self.usb, clock=lambda: self.now)
        self.wire.feedback()
        self.controller.tick()

    def acquire(self):
        return self.controller.command({'action': 'start'})['token']

    def target(self, token, yaw=30, pitch=60, sequence=0):
        return self.controller.command(dict(action='target', token=token, yaw=yaw, pitch=pitch, sequence=sequence))

    def test_idle_only_reads_then_enable_holds_measurement(self):
        self.assertEqual(self.wire.tx, [])
        self.acquire()
        self.controller.tick()
        kind, seq, payload = FrameParser().feed(self.wire.tx[-1])[0]
        self.assertEqual(kind, SETPOINT)
        self.assertEqual(struct.unpack('<ffBBH', payload), (12.5, -5, 1, 0, 0))

    def test_targets_hold_at_50hz_and_home(self):
        token = self.acquire()
        self.target(token, 90, -30)
        for _ in range(10):
            self.now += .02
            self.wire.feedback()
            self.controller.tick()
        self.assertEqual(len(self.wire.tx), 10)
        payload = FrameParser().feed(self.wire.tx[-1])[0][2]
        self.assertEqual(struct.unpack('<ffBBH', payload)[:2], (90, -30))
        self.target(token, 0, 0, 1)
        self.controller.tick()
        self.assertEqual(struct.unpack('<ffBBH', FrameParser().feed(self.wire.tx[-1])[0][2])[:2], (0, 0))

    def test_browser_lease_expiry_sends_stop_and_cannot_resume(self):
        token = self.acquire()
        self.now += .61
        self.wire.feedback()
        self.controller.tick()
        self.assertEqual(FrameParser().feed(self.wire.tx[-1])[0][0], STOP)
        with self.assertRaises(ControlError):
            self.target(token)
        self.controller.tick()
        self.assertEqual(len(self.wire.tx), 1)

    def test_expired_request_cannot_renew_before_worker_tick(self):
        token = self.acquire()
        self.now += .61
        self.wire.feedback()
        # Simulate fresh feedback, but a stalled worker has not expired the lease.
        self.controller.feedback_at = self.now
        with self.assertRaises(ControlError):
            self.target(token)
        self.assertIsNone(self.controller.token)
        self.assertEqual(FrameParser().feed(self.wire.tx[-1])[0][0], STOP)

    def test_stop_rejects_delayed_target_and_scoped_stop(self):
        old = self.acquire()
        self.controller.command({'action': 'stop', 'token': old})
        new = self.acquire()
        with self.assertRaises(ControlError):
            self.target(old)
        self.controller.command({'action': 'stop', 'token': old})
        self.assertEqual(self.controller.token, new)

    def test_exclusive_control_and_out_of_order_requests(self):
        token = self.acquire()
        with self.assertRaises(ControlError):
            self.acquire()
        self.target(token, pitch=20, sequence=4)
        with self.assertRaises(ControlError):
            self.target(token, yaw=80, pitch=20, sequence=3)
        self.assertEqual(self.controller.target, (30, 20))

    def test_invalid_angles_rejected_without_changing_target(self):
        token = self.acquire()
        for yaw, pitch in [(91, 0), (0, -91), (True, 0), ('10', 0), (None, 0), (float('nan'), 0), (0, float('inf'))]:
            with self.subTest(yaw=yaw, pitch=pitch), self.assertRaises(ValueError):
                self.target(token, yaw, pitch)
        self.assertEqual(self.controller.target, (12.5, -5))

    def test_stale_feedback_or_motor_fault_stops(self):
        self.acquire()
        self.now += .31
        self.controller.tick()
        self.assertFalse(self.controller.snapshot()['connected'])
        self.assertEqual(FrameParser().feed(self.wire.tx[-1])[0][0], STOP)
        self.wire.feedback(fault=8)
        self.controller.tick()
        with self.assertRaises(ControlError):
            self.acquire()
        self.wire.feedback(fault=1)  # Host timeout alone permits default zero hold.
        self.controller.tick()
        self.acquire()
        self.wire.feedback(fault=4)
        self.controller.tick()
        self.assertIsNone(self.controller.token)

    def test_usb_failure_invalidates_ownership(self):
        self.acquire()
        self.controller._disconnect(OSError('unplugged'))
        self.assertTrue(self.wire.closed)
        self.assertIsNone(self.controller.token)
        self.assertFalse(self.controller.snapshot()['connected'])

    def test_shutdown_stops_only_owned_motion(self):
        self.controller.close()
        self.assertEqual(self.wire.tx, [])
        self.controller.usb = self.usb
        self.controller.feedback_at = self.now
        self.acquire()
        self.controller.close()
        self.assertEqual(FrameParser().feed(self.wire.tx[-1])[0][0], STOP)

    def test_fragmentation_crc_noise_and_multiple_status_frames(self):
        packet = frame(STATUS, 5, struct.pack('<ffBBHI', 1, 2, 1, 0, 0, 40))
        broken = bytearray(packet)
        broken[-1] ^= 1
        self.wire.rx.extend(b'noise' + broken + packet[:8])
        self.assertIsNone(self.usb.read_status())
        self.wire.rx.extend(packet[8:])
        self.wire.feedback(45, -90)
        self.assertEqual(self.usb.read_status()[:2], (45, -90))
        self.assertIsNone(self.usb.read_status())
        self.assertEqual(crc16(b'123456789'), 0x4b37)

    def test_flask_validation_and_status(self):
        self.controller.start = lambda: None
        app = Flask(__name__)
        app.register_blueprint(create_gimbal_blueprint(self.controller))
        client = app.test_client()
        response = client.get('/api/gimbal/status')
        self.assertTrue(response.json['connected'])
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(client.post('/api/gimbal/control', json=[]).status_code, 400)
        self.assertEqual(client.post('/api/gimbal/control', json={'action': 'bad'}).status_code, 400)
        token = client.post('/api/gimbal/control', json={'action': 'start'}).json['token']
        self.assertEqual(client.post('/api/gimbal/control', json={'action': 'start'}).status_code, 409)
        self.assertEqual(client.post('/api/gimbal/control', json=dict(action='target', token=token, sequence=0, yaw=-90, pitch=30)).status_code, 200)
        self.assertEqual(client.post('/api/gimbal/control', json={'action': 'stop'}).status_code, 200)
        self.assertEqual(FrameParser().feed(self.wire.tx[-1])[0][0], STOP)


if __name__ == '__main__':
    unittest.main()
