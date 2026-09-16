"""Protocol and bridge callback regressions; no ROS/USB hardware required.

Run: python -m unittest discover -s tests -p test_gimbal_usb.py -v
"""
import importlib.util
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.gimbal_usb import (
    GimbalUsb, FrameParser, frame, crc16, STATUS, STOP, SETPOINT,
    STATUS_FORMAT, SETPOINT_FORMAT,
)


class SerialStub:
    def __init__(self):
        self.rx = bytearray()
        self.tx = []
        self.partial_write = False
        self.closed = False

    @property
    def in_waiting(self):
        return len(self.rx)

    def read(self, size):
        data = bytes(self.rx[:size])
        del self.rx[:size]
        return data

    def write(self, data):
        self.tx.append(data)
        return len(data) - int(self.partial_write)

    def close(self):
        self.closed = True


def status_packet(seq=0, yaw=12.5, pitch=-3.0, enabled=1, reserved=0):
    return frame(STATUS, seq, STATUS_FORMAT.pack(yaw, pitch, enabled, 0, reserved, 20))


class UsbProtocolTests(unittest.TestCase):
    def test_crc_known_vector(self):
        self.assertEqual(crc16(b"123456789"), 0x4b37)

    def test_every_packet_split_and_coalescing(self):
        packet = status_packet()
        expected = [(STATUS, 0, packet[7:-2])]
        for split in range(len(packet) + 1):
            parser = FrameParser()
            self.assertEqual(parser.feed(packet[:split]) + parser.feed(packet[split:]), expected)
        parser = FrameParser()
        self.assertEqual(parser.feed(packet + packet), expected * 2)

    def test_noise_crc_version_and_oversized_recovery(self):
        packet = status_packet()
        broken = bytearray(packet)
        broken[-1] ^= 1
        noise = b"x" * 9000 + bytes.fromhex("aa55010100ffffaa550201000000")
        parser = FrameParser()
        self.assertEqual(parser.feed(noise + broken + packet), [(STATUS, 0, packet[7:-2])])
        self.assertLess(len(parser.buffer), 41)
        self.assertEqual(parser.feed(b"noise\xaa"), [])
        self.assertEqual(parser.feed(packet[1:]), [(STATUS, 0, packet[7:-2])])

    def test_persistent_nonblocking_reads_and_newest_status(self):
        serial = SerialStub()
        usb = GimbalUsb(serial_port=serial)
        packet = status_packet()
        serial.rx.extend(packet[:8])
        self.assertIsNone(usb.read_status())
        self.assertIsNone(usb.read_status())
        serial.rx.extend(packet[8:] + status_packet(1, yaw=20))
        self.assertEqual(usb.read_status(), (20, -3, 1, 0, 0, 20))

    def test_invalid_status_fields_are_ignored(self):
        serial = SerialStub()
        usb = GimbalUsb(serial_port=serial)
        serial.rx.extend(status_packet(yaw=float("nan")) + status_packet(enabled=2) +
                         status_packet(reserved=1) + frame(STATUS, 0, b"short") +
                         frame(STOP, 0) + status_packet())
        self.assertEqual(len(usb.read_statuses()), 1)

    def test_commands_and_sequence_wrap(self):
        serial = SerialStub()
        usb = GimbalUsb(serial_port=serial)
        usb.seq = 255
        usb.send_setpoint(5, -2)
        usb.stop()
        packets = FrameParser().feed(b"".join(serial.tx))
        self.assertEqual(packets, [(SETPOINT, 255, SETPOINT_FORMAT.pack(5, -2, 1, 0, 0)),
                                   (STOP, 0, b"")])

    def test_partial_write_and_nonfinite_setpoint_fail(self):
        serial = SerialStub()
        usb = GimbalUsb(serial_port=serial)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                usb.send_setpoint(value, 0)
        self.assertEqual(serial.tx, [])
        serial.partial_write = True
        with self.assertRaises(OSError):
            usb.stop()
        self.assertEqual(usb.seq, 0)


def load_bridge_with_ros_stubs():
    # Exercise actual bridge callbacks. This does not exercise the ROS executor/DDS.
    modules = {name: ModuleType(name) for name in
               ("rclpy", "rclpy.node", "rclpy.qos", "std_msgs", "std_msgs.msg")}
    modules["rclpy.node"].Node = object
    modules["rclpy.qos"].QoSProfile = Mock()
    modules["rclpy.qos"].DurabilityPolicy = SimpleNamespace(VOLATILE=0)
    modules["std_msgs.msg"].Float32MultiArray = lambda **kw: SimpleNamespace(**kw)
    spec = importlib.util.spec_from_file_location("bridge_under_test", ROOT / "ros2_usb_gimbal.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class BridgeCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_bridge_with_ros_stubs()

    def setUp(self):
        self.node = self.module.GimbalUsbNode.__new__(self.module.GimbalUsbNode)
        self.node.usb = Mock()
        self.node.pub = Mock()
        self.node.get_logger = Mock(return_value=Mock())
        self.node.failed = False

    def test_command_and_stop(self):
        self.node.command(SimpleNamespace(data=[5, -2, 1]))
        self.node.usb.send_setpoint.assert_called_once_with(5.0, -2.0)
        self.node.command(SimpleNamespace(data=[0, 0, 0]))
        self.node.usb.stop.assert_called_once_with()

    def test_invalid_commands_do_not_transmit(self):
        for data in ([], [1], [1, 2, 3], [1, 2, 1, 0], [float("nan"), 0]):
            self.node.command(SimpleNamespace(data=data))
        self.assertEqual(self.node.usb.mock_calls, [])

    def test_poll_reports_actual_status_without_replaying_commands(self):
        self.node.usb.read_status.return_value = (2, -1, 0, 6, 0, 120)
        self.node.poll()
        self.assertEqual(self.node.pub.publish.call_args.args[0].data, [2, -1, 0, 6, 120])
        self.node.usb.send_setpoint.assert_not_called()

    def test_io_failure_latches_until_restart(self):
        self.node.usb.send_setpoint.side_effect = OSError("disconnected")
        self.node.command(SimpleNamespace(data=[1, 2]))
        self.assertTrue(self.node.failed)
        self.node.command(SimpleNamespace(data=[3, 4]))
        self.node.poll()
        self.node.usb.send_setpoint.assert_called_once()
        self.node.usb.read_status.assert_not_called()

    def test_close_releases_port_even_if_stop_fails(self):
        self.node.usb.stop.side_effect = OSError("disconnected")
        self.node.close()
        self.node.usb.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
