"""USB CDC framing shared by the web controller and ROS bridge."""
import math
import struct

import serial

SOF = b"\xaa\x55"
VERSION = 1
SETPOINT, ENABLE, STOP, PING, STATUS = 1, 2, 3, 4, 0x81
YAW_LIMIT = 90.0
PITCH_LIMIT = 30.0


def validate_target(yaw, pitch):
    for value, limit in ((yaw, YAW_LIMIT), (pitch, PITCH_LIMIT)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("角度必须是数字")
        if not math.isfinite(value) or not -limit <= value <= limit:
            raise ValueError("角度超出范围：yaw ±90°，pitch ±30°")


def crc16(data: bytes) -> int:
    crc = 0xffff
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xa001) if crc & 1 else crc >> 1
    return crc


def frame(kind, seq, payload=b""):
    body = struct.pack("<BBBH", VERSION, kind, seq & 255, len(payload)) + payload
    return SOF + body + struct.pack("<H", crc16(body))


class GimbalUsb:
    def __init__(self, port=None, baudrate=115200, timeout=0.02, *, serial_port=None):
        self.ser = serial_port if serial_port is not None else serial.Serial(
            port, baudrate, timeout=timeout, write_timeout=0.1, exclusive=True)
        self.seq = 0
        self.parser = FrameParser()

    def _send(self, kind, payload=b""):
        packet = frame(kind, self.seq, payload)
        if self.ser.write(packet) != len(packet):
            raise serial.SerialTimeoutException("Incomplete gimbal command")
        self.seq = (self.seq + 1) & 255

    def send_setpoint(self, yaw_deg, pitch_deg, enable=True):
        validate_target(yaw_deg, pitch_deg)
        self._send(SETPOINT, struct.pack("<ffBBH", yaw_deg, pitch_deg,
                                        int(bool(enable)), 0, 0))

    def enable(self):
        self._send(ENABLE)

    def stop(self):
        self._send(STOP)

    def ping(self):
        self._send(PING)

    def close(self):
        self.ser.close()

    def read_status(self):
        latest = None
        for kind, sequence, payload in self.parser.feed(self.ser.read(4096)):
            if kind == STATUS and len(payload) == 16:
                status = struct.unpack("<ffBBHI", payload)
                if (all(math.isfinite(v) for v in status[:2])
                        and status[2] in (0, 1) and status[4] == 0):
                    latest = status
        return latest


class FrameParser:
    """Recover complete frames from a noisy, fragmented USB byte stream."""
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        self.buffer.extend(data)
        packets = []
        while len(self.buffer) >= 2:
            if self.buffer[:2] != SOF:
                del self.buffer[0]
                continue
            if len(self.buffer) < 7:
                break
            version, kind, sequence, size = struct.unpack_from("<BBBH", self.buffer, 2)
            if version != VERSION or size > 32:
                del self.buffer[0]
                continue
            total = size + 9
            if len(self.buffer) < total:
                break
            expected = struct.unpack_from("<H", self.buffer, 7 + size)[0]
            if crc16(self.buffer[2:7 + size]) != expected:
                del self.buffer[0]
                continue
            packets.append((kind, sequence, bytes(self.buffer[7:7 + size])))
            del self.buffer[:total]
        return packets
