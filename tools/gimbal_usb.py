"""USB CDC gimbal protocol. Angles: degrees relative to motor encoder zero."""
import math
import struct

import serial

SOF = b"\xaa\x55"
VERSION = 1
MAX_PAYLOAD = 32
SETPOINT, ENABLE, STOP, PING, STATUS = 1, 2, 3, 4, 0x81
STATUS_FORMAT = struct.Struct("<ffBBHI")
SETPOINT_FORMAT = struct.Struct("<ffBBH")
FAULT_NAMES = {
    # host_timeout is informational: firmware may be actively holding zero.
    1: "host_timeout", 2: "yaw_offline", 4: "pitch_offline",
    8: "dm_error", 16: "angle_limit", 32: "not_referenced",
}


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
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xa001) if crc & 1 else crc >> 1
    return crc


def frame(kind, seq, payload=b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload exceeds 32 bytes")
    body = struct.pack("<BBBH", VERSION, kind, seq & 255, len(payload)) + payload
    return SOF + body + struct.pack("<H", crc16(body))


class FrameParser:
    """Retain partial frames and recover from noise, oversized lengths and CRC errors."""
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        self.buffer.extend(data)
        frames = []
        while len(self.buffer) >= 2:
            start = self.buffer.find(SOF)
            if start < 0:
                self.buffer[:] = self.buffer[-1:] if self.buffer[-1] == SOF[0] else b""
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < 7:
                break
            version, kind, seq, size = struct.unpack_from("<BBBH", self.buffer, 2)
            if version != VERSION or size > MAX_PAYLOAD:
                del self.buffer[0]
                continue
            total = 9 + size
            if len(self.buffer) < total:
                break
            expected = struct.unpack_from("<H", self.buffer, 7 + size)[0]
            if crc16(self.buffer[2:7 + size]) != expected:
                del self.buffer[0]
                continue
            frames.append((kind, seq, bytes(self.buffer[7:7 + size])))
            del self.buffer[:total]
        return frames


class GimbalUsb:
    def __init__(self, port=None, baudrate=115200, timeout=0.0, serial_port=None):
        if serial_port is None:
            serial_port = serial.Serial(port, baudrate, timeout=timeout, write_timeout=0.05, exclusive=True)
        self.ser = serial_port
        self.seq = 0
        self.parser = FrameParser()

    def _send(self, kind, payload=b""):
        packet = frame(kind, self.seq, payload)
        if self.ser.write(packet) != len(packet):
            raise serial.SerialTimeoutException("Incomplete gimbal command")
        self.seq = (self.seq + 1) & 255

    def send_setpoint(self, yaw_deg, pitch_deg, enable=True):
        validate_target(yaw_deg, pitch_deg)
        self._send(SETPOINT, SETPOINT_FORMAT.pack(yaw_deg, pitch_deg, int(bool(enable)), 0, 0))

    def enable(self):
        """Requires a valid target received within 100 ms; cannot revive an expired target."""
        self._send(ENABLE)

    def stop(self):
        """Latch motor stop, including during host silence; send a new target to resume."""
        self._send(STOP)

    def ping(self):
        """Request real status; does not refresh the target or change zero hold/STOP."""
        self._send(PING)

    def read_statuses(self):
        data = self.ser.read(min(self.ser.in_waiting, 4096))
        result = []
        for kind, _seq, payload in self.parser.feed(data):
            if kind == STATUS and len(payload) == STATUS_FORMAT.size:
                status = STATUS_FORMAT.unpack(payload)
                if (all(math.isfinite(v) for v in status[:2]) and
                        status[2] in (0, 1) and status[4] == 0):
                    result.append(status)
        return result

    def read_status(self):
        statuses = self.read_statuses()
        return statuses[-1] if statuses else None

    def close(self):
        self.ser.close()
