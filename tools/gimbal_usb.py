"""USB CDC protocol for the BasicFramework_F4 gimbal controller."""
import struct, time, serial

SOF = b"\xaa\x55"
VERSION = 1
SETPOINT, ENABLE, STOP, PING, STATUS = 1, 2, 3, 4, 0x81

def crc16(data: bytes) -> int:
    crc = 0xffff
    for b in data:
        crc ^= b
        for _ in range(8): crc = ((crc >> 1) ^ 0xa001) if crc & 1 else crc >> 1
    return crc

def frame(kind, seq, payload=b""):
    body = struct.pack("<BBB H", VERSION, kind, seq & 255, len(payload)) + payload
    return SOF + body + struct.pack("<H", crc16(body))

class GimbalUsb:
    def __init__(self, port, baudrate=115200, timeout=0.02):
        self.ser, self.seq = serial.Serial(port, baudrate, timeout=timeout), 0
    def send_setpoint(self, yaw_deg, pitch_deg, enable=True):
        payload = struct.pack("<ffBBH", yaw_deg, pitch_deg, 1 if enable else 0, 0, 0)
        self.ser.write(frame(SETPOINT, self.seq, payload)); self.seq += 1
    def enable(self): self.ser.write(frame(ENABLE, self.seq)); self.seq += 1
    def stop(self): self.ser.write(frame(STOP, self.seq)); self.seq += 1
    def ping(self): self.ser.write(frame(PING, self.seq)); self.seq += 1
    def read_status(self):
        data = self.ser.read(64)
        i = data.find(SOF)
        if i < 0 or len(data) < i + 9: return None
        n = struct.unpack_from("<H", data, i + 5)[0]
        if len(data) < i + 9 + n: return None
        body = data[i+2:i+7+n]
        if crc16(body) != struct.unpack_from("<H", data, i+7+n)[0] or body[1] != STATUS: return None
        if n == 16: return struct.unpack_from("<ffBBHI", data, i+7)
        return None
