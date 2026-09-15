"""车体(base_link)与云台坐标变换。约定 x 前、y 左、z 上，角度单位为度。"""
import math

def wrap_deg(a): return (a + 180.0) % 360.0 - 180.0

class GimbalGeometry:
    def __init__(self, yaw0_deg=0.0, pitch0_deg=0.0, yaw_sign=1.0, pitch_sign=1.0):
        self.yaw0, self.pitch0 = yaw0_deg, pitch0_deg
        self.yaw_sign, self.pitch_sign = yaw_sign, pitch_sign
    def body_vector_to_gimbal(self, x, y, z):
        yaw = math.degrees(math.atan2(y, x))
        pitch = math.degrees(math.atan2(z, math.hypot(x, y)))
        return (wrap_deg(self.yaw0 + self.yaw_sign*yaw),
                self.pitch0 + self.pitch_sign*pitch)
    def gimbal_to_body_vector(self, yaw_deg, pitch_deg, distance=1.0):
        yaw = math.radians(self.yaw_sign*(yaw_deg-self.yaw0))
        pitch = math.radians(self.pitch_sign*(pitch_deg-self.pitch0))
        cp = math.cos(pitch)
        return (distance*cp*math.cos(yaw), distance*cp*math.sin(yaw), distance*math.sin(pitch))
    def world_target_to_gimbal(self, target_world, base_world, base_yaw_deg):
        dx, dy, dz = [target_world[i]-base_world[i] for i in range(3)]
        a = math.radians(base_yaw_deg)
        return self.body_vector_to_gimbal(math.cos(a)*dx+math.sin(a)*dy,
            -math.sin(a)*dx+math.cos(a)*dy, dz)
