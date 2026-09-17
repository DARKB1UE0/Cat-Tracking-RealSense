#!/usr/bin/env python3
"""ROS 2 USB bridge: publish fresh commands at 50 Hz; no automatic command replay."""
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float32MultiArray
from tools.gimbal_usb import GimbalUsb


class GimbalUsbNode(Node):
    def __init__(self):
        super().__init__("gimbal_usb_bridge")
        port = self.declare_parameter("port", os.environ.get("GIMBAL_USB_PORT", "/dev/ttyACM0")).value
        baud = int(self.declare_parameter("baudrate", 115200).value)
        self.usb = GimbalUsb(port, baud)
        self.failed = False
        self.pub = self.create_publisher(Float32MultiArray, "/gimbal/status", 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE)
        self.sub = self.create_subscription(Float32MultiArray, "/gimbal/command", self.command, qos)
        self.timer = self.create_timer(0.01, self.poll)
        # Leave idle/STOP policy to firmware; opening the bridge sends no command.
        self.get_logger().info(f"云台 USB 已连接: {port}；指令建议 50 Hz，100 ms 无指令回零保持，显式 STOP 持续停机")

    def fail(self, exc):
        if not self.failed:
            self.get_logger().error(f"USB 通信失败，已停用接口；下位机超时回零保持，STOP 与电机故障优先: {exc}")
            self.failed = True

    def command(self, msg):
        if self.failed:
            return
        if (len(msg.data) not in (2, 3) or
                not all(math.isfinite(v) for v in msg.data) or
                (len(msg.data) == 3 and msg.data[2] not in (0.0, 1.0))):
            self.get_logger().warning("指令需要 [yaw_deg, pitch_deg, 0或1]，角度必须为有限值")
            return
        try:
            if len(msg.data) == 3 and msg.data[2] == 0:
                self.usb.stop()
            else:
                self.usb.send_setpoint(float(msg.data[0]), float(msg.data[1]))
        except (OSError, ValueError) as exc:
            self.fail(exc)

    def poll(self):
        if self.failed:
            return
        try:
            status = self.usb.read_status()
        except OSError as exc:
            self.fail(exc)
            return
        if status is not None:
            self.pub.publish(Float32MultiArray(data=[
                float(status[0]), float(status[1]), float(status[2]),
                float(status[3]), float(status[5])]))

    def close(self):
        # Silence selects firmware zero hold; only an explicit command latches STOP.
        self.usb.close()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GimbalUsbNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.close()
            finally:
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
