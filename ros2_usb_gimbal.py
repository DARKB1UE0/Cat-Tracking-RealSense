#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from tools.gimbal_usb import GimbalUsb

class GimbalUsbNode(Node):
    def __init__(self):
        super().__init__('gimbal_usb_bridge')
        port = self.declare_parameter('port', os.environ.get('GIMBAL_USB_PORT', '/dev/ttyACM0')).value
        baud = int(self.declare_parameter('baudrate', 115200).value)
        self.usb = GimbalUsb(port, baud)
        self.pub = self.create_publisher(Float32MultiArray, '/gimbal/status', 10)
        self.sub = self.create_subscription(Float32MultiArray, '/gimbal/command', self.command, 10)
        self.timer = self.create_timer(0.02, self.poll)
        self.get_logger().info(f'云台 USB 已连接: {port}')
    def command(self, msg):
        if len(msg.data) < 2: return
        self.usb.send_setpoint(float(msg.data[0]), float(msg.data[1]), bool(msg.data[2]) if len(msg.data)>2 else True)
    def poll(self):
        s = self.usb.read_status()
        if s: self.pub.publish(Float32MultiArray(data=[float(s[0]),float(s[1]),float(s[2]),float(s[3]),float(s[5])]))
def main(args=None):
    rclpy.init(args=args); node = GimbalUsbNode()
    try: rclpy.spin(node)
    finally: node.usb.stop(); node.destroy_node(); rclpy.shutdown()
if __name__ == '__main__': main()
