#!/usr/bin/env python3
"""将双目相机目标点变换到 map，并在 RViz 显示。"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from tf2_geometry_msgs import do_transform_point
class TargetMarker(Node):
    def __init__(self):
        super().__init__('stereo_target_marker'); self.map_frame=self.declare_parameter('map_frame','map').value
        self.tf=Buffer(); TransformListener(self.tf,self); self.br=TransformBroadcaster(self)
        self.pub=self.create_publisher(Marker,'/target_marker',10)
        self.sub=self.create_subscription(PointStamped,'/cat_target_camera',self.callback,10)
        from std_msgs.msg import Float32MultiArray
        self.gimbal_sub=self.create_subscription(Float32MultiArray,'/gimbal/status',self.gimbal_status,10)
        self.yaw=self.pitch=0.0; self.create_timer(0.02,self.publish_camera_tf)
    def gimbal_status(self,msg):
        if len(msg.data)>=2: self.yaw,self.pitch=float(msg.data[0]),float(msg.data[1])
    def publish_camera_tf(self):
        import math
        t=TransformStamped(); t.header.stamp=self.get_clock().now().to_msg(); t.header.frame_id='base_link'; t.child_frame_id='camera_link'
        t.transform.translation.x=float(self.get_parameter_or('camera_x',0.20).value); t.transform.translation.z=float(self.get_parameter_or('camera_z',0.35).value)
        y=math.radians(self.yaw); p=math.radians(self.pitch); cy,sy=math.cos(y/2),math.sin(y/2); cp,sp=math.cos(p/2),math.sin(p/2)
        t.transform.rotation.w=cy*cp; t.transform.rotation.x=-sy*sp; t.transform.rotation.y=cy*sp; t.transform.rotation.z=sy*cp; self.br.sendTransform(t)
    def callback(self,msg):
        try: p=do_transform_point(msg,self.tf.lookup_transform(self.map_frame,msg.header.frame_id,rclpy.time.Time()))
        except TransformException as e: self.get_logger().warning(str(e)); return
        m=Marker(); m.header=p.header; m.header.frame_id=self.map_frame; m.ns='stereo_target'; m.id=0; m.type=Marker.SPHERE; m.action=Marker.ADD; m.pose.position=p.point; m.pose.orientation.w=1.0; m.scale.x=m.scale.y=m.scale.z=.18; m.color.r=1.; m.color.a=1.; m.lifetime.sec=1; self.pub.publish(m)
def main(args=None):
    rclpy.init(args=args); n=TargetMarker(); rclpy.spin(n)
if __name__=='__main__': main()
