#!/usr/bin/env python3
"""将双目相机目标点变换到 map，并在 RViz 显示。"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point
class TargetMarker(Node):
    def __init__(self):
        super().__init__('stereo_target_marker'); self.map_frame=self.declare_parameter('map_frame','map').value
        self.tf=Buffer(); TransformListener(self.tf,self); self.pub=self.create_publisher(Marker,'/target_marker',10)
        self.sub=self.create_subscription(PointStamped,'/cat_target_camera',self.callback,10)
    def callback(self,msg):
        try: p=do_transform_point(msg,self.tf.lookup_transform(self.map_frame,msg.header.frame_id,rclpy.time.Time()))
        except TransformException as e: self.get_logger().warning(str(e)); return
        m=Marker(); m.header=p.header; m.header.frame_id=self.map_frame; m.ns='stereo_target'; m.id=0; m.type=Marker.SPHERE; m.action=Marker.ADD; m.pose.position=p.point; m.pose.orientation.w=1.0; m.scale.x=m.scale.y=m.scale.z=.18; m.color.r=1.; m.color.a=1.; m.lifetime.sec=1; self.pub.publish(m)
def main(args=None):
    rclpy.init(args=args); n=TargetMarker(); rclpy.spin(n)
if __name__=='__main__': main()
