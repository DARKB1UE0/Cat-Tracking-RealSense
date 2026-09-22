"""Actual ROS publishers, TF and marker cleanup, in an isolated ROS domain."""
import math
import os
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

try:
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from tf2_ros import StaticTransformBroadcaster
    from geometry_msgs.msg import TransformStamped, PointStamped
    from visualization_msgs.msg import MarkerArray, Marker
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from cat_markers import CatMarkers


@unittest.skipUnless(ROS_AVAILABLE, 'source /opt/ros/humble/setup.bash first')
class MarkerRosTests(unittest.TestCase):
    def test_map_position_lifetime_missing_tf_and_clear(self):
        # Both nodes use a private domain; nothing is published to the robot.
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'ROS_DOMAIN_ID': '91', 'CAT_CAMERA_CONFIG': str(Path(temp)/'config.json')}):
            Path(temp, 'config.json').write_text((Path(__file__).resolve().parents[1]/'camera_mount.json').read_text())
            context = Context()
            rclpy.init(context=context)
            node = rclpy.create_node('cat_marker_test', context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            received, positions = [], []
            node.create_subscription(MarkerArray, '/cat/markers', received.append, 10)
            node.create_subscription(PointStamped, '/cat/position', positions.append, 10)
            service = CatMarkers()
            service.start()
            def wait_for(predicate, timeout=4):
                end = time.monotonic()+timeout
                while time.monotonic()<end:
                    executor.spin_once(timeout_sec=.02)
                    if predicate(): return
                self.fail('Timeout: '+repr(service.snapshot()))
            try:
                wait_for(lambda: service.ready)
                wait_for(lambda: node.count_publishers('/cat/markers') > 0)
                service.observe([0, 0, 2], service.capture(dict(connected=True, fault=1, yaw=90, pitch=0)))
                wait_for(lambda: 'TF' in service.snapshot()['message'])
                self.assertFalse(positions)
                broadcaster = StaticTransformBroadcaster(node)
                t = TransformStamped()
                t.header.stamp = node.get_clock().now().to_msg()
                t.header.frame_id = 'map'; t.child_frame_id = 'base_link'
                t.transform.translation.x, t.transform.translation.y = 10., 20.
                t.transform.rotation.z = math.sin(math.pi/4)
                t.transform.rotation.w = math.cos(math.pi/4)
                broadcaster.sendTransform(t)
                wait_for(lambda: service.buffer.can_transform('map','base_link',rclpy.time.Time()))
                capture = service.capture(dict(connected=True, fault=1, yaw=90, pitch=0))
                service.observe([0, 0, 2], capture)
                wait_for(lambda: any(m.action == Marker.ADD for a in received for m in a.markers))
                wait_for(lambda: len(positions)>0)
                point = positions[-1]
                self.assertEqual(point.header.frame_id, 'map')
                self.assertEqual(point.header.stamp, capture[0].to_msg())
                self.assertAlmostEqual(point.point.x, 8.)
                self.assertAlmostEqual(point.point.y, 20.)
                self.assertAlmostEqual(point.point.z, .15)
                added = [m for a in received for m in a.markers if m.action == Marker.ADD]
                self.assertEqual({m.type for m in added}, {Marker.SPHERE, Marker.TEXT_VIEW_FACING})
                self.assertTrue(all(m.lifetime.sec < 2 and not m.frame_locked for m in added))
                self.assertTrue(all(0 <= m.lifetime.nanosec < 1_000_000_000 for m in added))
                received.clear()
                service.clear('追踪已停止')
                wait_for(lambda: any(m.action == Marker.DELETE for a in received for m in a.markers))
                service.observe([0,0,2], capture)
                self.assertIsNone(service.pending)
                # Valid new target must expire without fresh observations.
                service.observe([0,0,2], service.capture(dict(connected=True, fault=1, yaw=90, pitch=0)))
                wait_for(lambda: service.snapshot()['position'] is not None)
                wait_for(lambda: service.snapshot()['message'] == '目标位置已过期', timeout=3)
                old_capture = service.capture(dict(connected=True, fault=1, yaw=0, pitch=0))
                service.set_target_kind('person')
                service.observe([0, 0, 2], old_capture)
                self.assertIsNone(service.pending)
                received.clear()
                service.observe([0, 0, 2], service.capture(dict(connected=True, fault=1, yaw=0, pitch=0)))
                wait_for(lambda: any(m.action == Marker.ADD and m.text.startswith('Target person')
                                     for a in received for m in a.markers))
                self.assertIn('人物', service.snapshot()['message'])
            finally:
                service.close(); executor.shutdown(); node.destroy_node(); context.shutdown()


if __name__ == '__main__': unittest.main()
