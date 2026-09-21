"""Exercise real Nav2 action cancellation and command arbitration in domain 92."""
import os
import threading
import time
import unittest
from unittest.mock import patch

from follow_navigation import FollowNavigation
try:
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.action import ActionServer, GoalResponse, CancelResponse
    from nav2_msgs.action import NavigateToPose
    from nav2_msgs.msg import SpeedLimit
    from geometry_msgs.msg import Twist, TransformStamped
    from tf2_ros import TransformBroadcaster
    from action_msgs.msg import GoalStatus, GoalStatusArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


@unittest.skipUnless(ROS_AVAILABLE, 'ROS Humble required')
class NavigationRosTests(unittest.TestCase):
    def test_cancel_late_accept_manual_arbitration_and_speed_limit(self):
        with patch.dict(os.environ, {'ROS_DOMAIN_ID': '92'}):
            context = Context(); rclpy.init(context=context)
            node = rclpy.create_node('mock_cat_nav2', context=context)
            executor = MultiThreadedExecutor(num_threads=4, context=context)
            executor.add_node(node)
            group = ReentrantCallbackGroup()
            velocities, limits, goals = [], [], []
            delay_accept = threading.Event(); request_received = threading.Event()
            reject = threading.Event()
            delay_accept.set()
            def accept(goal):
                goals.append(goal); request_received.set()
                delay_accept.wait(timeout=3)
                return GoalResponse.REJECT if reject.is_set() else GoalResponse.ACCEPT
            def execute(handle):
                end = time.monotonic()+5
                while time.monotonic()<end:
                    if handle.is_cancel_requested:
                        # Hold busy after ACK to test waiting for the result.
                        time.sleep(.2); handle.canceled(); return NavigateToPose.Result()
                    time.sleep(.01)
                handle.abort(); return NavigateToPose.Result()
            server = ActionServer(node, NavigateToPose, '/navigate_to_pose', execute,
                                  goal_callback=accept, cancel_callback=lambda h: CancelResponse.ACCEPT,
                                  callback_group=group)
            node.create_subscription(Twist, '/cmd_vel', velocities.append, 10)
            node.create_subscription(SpeedLimit, '/speed_limit', limits.append, 10)
            manual = node.create_publisher(Twist, '/cmd_vel_manual', 10)
            broadcaster = TransformBroadcaster(node)
            def tf():
                t = TransformStamped(); t.header.stamp = node.get_clock().now().to_msg()
                t.header.frame_id = 'map'; t.child_frame_id = 'base_link'; t.transform.rotation.w = 1.
                broadcaster.sendTransform(t)
            timer = node.create_timer(.03, tf, callback_group=group)
            worker = threading.Thread(target=executor.spin, daemon=True); worker.start()
            nav = FollowNavigation(); nav.start()
            def wait(predicate, timeout=4):
                end = time.monotonic()+timeout
                while time.monotonic()<end:
                    if predicate(): return
                    time.sleep(.01)
                self.fail('Timeout '+str(nav.snapshot()))
            def nonzero_count(): return sum(abs(m.linear.x)>.001 for m in velocities)
            command = Twist(); command.linear.x = .1
            try:
                wait(lambda: nav.snapshot()['ready'] and node.count_subscribers('/cmd_vel_manual')>0)
                manual.publish(command); wait(lambda: nonzero_count()>0)
                wait(lambda: velocities and velocities[-1].linear.x==0.)  # manual watchdog
                time.sleep(.3)
                nav.set_following(True)
                wait(lambda: limits and limits[-1].speed_limit == .2)
                time.sleep(.16)
                self.assertTrue(nav.send_goal((2., 0., 0.)))
                wait(lambda: nav.goal_handle is not None)
                self.assertFalse(nav.snapshot()['foreign_busy'])
                self.assertEqual(goals[-1].pose.header.frame_id, 'map')
                before = nonzero_count(); manual.publish(command); time.sleep(.1)
                self.assertEqual(nonzero_count(), before)
                self.assertFalse(nav.send_goal((3., 0., 0.)))
                nav.set_following(False); nav.cancel()
                self.assertTrue(nav.snapshot()['busy'])
                self.assertFalse(nav.send_goal((3., 0., 0.)))
                wait(lambda: not nav.snapshot()['busy'])
                self.assertEqual(nav.snapshot()['outcome'], 'canceled')
                wait(lambda: limits[-1].speed_limit == 0.)
                # Stop before a delayed goal response: accepted goal must still cancel.
                delay_accept.clear(); request_received.clear()
                nav.set_following(True); time.sleep(.16)
                self.assertTrue(nav.send_goal((3., 0., 0.)))
                wait(request_received.is_set)
                nav.set_following(False); nav.cancel()
                self.assertTrue(nav.snapshot()['busy'])
                delay_accept.set()
                wait(lambda: not nav.snapshot()['busy'])
                self.assertEqual(nav.snapshot()['outcome'], 'canceled')
                wait(lambda: not nav.snapshot()['foreign_busy'])
                # A late rejection after stop must release cancel state and speed cap.
                delay_accept.clear(); request_received.clear(); reject.set()
                nav.set_following(True); time.sleep(.16)
                self.assertTrue(nav.send_goal((3., 0., 0.)))
                wait(request_received.is_set)
                nav.set_following(False); nav.cancel(); delay_accept.set()
                wait(lambda: not nav.snapshot()['busy'])
                self.assertFalse(nav.cancel_requested)
                self.assertEqual(nav.snapshot()['outcome'], 'failed')
                wait(lambda: limits[-1].speed_limit == 0.)
                reject.clear()
                # Foreign RViz task blocks manual traffic as well as automatic goals.
                status = GoalStatusArray(); entry = GoalStatus(); entry.status = GoalStatus.STATUS_EXECUTING
                entry.goal_info.goal_id.uuid = [7]*16; status.status_list = [entry]
                nav._status(status, 'other')
                before = nonzero_count(); manual.publish(command); time.sleep(.1)
                self.assertEqual(nonzero_count(), before)
                nav.set_following(True); time.sleep(.16)
                self.assertFalse(nav.send_goal((4., 0., 0.)))
                nav.set_following(False); nav._status(GoalStatusArray(), 'other')
                manual.publish(command); wait(lambda: nonzero_count()>before)
            finally:
                delay_accept.set(); nav.close(); timer.cancel()
                executor.shutdown(); worker.join(timeout=2)
                server.destroy(); node.destroy_node(); context.shutdown()


if __name__ == '__main__': unittest.main()
