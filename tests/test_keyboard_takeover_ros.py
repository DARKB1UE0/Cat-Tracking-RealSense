"""Keyboard handoff from external Nav2 tasks, isolated from robot in domain 93."""
import os
import threading
import time
import unittest
from unittest.mock import patch
from follow_navigation import FollowNavigation
from test_follow_navigation_ros import ROS_AVAILABLE


@unittest.skipUnless(ROS_AVAILABLE, 'ROS Humble required')
class KeyboardTakeoverTests(unittest.TestCase):
    def test_external_goals_cancel_then_manual_works_without_map_tf(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.action import ActionClient, ActionServer, CancelResponse
        from nav2_msgs.action import NavigateToPose, NavigateThroughPoses, FollowWaypoints
        from geometry_msgs.msg import Twist
        from action_msgs.msg import GoalStatus
        with patch.dict(os.environ, {'ROS_DOMAIN_ID': '93'}):
            context = Context(); rclpy.init(context=context)
            node = rclpy.create_node('external_nav_tasks', context=context)
            executor = MultiThreadedExecutor(num_threads=6, context=context)
            executor.add_node(node)
            group = ReentrantCallbackGroup()
            servers, clients, results, velocities = [], [], [], []
            reject_cancel = threading.Event()
            def cancel(handle):
                return CancelResponse.REJECT if reject_cancel.is_set() else CancelResponse.ACCEPT
            for name, action_type in [('/navigate_to_pose', NavigateToPose),
                                      ('/navigate_through_poses', NavigateThroughPoses),
                                      ('/follow_waypoints', FollowWaypoints)]:
                def execute(handle, cls=action_type):
                    end = time.monotonic()+12
                    while time.monotonic()<end:
                        if handle.is_cancel_requested:
                            time.sleep(.2); handle.canceled(); return cls.Result()
                        time.sleep(.01)
                    handle.abort(); return cls.Result()
                servers.append(ActionServer(node, action_type, name, execute,
                                            cancel_callback=cancel, callback_group=group))
                clients.append((ActionClient(node, action_type, name), action_type))
            node.create_subscription(Twist, '/cmd_vel', velocities.append, 10)
            manual = node.create_publisher(Twist, '/cmd_vel_manual', 10)
            worker = threading.Thread(target=executor.spin, daemon=True); worker.start()
            nav = FollowNavigation(); nav.start()
            def wait(predicate, timeout=5):
                end=time.monotonic()+timeout
                while time.monotonic()<end:
                    if predicate(): return
                    time.sleep(.01)
                self.fail('Timeout '+str(nav.snapshot()))
            try:
                wait(lambda: nav.initialized)
                self.assertFalse(nav.snapshot()['ready'])
                for client, cls in clients:
                    self.assertTrue(client.wait_for_server(timeout_sec=2))
                    future=client.send_goal_async(cls.Goal()); wait(future.done)
                    self.assertTrue(future.result().accepted)
                    results.append(future.result().get_result_async())
                wait(lambda: sum(len(ids) for ids in nav.foreign.values())==3)
                command=Twist(); command.linear.x=.8; command.angular.z=1.8
                manual.publish(command); time.sleep(.1)
                self.assertFalse(any(v.linear.x for v in velocities))
                reject_cancel.set()
                # Humble can ACK cancel-all with an empty list when callbacks
                # reject it. Active status must still prevent taking control.
                with self.assertRaisesRegex(RuntimeError, '拒绝取消|尚未确认停止'):
                    nav.take_manual_control(timeout=.6)
                self.assertFalse(nav.taking_over)
                reject_cancel.clear()
                errors=[]
                def takeover():
                    try: nav.take_manual_control()
                    except Exception as exc: errors.append(exc)
                handoff=threading.Thread(target=takeover); handoff.start()
                wait(lambda: nav.taking_over)
                manual.publish(command); time.sleep(.1)
                self.assertFalse(any(v.linear.x for v in velocities))
                handoff.join(timeout=7)
                self.assertFalse(handoff.is_alive()); self.assertEqual(errors, [])
                wait(lambda: all(f.done() for f in results))
                self.assertTrue(all(f.result().status==GoalStatus.STATUS_CANCELED for f in results))
                self.assertFalse(nav.snapshot()['foreign_busy'])
                manual.publish(command)
                wait(lambda: any(v.linear.x==.8 and v.angular.z==1.8 for v in velocities))
            finally:
                reject_cancel.clear(); nav.close(); executor.shutdown(); worker.join(timeout=2)
                for client, _ in clients: client.destroy()
                for server in servers: server.destroy()
                node.destroy_node(); context.shutdown()
