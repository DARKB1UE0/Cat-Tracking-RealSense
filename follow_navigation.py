"""Nav2 action client and manual-command arbitration for automatic cat following."""
import math
import threading
import time
from pathlib import Path


def follow_behavior_tree(align=False):
    from ament_index_python.packages import get_package_share_directory
    filename = 'align_target.xml' if align else 'follow_target.xml'
    path = Path(get_package_share_directory('wheeltec_bringup')) / 'config' / filename
    if not path.is_file():
        raise RuntimeError('缺少目标跟随配置，请编译 wheeltec_bringup 并重启导航与网页')
    return str(path)


class FollowNavigation:
    FOLLOW_SPEED_LIMIT = 0.50  # m/s; keep FollowTarget and AlignTarget baselines in sync

    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.node = None
        self.initialized = False
        self.following = False
        self.goal_future = self.goal_handle = self.cancel_future = None
        self.cancel_requested = False
        self.cancel_at = 0
        self.result_seq = 0
        self.outcome = None
        self.foreign = {}
        self.terminal = {}
        self.own_ids = set()
        self.manual_at = 0
        self.last_manual = None
        self.robot = None
        self.ready = False
        self.error = '等待 ROS 导航连接'
        self.active_at = 0
        self.set_at = 0
        self.taking_over = False
        self.cancel_clients = {}

    def start(self):
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='follow-nav2', daemon=True)
                self.thread.start()

    def _busy(self):
        return self.goal_future is not None or self.goal_handle is not None

    def _foreign_busy(self):
        return any(identifier not in self.own_ids for ids in self.foreign.values() for identifier in ids)

    def snapshot(self):
        with self.lock:
            return dict(ready=self.ready, error=self.error, robot=self.robot,
                        busy=self._busy() or self.taking_over, foreign_busy=self._foreign_busy(),
                        manual_active=time.monotonic()-self.manual_at < .6,
                        result_seq=self.result_seq, outcome=self.outcome)

    def _zero(self):
        if self.initialized:
            from geometry_msgs.msg import Twist
            self.velocity.publish(Twist())

    def take_manual_control(self, timeout=6.0):
        """Cancel Nav2 tasks, await terminal status and drain smoother output.

        Runs in an HTTP thread; the ROS executor must remain free for responses.
        Manual control does not require a valid map transform.
        """
        from action_msgs.srv import CancelGoal
        with self.lock:
            if not self.initialized or self.taking_over:
                raise RuntimeError('导航接口未就绪或正在切换控制，请稍后重试')
            self.taking_over = True
        try:
            end = time.monotonic() + timeout
            futures = []
            for topic, client in self.cancel_clients.items():
                available = client.wait_for_service(timeout_sec=.3)
                with self.lock:
                    if not available and self.foreign.get(topic):
                        raise RuntimeError('导航取消服务不可用，尚未接管键盘')
                if available:
                    # Zero UUID + timestamp explicitly cancels every current goal.
                    futures.append((topic, client.call_async(CancelGoal.Request())))
            quiet_since = None
            while time.monotonic() < end:
                with self.lock:
                    if not self.initialized:
                        raise RuntimeError('ROS 接口已断开')
                    self._zero()
                    done = all(f.done() for _, f in futures)
                    confirmed = done
                    if done:
                        for topic, future in futures:
                            response = future.result()
                            if response.return_code != CancelGoal.Response.ERROR_NONE:
                                raise RuntimeError('导航拒绝取消，请检查 Nav2 状态后重试')
                            confirmed = confirmed and all(
                                bytes(goal.goal_id.uuid) in self.terminal.get(topic, set())
                                for goal in response.goals_canceling)
                    if confirmed and not self._busy() and not self._foreign_busy():
                        quiet_since = quiet_since or time.monotonic()
                        # Allow both the previous 1 s and current 0.25 s smoother timeout.
                        if time.monotonic() - quiet_since >= 1.1:
                            self.last_manual = None
                            return
                    else:
                        quiet_since = None
                time.sleep(.02)
            raise RuntimeError('导航尚未确认停止，键盘未接管，请重试')
        finally:
            with self.lock:
                self.taking_over = False

    def _limit(self, speed):
        from nav2_msgs.msg import SpeedLimit
        msg = SpeedLimit()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.percentage = False
        msg.speed_limit = float(speed)
        self.speed_limit.publish(msg)

    def set_following(self, enabled):
        with self.lock:
            self.following = enabled
            self.active_at = time.monotonic()
            if self.initialized:
                self._limit(self.FOLLOW_SPEED_LIMIT if enabled or self._busy() else 0)

    def cancel(self):
        with self.lock:
            if not self._busy():
                return
            self.cancel_requested = True
            self._zero()
            self._send_cancel()

    def _send_cancel(self):
        if self.goal_handle is None or self.cancel_future is not None:
            return
        if time.monotonic() - self.cancel_at < .5:
            return
        self.cancel_at = time.monotonic()
        self.cancel_future = self.goal_handle.cancel_goal_async()
        self.cancel_future.add_done_callback(self._canceled)

    def _canceled(self, future):
        with self.lock:
            if future is not self.cancel_future:
                return  # A terminal result/new goal can precede an old cancel ACK.
            # Wait for the terminal result, even after cancellation ACK.
            self.cancel_future = None
            try:
                response = future.result()
                if not response.goals_canceling:
                    self.error = '等待导航停止确认'
            except Exception:
                self.error = '导航取消请求失败，等待重试'

    def send_goal(self, pose, align=False):
        with self.lock:
            if (not self.ready or not self.following or self._busy() or self.taking_over
                    or self._foreign_busy() or time.monotonic()-self.active_at < .15):
                return False
            from nav2_msgs.action import NavigateToPose
            from unique_identifier_msgs.msg import UUID
            import uuid
            goal = NavigateToPose.Goal()
            goal.behavior_tree = follow_behavior_tree(align=align)
            goal.pose.header.frame_id = 'map'
            goal.pose.header.stamp = self.node.get_clock().now().to_msg()
            goal.pose.pose.position.x, goal.pose.pose.position.y = pose[:2]
            goal.pose.pose.orientation.z = math.sin(pose[2]/2)
            goal.pose.pose.orientation.w = math.cos(pose[2]/2)
            identifier = uuid.uuid4().bytes
            self.own_ids.add(identifier)
            self.cancel_requested = False
            self.outcome = None
            self.set_at = time.monotonic()
            self.goal_future = self.client.send_goal_async(goal, goal_uuid=UUID(uuid=list(identifier)))
            self.goal_future.add_done_callback(self._accepted)
            return True

    def _accepted(self, future):
        with self.lock:
            self.goal_future = None
            try:
                handle = future.result()
                if not handle.accepted:
                    self.cancel_requested = False
                    self.outcome = 'failed'
                    self.result_seq += 1
                    if not self.following:
                        self._limit(0)
                    return
                self.goal_handle = handle
                handle.get_result_async().add_done_callback(self._result)
                if self.cancel_requested or not self.following:
                    self.cancel_requested = True
                    self._send_cancel()
            except Exception:
                self.cancel_requested = False
                self.outcome = 'failed'
                self.result_seq += 1

    def _result(self, future):
        with self.lock:
            from action_msgs.msg import GoalStatus
            try:
                status = future.result().status
                if self.cancel_requested:
                    self.outcome = 'canceled'
                elif status == GoalStatus.STATUS_SUCCEEDED:
                    self.outcome = 'succeeded'
                else:
                    self.outcome = 'failed'
            except Exception:
                self.outcome = 'failed'
            self.goal_handle = None
            self.cancel_future = None
            self.cancel_requested = False
            self.result_seq += 1
            if not self.following:
                self._limit(0)

    def _status(self, message, topic):
        with self.lock:
            self.foreign[topic] = {bytes(s.goal_info.goal_id.uuid) for s in message.status_list if s.status in (1, 2, 3)}
            self.terminal[topic] = {bytes(s.goal_info.goal_id.uuid) for s in message.status_list if s.status in (4, 5, 6)}

    def _manual(self, message):
        with self.lock:
            values = [message.linear.x, message.linear.y, message.linear.z,
                      message.angular.x, message.angular.y, message.angular.z]
            if not all(math.isfinite(v) for v in values):
                return
            if self.following or self._busy() or self._foreign_busy() or self.taking_over:
                return
            if any(abs(v) > 1e-6 for v in values):
                self.manual_at = time.monotonic()
                self.last_manual = self.manual_at
            else:
                self.manual_at = 0
                self.last_manual = None
            self.velocity.publish(message)

    def _run(self):
        context = executor = node = None
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.action import ActionClient
            from rclpy.qos import QoSProfile, DurabilityPolicy
            from tf2_ros import Buffer, TransformListener
            from nav2_msgs.action import NavigateToPose
            from nav2_msgs.msg import SpeedLimit
            from geometry_msgs.msg import Twist
            from action_msgs.msg import GoalStatusArray
            from action_msgs.srv import CancelGoal
            context = Context()
            rclpy.init(context=context)
            node = rclpy.create_node('cat_follow_navigation', context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            buffer = Buffer()
            listener = TransformListener(buffer, node)
            with self.lock:
                self.node = node
                self.client = ActionClient(node, NavigateToPose, '/navigate_to_pose')
                # Velocity is a latest-value stream, not a queue of old moves.
                self.velocity = node.create_publisher(Twist, '/cmd_vel', 1)
                self.speed_limit = node.create_publisher(SpeedLimit, '/speed_limit', 10)
                node.create_subscription(Twist, '/cmd_vel_manual', self._manual, 1)
                qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                for action in ['/navigate_to_pose', '/navigate_through_poses', '/follow_waypoints']:
                    topic = action + '/_action/status'
                    self.cancel_clients[topic] = node.create_client(CancelGoal, action + '/_action/cancel_goal')
                    node.create_subscription(GoalStatusArray, topic, lambda msg, t=topic: self._status(msg, t), qos)
                self.initialized = True
            while not self.stop_event.is_set():
                executor.spin_once(timeout_sec=.02)
                with self.lock:
                    try:
                        transform = buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                        age = (node.get_clock().now()-rclpy.time.Time.from_msg(transform.header.stamp)).nanoseconds/1e9
                        if age > .5 or age < -.5:
                            raise ValueError('地图定位已过期')
                        pos, q = transform.transform.translation, transform.transform.rotation
                        heading = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                        self.robot = (pos.x, pos.y, pos.z, heading)
                        self.ready = self.client.server_is_ready() and node.count_subscribers('/speed_limit') > 0
                        self.error = '' if self.ready else '等待 Nav2 导航和速度限制接口'
                    except Exception:
                        self.robot = None
                        self.ready = False
                        self.error = '等待有效的 map → base_link 定位'
                    if self.goal_future is not None and time.monotonic()-self.set_at > 2:
                        self.ready = False
                        self.error = '导航目标响应超时'
                        self.cancel_requested = True
                    if self.following or self._busy():
                        self._limit(self.FOLLOW_SPEED_LIMIT)
                    if self.cancel_requested:
                        self._zero()
                        self._send_cancel()
                    if self.last_manual and time.monotonic()-self.last_manual > .35:
                        if not self.following and not self._busy() and not self._foreign_busy():
                            self._zero()
                        self.last_manual = None
            with self.lock:
                self._zero()
                self._limit(0)
        except Exception as exc:
            with self.lock:
                self.ready = False
                self.error = f'导航接口不可用：{exc}'
                self._zero()
        finally:
            with self.lock:
                self.initialized = False
            if executor: executor.shutdown()
            if node: node.destroy_node()
            if context and context.ok(): context.shutdown()

    def close(self):
        self.set_following(False)
        self.cancel()
        # Allow callbacks to cancel a goal that was still awaiting acceptance.
        end = time.monotonic()+2
        while self.snapshot()['busy'] and time.monotonic()<end:
            time.sleep(.02)
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=1)
