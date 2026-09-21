"""RealSense target observations -> timestamped TF -> RViz markers.

ROS is loaded lazily so camera-only use still works without a sourced ROS shell.
"""
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np


def target_point(depth_frame, box, deproject):
    """Median valid depth in a small central patch; optical x right/y down/z forward."""
    intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    x, y, w, h = box
    u = min(intr.width - 1, max(0, int(x + w / 2)))
    v = min(intr.height - 1, max(0, int(y + h / 2)))
    values = []
    for row in range(max(0, v - 2), min(intr.height, v + 3)):
        for col in range(max(0, u - 2), min(intr.width, u + 3)):
            z = depth_frame.get_distance(col, row)
            if math.isfinite(z) and 0.1 <= z <= 10.0:
                values.append(z)
    if len(values) < 5:
        return None
    point = tuple(deproject(intr, [u, v], float(np.median(values))))
    return point if all(math.isfinite(p) for p in point) else None


def rotation(rpy):
    r, p, y = np.radians(rpy)
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def mount_point(point, config, feedback):
    """Transform optical point into base_link using calibrated mount geometry."""
    if config['mode'] == 'fixed':
        return rotation(config['camera_rpy_deg']) @ point + config['camera_xyz_m']
    if (not feedback or not feedback.get('connected') or feedback.get('fault', 0) & ~1):
        raise ValueError('等待新鲜的云台角度反馈')
    # Rotations use right-hand rule. Positive rotation about +Y tips forward down;
    # pitch_sign must reflect the actual encoder direction of the installation.
    yaw = rotation([0, 0, feedback['yaw'] * config['yaw_sign']])
    secondary_deg = (feedback['pitch'] * config['pitch_sign']
                     if config.get('follow_secondary', config.get('follow_pitch', True)) else 0)
    # USB calls the second axis "pitch"; this robot physically has a roll axis.
    pitch = rotation([secondary_deg, 0, 0] if config.get('second_axis') == 'roll'
                     else [0, secondary_deg, 0])
    camera = rotation(config['camera_rpy_deg']) @ point + config['camera_xyz_m']
    relative = yaw @ (np.array(config['pitch_xyz_m']) + pitch @ camera)
    return rotation(config['mount_rpy_deg']) @ relative + config['mount_xyz_m']


def load_config(path):
    if not Path(path).exists():
        return {'mode': 'tf', 'camera_frame': 'camera_color_optical_frame', 'map_frame': 'map'}
    config = json.loads(Path(path).read_text())
    mode = config.get('mode')
    if mode not in ('tf', 'fixed', 'gimbal'):
        raise ValueError('相机安装配置 mode 必须为 tf、fixed 或 gimbal')
    fields = [] if mode == 'tf' else ['camera_xyz_m', 'camera_rpy_deg']
    if mode == 'gimbal':
        fields += ['mount_xyz_m', 'mount_rpy_deg', 'pitch_xyz_m']
        if config.get('second_axis', 'pitch') not in ('pitch', 'roll'):
            raise ValueError('second_axis 必须为 pitch 或 roll')
        if type(config.get('follow_secondary', True)) is not bool:
            raise ValueError('follow_secondary 必须为布尔值')
        if type(config.get('follow_pitch', True)) is not bool:
            raise ValueError('follow_pitch 必须为布尔值')
        for key in ('yaw_sign', 'pitch_sign'):
            if config.get(key) not in (-1, 1):
                raise ValueError(f'相机安装参数 {key} 必须为 -1 或 1')
    for key in fields:
        value = config.get(key)
        if (not isinstance(value, list) or len(value) != 3
                or not all(type(v) in (float, int) and math.isfinite(v) for v in value)):
            raise ValueError(f'相机安装参数 {key} 必须为三个有限数字')
    for key, default in [('map_frame', 'map'), ('camera_frame', 'camera_color_optical_frame')]:
        config.setdefault(key, default)
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key} 不能为空')
    return config


class CatMarkers:
    def __init__(self):
        self.lock = threading.RLock()
        self.thread = None
        self.stop_event = threading.Event()
        self.node = None
        self.pending = None
        self.generation = 0
        self.clear_pending = False
        self.ready = False
        self.state = {'message': '等待目标识别', 'position': None}
        self.last_publish = 0
        self.config_path = os.environ.get('CAT_CAMERA_CONFIG', str(Path(__file__).with_name('camera_mount.json')))

    def start(self):
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='cat-rviz', daemon=True)
                self.thread.start()

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def follow_observation(self):
        """Only a published, fresh map observation can drive navigation."""
        with self.lock:
            if not self.state.get('position') or not self.ready or not self.last_publish:
                return None
            return dict(position=list(self.state['position']), captured_at=self.last_publish,
                        config=dict(self.config), frame=self.config['map_frame'])

    def capture(self, feedback):
        self.start()
        with self.lock:
            if not self.ready:
                return None
            return (self.node.get_clock().now(), time.monotonic(), self.generation, feedback)

    def observe(self, point, capture):
        with self.lock:
            if capture and capture[2] == self.generation:
                self.pending = (point, capture)

    def clear(self, reason='未识别到目标猫'):
        with self.lock:
            self.generation += 1
            self.pending = None
            self.clear_pending = True
            self.state = {'message': reason, 'position': None}

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def _delete(self):
        from visualization_msgs.msg import Marker, MarkerArray
        markers = []
        for identifier in (0, 1):
            marker = Marker()
            marker.header.frame_id = self.config['map_frame']
            marker.ns = 'target_cat'
            marker.id = identifier
            marker.action = Marker.DELETE
            markers.append(marker)
        self.publisher.publish(MarkerArray(markers=markers))
        self.last_publish = 0

    def _publish(self, point, capture):
        from geometry_msgs.msg import PointStamped
        from tf2_geometry_msgs import do_transform_point
        from visualization_msgs.msg import Marker, MarkerArray
        from builtin_interfaces.msg import Duration
        stamp, captured_at, _, feedback = capture
        age = time.monotonic() - captured_at
        if age > 2.0:
            raise ValueError('识别结果已过期，等待新图像')
        source = PointStamped()
        source.header.stamp = stamp.to_msg()
        if self.config['mode'] == 'tf':
            source.header.frame_id = self.config['camera_frame']
        else:
            point = mount_point(point, self.config, feedback)
            source.header.frame_id = 'base_link'
        source.point.x, source.point.y, source.point.z = map(float, point)
        transform = self.buffer.lookup_transform(self.config['map_frame'], source.header.frame_id, stamp)
        located = do_transform_point(source, transform)
        located.header.stamp = stamp.to_msg()
        self.position_pub.publish(located)
        markers = []
        for identifier, kind in [(0, Marker.SPHERE), (1, Marker.TEXT_VIEW_FACING)]:
            marker = Marker()
            marker.header = located.header
            marker.ns, marker.id, marker.type = 'target_cat', identifier, kind
            marker.action = Marker.ADD
            marker.pose.position.x = located.point.x
            marker.pose.position.y = located.point.y
            marker.pose.position.z = located.point.z + (0.3 if identifier else 0)
            marker.pose.orientation.w = 1.0
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.1, 1.0, 0.25, 1.0
            marker.scale.x = marker.scale.y = 0.2
            marker.scale.z = 0.2 if not identifier else 0.18
            marker.text = f'Target cat ({located.point.x:.2f}, {located.point.y:.2f}) m' if identifier else ''
            marker.frame_locked = False
            ttl_ns = max(1, int((2.0 - age) * 1e9))
            marker.lifetime = Duration(sec=ttl_ns // 1_000_000_000,
                                       nanosec=ttl_ns % 1_000_000_000)
            markers.append(marker)
        self.publisher.publish(MarkerArray(markers=markers))
        self.last_publish = captured_at
        self.state = {'message': '目标猫位置已标注', 'frame': self.config['map_frame'],
                      'position': [located.point.x, located.point.y, located.point.z]}

    def _run(self):
        context = None
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from tf2_ros import Buffer, TransformListener, TransformException
            from visualization_msgs.msg import MarkerArray
            from geometry_msgs.msg import PointStamped
            self.config = load_config(self.config_path)
            context = Context()
            rclpy.init(context=context)
            node = rclpy.create_node('cat_target_markers', context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            self.buffer = Buffer()
            self.listener = TransformListener(self.buffer, node)
            self.publisher = node.create_publisher(MarkerArray, '/cat/markers', 10)
            self.position_pub = node.create_publisher(PointStamped, '/cat/position', 10)
            with self.lock:
                self.node, self.ready = node, True
            while not self.stop_event.is_set():
                executor.spin_once(timeout_sec=0.02)
                with self.lock:
                    if self.clear_pending:
                        self._delete()
                        self.clear_pending = False
                    item, self.pending = self.pending, None
                    if item:
                        try:
                            self._publish(*item)
                        except TransformException:
                            self._delete()
                            self.state = {'message': '等待地图/相机 TF；请配置相机安装参数', 'position': None}
                        except ValueError as exc:
                            self._delete()
                            self.state = {'message': str(exc), 'position': None}
                    if self.last_publish and time.monotonic() - self.last_publish > 2.0:
                        self._delete()
                        self.state = {'message': '目标位置已过期', 'position': None}
            self._delete()
            executor.shutdown()
        except Exception as exc:
            with self.lock:
                self.state = {'message': f'RViz 标注不可用：{exc}', 'position': None}
        finally:
            with self.lock:
                self.ready = False
                if self.node is not None:
                    self.node.destroy_node()
                    self.node = None
            if context and context.ok():
                context.shutdown()
