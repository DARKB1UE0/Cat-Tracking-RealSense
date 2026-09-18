import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from cat_markers import CatMarkers, load_config, mount_point, rotation, target_point


class Depth:
    def __init__(self, values):
        self.values = values
        self.profile = self

    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return SimpleNamespace(width=20, height=20)

    def get_distance(self, x, y):
        return self.values.get((x, y), 2.0)


class GeometryTests(unittest.TestCase):
    def test_patch_rejects_invalid_depth_and_uses_median(self):
        depth = Depth({(10, 10): 0, (9, 9): float('nan'), (8, 8): 9.0})
        calls = []
        def deproject(intr, uv, z):
            calls.append((uv, z))
            return [0.2, 0.1, z]
        self.assertEqual(target_point(depth, (5, 5, 10, 10), deproject), (0.2, 0.1, 2.0))
        self.assertEqual(calls, [([10, 10], 2.0)])
        depth.get_distance = lambda x, y: 0
        self.assertIsNone(target_point(depth, (5, 5, 10, 10), deproject))

    def test_edge_patch_clipped(self):
        def distance(x, y):
            self.assertTrue(0 <= x < 20 and 0 <= y < 20)
            return 1
        depth = Depth({})
        depth.get_distance = distance
        self.assertEqual(target_point(depth, (-5, -5, 10, 10), lambda i, uv, z: [0, 0, z]), (0, 0, 1))

    def test_optical_axes_and_fixed_mount(self):
        config = dict(mode='fixed', camera_rpy_deg=[-90, 0, -90], camera_xyz_m=[.2, 0, .4])
        np.testing.assert_allclose(mount_point([0, 0, 2], config, None), [2.2, 0, .4], atol=1e-10)
        np.testing.assert_allclose(rotation([-90, 0, -90]) @ [1, 2, 3], [3, -1, -2], atol=1e-10)

    def test_gimbal_measured_angles_and_lever_arms(self):
        config = dict(mode='gimbal', mount_xyz_m=[.2, 0, .3], mount_rpy_deg=[0, 0, 0],
                      pitch_xyz_m=[.1, 0, 0], camera_xyz_m=[.05, 0, 0],
                      camera_rpy_deg=[-90, 0, -90], yaw_sign=1, pitch_sign=1)
        feedback = dict(connected=True, fault=1, yaw=90, pitch=0)
        np.testing.assert_allclose(mount_point([0, 0, 2], config, feedback), [.2, 2.15, .3], atol=1e-10)
        feedback.update(yaw=0, pitch=30)
        np.testing.assert_allclose(mount_point([0, 0, 2], config, feedback),
                                   [.3 + 2.05*np.cos(np.pi/6), 0, .3 - 2.05*.5], atol=1e-10)
        feedback['connected'] = False
        with self.assertRaises(ValueError): mount_point([0, 0, 2], config, feedback)

    def test_config_requires_measured_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'mount.json'
            self.assertEqual(load_config(path)['mode'], 'tf')
            path.write_text(json.dumps({'mode': 'gimbal'}))
            with self.assertRaises(ValueError): load_config(path)
            path.write_text(json.dumps({'mode': 'fixed', 'camera_xyz_m': [0, 0, 0], 'camera_rpy_deg': [0, 0, float('nan')]}))
            with self.assertRaises(ValueError): load_config(path)

    def test_configured_camera_follows_measured_yaw(self):
        config = load_config(Path(__file__).resolve().parents[1] / 'camera_mount.json')
        feedback = dict(connected=True, fault=1, yaw=90, pitch=30)
        np.testing.assert_allclose(mount_point([0, 0, 2], config, feedback),
                                   [0, 2, .15], atol=1e-10)
        feedback['yaw'] = -90
        np.testing.assert_allclose(mount_point([0, 0, 2], config, feedback),
                                   [0, -2, .15], atol=1e-10)

    def test_stop_invalidates_in_flight_detection(self):
        markers = CatMarkers()
        capture = (None, 0, markers.generation, None)
        markers.clear('stop')
        markers.observe([1, 2, 3], capture)
        self.assertIsNone(markers.pending)


if __name__ == '__main__': unittest.main()
