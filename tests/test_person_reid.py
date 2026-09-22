"""ReID preprocessing, verified model caching, and invalid-output handling."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import person_reid


class PersonReIDTests(unittest.TestCase):
    def test_input_keeps_bgr_range_and_full_body(self):
        image = np.full((300, 150, 3), (12, 80, 240), np.uint8)
        image[:50] = (1, 2, 3)
        tensor = person_reid.prepare_input(image)
        self.assertEqual(tensor.shape, (1, 3, 256, 128))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [1, 2, 3])
        np.testing.assert_array_equal(tensor[0, :, -1, -1], [12, 80, 240])
        for invalid in (None, np.empty((0, 3, 3)), np.zeros((2, 2)), np.full((3, 3, 3), np.nan)):
            with self.assertRaises(ValueError): person_reid.prepare_input(invalid)

    def test_embedding_normalization_rejects_bad_output(self):
        vector = person_reid.normalize_embedding(np.arange(256)[None])
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1., places=6)
        for invalid in (np.zeros(256), np.ones(255), np.full(256, np.nan), np.full(256, np.inf)):
            with self.assertRaises(ValueError): person_reid.normalize_embedding(invalid)

    def test_verified_cache_is_offline_and_corruption_is_repaired(self):
        payload = b'model test bytes'
        specification = {'xml': (len(payload), hashlib.sha384(payload).hexdigest())}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(person_reid, 'MODEL_FILES', specification), \
             patch.object(person_reid, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(payload)) as request:
            path = person_reid.ensure_model(tmp)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(person_reid.ensure_model(tmp), path)
            self.assertEqual(request.call_count, 1)
            path.write_bytes(b'corrupt')
            person_reid.ensure_model(tmp)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(request.call_count, 2)

    def test_bad_download_and_offline_error_leave_no_partial_weights(self):
        payload = b'good'
        specification = {'xml': (len(payload), hashlib.sha384(payload).hexdigest())}
        for response in (lambda *a, **k: io.BytesIO(b'evil'), OSError('offline')):
            with tempfile.TemporaryDirectory() as tmp, \
                 patch.object(person_reid, 'MODEL_FILES', specification), \
                 patch.object(person_reid, 'urlopen', side_effect=response):
                with self.assertRaisesRegex(RuntimeError, '下载/校验失败'):
                    person_reid.ensure_model(tmp)
                self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
