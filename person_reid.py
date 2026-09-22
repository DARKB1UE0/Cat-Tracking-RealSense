"""Dedicated OmniScaleNet ReID with verified, locally cached Open Model Zoo weights.

Model: person-reidentification-retail-0287 (Apache-2.0).
Input is a full body crop, BGR float32 in [0, 255], NCHW 1x3x256x128.
The IR already contains normalization. Do not apply ImageNet normalization again.
"""
import hashlib
import os
from pathlib import Path
import tempfile
import threading
from urllib.request import urlopen

import cv2
import numpy as np


MODEL_NAME = 'person-reidentification-retail-0287'
MODEL_BASE = ('https://storage.openvinotoolkit.org/repositories/open_model_zoo/'
              f'2023.0/models_bin/1/{MODEL_NAME}/FP32')
# Sizes and SHA-384 digests from Open Model Zoo's model.yml.
MODEL_FILES = {
    'xml': (451147, 'a8a6956dec193dfa0095319c5f0bfccd5b9eaaac4d30ddc12a79aa7a99f0dcf953aae7b76764239d383ff39f8f8a1477'),
    'bin': (2362128, '7e962ed07992edeff11eac8f6280ea5761d8ddfb699bcd3c79cfeac10773c5f4d4577ca4c048a7fa50977cdfc14a74ba'),
}
_download_lock = threading.Lock()


def model_directory():
    override = os.environ.get('PERSON_REID_MODEL_DIR')
    if override:
        return Path(override).expanduser()
    cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache')))
    return cache / 'cat-tracking' / MODEL_NAME


def _verified(path, size, checksum):
    return (path.is_file() and path.stat().st_size == size and
            hashlib.sha384(path.read_bytes()).hexdigest() == checksum)


def ensure_model(directory=None):
    directory = Path(directory) if directory is not None else model_directory()
    with _download_lock:
        directory.mkdir(parents=True, exist_ok=True)
        for suffix, (size, checksum) in MODEL_FILES.items():
            path = directory / f'{MODEL_NAME}.{suffix}'
            if _verified(path, size, checksum):
                continue
            temporary = None
            try:
                with urlopen(f'{MODEL_BASE}/{path.name}', timeout=30) as response, \
                     tempfile.NamedTemporaryFile(dir=directory, delete=False) as out:
                    temporary = Path(out.name)
                    # Bound response size and verify before replacing the cached file.
                    out.write(response.read(size + 1))
                if not _verified(temporary, size, checksum):
                    raise ValueError('模型大小或 SHA-384 校验失败')
                temporary.replace(path)
            except Exception as exc:
                raise RuntimeError(
                    f'人物 ReID 模型下载/校验失败：{exc}。'
                    '请联网运行 python3 person_reid.py 预下载模型后重试。') from exc
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return directory / f'{MODEL_NAME}.xml'


def prepare_input(image):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError('人物 ReID 输入必须为非空 BGR 人体图像')
    if not np.isfinite(image).all():
        raise ValueError('人物 ReID 输入包含无效像素')
    resized = cv2.resize(image, (128, 256), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)


def normalize_embedding(output):
    vector = np.asarray(output, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size != 256 or not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('人物 ReID 模型输出无效，已拒绝匹配')
    return vector / norm


class PersonReID:
    def __init__(self, directory=None):
        try:
            from openvino import Core
        except ImportError as exc:
            raise RuntimeError('人物 ReID 需要 OpenVINO，请运行 python3 -m pip install -r requirements.txt') from exc
        self.model_path = ensure_model(directory)
        core = Core()
        model = core.read_model(str(self.model_path))
        if tuple(model.input().shape) != (1, 3, 256, 128) or tuple(model.output().shape) != (1, 256):
            raise ValueError('人物 ReID 模型输入输出维度不匹配')
        self.compiled = core.compile_model(model, 'CPU', {
            'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_NUM_THREADS': 2,
        })
        self.output = self.compiled.output(0)
        self.lock = threading.Lock()

    def embed(self, image):
        tensor = prepare_input(image)
        with self.lock:
            result = self.compiled([tensor])[self.output]
            return normalize_embedding(result)

    def similarity(self, reference, image):
        # Negative cosine similarity is a valid non-match; display on a 0..1 scale.
        return float(np.clip(np.dot(reference, self.embed(image)), 0., 1.))


if __name__ == '__main__':
    model = PersonReID()
    vector = model.embed(np.full((256, 128, 3), 127, dtype=np.uint8))
    print(f'ReID ready: {model.model_path} ({vector.size} dimensions, CPU)')
