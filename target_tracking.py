"""Reference appearance matching for cats and clothed people in test mode."""
import math
import threading
import cv2
import numpy as np

PERSON_MATCH_THRESHOLD = .15


def clothing_crop(image):
    """Exclude head, feet and outer background; retain upper/lower clothing."""
    h, w = image.shape[:2]
    return image[int(h*.20):int(h*.90), int(w*.15):int(w*.85)]


def clothing_histograms(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    middle = max(1, len(hsv)//2)
    histograms = []
    for part in (hsv[:middle], hsv[middle:]):
        if not part.size:
            raise ValueError('参考人物区域太小')
        hist = cv2.calcHist([part], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
        histograms.append(hist.ravel() / max(float(hist.sum()), 1.))
    return np.asarray(histograms)


def clothing_similarity(reference, candidate):
    return float(np.sqrt(reference * candidate).sum(axis=1).mean())


def select_candidate(candidates, threshold, margin):
    ranked = sorted(candidates, key=lambda c: c['match_score'], reverse=True)
    if not ranked or not math.isfinite(ranked[0]['match_score']) or ranked[0]['match_score'] < threshold:
        return None
    if len(ranked) > 1 and ranked[0]['match_score']-ranked[1]['match_score'] < margin:
        return None
    return ranked[0]


def box_overlap(a, b):
    x = max(a[0], b[0]); y = max(a[1], b[1])
    w = max(0, min(a[0]+a[2], b[0]+b[2])-x)
    h = max(0, min(a[1]+a[3], b[1]+b[3])-y)
    intersection = w*h
    return intersection/max(1, a[2]*a[3]+b[2]*b[3]-intersection)


class PersonConfirmation:
    def __init__(self):
        self.box = None
        self.count = 0

    def select(self, candidates):
        best = select_candidate(candidates, PERSON_MATCH_THRESHOLD, .08)
        if best is None or not best.get('match_eligible', True):
            self.box = None; self.count = 0
            return None
        self.count = self.count+1 if self.box and box_overlap(self.box, best['box']) >= .15 else 1
        self.box = best['box']
        return best if self.count >= 3 else None


class ReferenceTracker:
    def __init__(self, image_path, mode='cat'):
        if mode not in ('cat', 'person'):
            raise ValueError('未知识别模式')
        self.mode = mode
        self.class_id = 0 if mode == 'person' else 15
        self.lock = threading.Lock()
        self.confirmation = PersonConfirmation()
        if mode == 'person':
            from ultralytics import YOLO
            from person_reid import PersonReID
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f'无法读取参考图片: {image_path}')
            self.yolo_model = YOLO('yolov8n.pt')
            reference = self._person_reference(self.yolo_model, image)
            self.reid = PersonReID()
            self.reference_feature = self.reid.embed(reference)
            self.reference_colors = clothing_histograms(clothing_crop(reference))
        else:
            from track_specific_cat import CatTracker
            self.engine = CatTracker(image_path)
            self.yolo_model = self.engine.yolo_model

    def _boxes(self, model, image):
        result = model(image, classes=[self.class_id], verbose=False)[0]
        boxes = []
        if result.boxes is None:
            return boxes
        h, w = image.shape[:2]
        for detection in result.boxes:
            if int(detection.cls) != self.class_id or float(detection.conf) <= .5:
                continue
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2-x1 > 30 and y2-y1 > (60 if self.mode == 'person' else 30):
                boxes.append(((x1, y1, x2-x1, y2-y1), float(detection.conf)))
        return boxes

    def _person_reference(self, model, image):
        boxes = self._boxes(model, image)
        if len(boxes) != 1:
            raise ValueError('测试模式参考照片必须清晰包含且仅包含一人，请使用衣着完整可见的单人照')
        (x, y, w, h), _ = boxes[0]
        # ReID is trained on full-body boxes; only the auxiliary color crop is trimmed.
        return image[y:y+h, x:x+w]

    def detect(self, image):
        # Multiple MJPEG clients must not run the same model concurrently.
        with self.lock:
            try:
                return self._detect(image)
            except Exception:
                # An invalid/missing observation breaks consecutive confirmation.
                self.confirmation = PersonConfirmation()
                raise

    def _detect(self, image):
        candidates = []
        for box, confidence in self._boxes(self.yolo_model, image):
            x, y, w, h = box
            roi = image[y:y+h, x:x+w]
            details = {}
            if self.mode == 'person':
                body = clothing_crop(roi)
                appearance = self.reid.similarity(self.reference_feature, roi)
                color = clothing_similarity(self.reference_colors, clothing_histograms(body))
                score = .90*appearance + .10*color
                # Keep measured scores visible even when selection is rejected.
                details = dict(appearance_score=appearance, color_score=color,
                               match_eligible=(math.isfinite(appearance) and
                                               math.isfinite(color) and
                                               score >= PERSON_MATCH_THRESHOLD))
            else:
                score = self.engine.match_cat(roi)
            candidates.append(dict(box=box, confidence=confidence, match_score=score, **details))
        target = (self.confirmation.select(candidates) if self.mode == 'person'
                  else select_candidate(candidates, .75, .05))
        return candidates, target
