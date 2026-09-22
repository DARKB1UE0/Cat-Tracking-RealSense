"""Person appearance and ambiguity tests, without camera/model downloads."""
import threading
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from target_tracking import clothing_crop, clothing_histograms, clothing_similarity, PersonConfirmation, ReferenceTracker
from tracking_session import TrackingSession


class PersonAppearanceTests(unittest.TestCase):
    def test_rejected_person_preserves_measured_scores(self):
        tracker = ReferenceTracker.__new__(ReferenceTracker)
        tracker.mode = 'person'; tracker.class_id = 0; tracker.lock = threading.Lock()
        tracker.confirmation = PersonConfirmation()
        model = Mock(return_value=[Mock(boxes=[Mock(cls=0, conf=.9, xyxy=[[0, 0, 80, 200]])])])
        tracker.yolo_model = model
        tracker.reid = Mock()
        tracker.reference_feature = np.ones(256, np.float32)
        image = np.full((200, 80, 3), (0, 0, 220), np.uint8)
        tracker.reference_colors = clothing_histograms(clothing_crop(image))
        # Keep the real score visible when the combined threshold fails.
        tracker.reid.similarity.return_value = .1
        for _ in range(4):
            candidates, target = tracker.detect(np.full_like(image, (220, 0, 0)))
            self.assertIsNone(target)
            self.assertAlmostEqual(candidates[0]['match_score'], .09)
            self.assertAlmostEqual(candidates[0]['appearance_score'], .1)
            self.assertAlmostEqual(candidates[0]['color_score'], 0.)
            self.assertFalse(candidates[0]['match_eligible'])
        # No hidden 0.65 ReID gate: .2 * .9 + 1 * .1 = .28 can confirm.
        tracker.reid.similarity.return_value = .2
        self.assertIsNone(tracker.detect(image)[1])
        self.assertIsNone(tracker.detect(image)[1])
        self.assertIsNotNone(tracker.detect(image)[1])
        tracker.detect(np.zeros((1, 1, 3), np.uint8))
        # Color is auxiliary; strong ReID can confirm despite a low color score.
        tracker.reid.similarity.return_value = .9
        changed_color = np.full_like(image, (220, 0, 0))
        self.assertIsNone(tracker.detect(changed_color)[1])
        self.assertIsNone(tracker.detect(changed_color)[1])
        candidates, target = tracker.detect(changed_color)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(candidates[0]['match_score'], .81)
        self.assertTrue(candidates[0]['match_eligible'])
        self.assertEqual(tracker.reid.similarity.call_args.args[1].shape, image.shape)
        tracker.detect(np.zeros((1, 1, 3), np.uint8))  # No valid boxes resets confirmation.
        # Good scores still require three successive confirmations.
        self.assertIsNone(tracker.detect(image)[1])
        self.assertIsNone(tracker.detect(image)[1])
        self.assertIsNotNone(tracker.detect(image)[1])
        tracker.reid.similarity.side_effect = ValueError('invalid embedding')
        with self.assertRaisesRegex(ValueError, 'invalid embedding'): tracker.detect(image)
        tracker.reid.similarity.side_effect = None
        self.assertIsNone(tracker.detect(image)[1])
        self.assertIsNone(tracker.detect(image)[1])
        self.assertIsNotNone(tracker.detect(image)[1])

    def test_person_initialization_uses_reid_without_cat_engine(self):
        image = np.full((200, 80, 3), 120, np.uint8)
        detector = Mock(return_value=[Mock(boxes=[Mock(cls=0, conf=.9, xyxy=[[0, 0, 80, 200]])])])
        cat_module = Mock()
        with patch.dict(sys.modules, {'ultralytics': Mock(YOLO=Mock(return_value=detector)),
                                     'track_specific_cat': cat_module}), \
             patch('target_tracking.cv2.imread', return_value=image), \
             patch('person_reid.PersonReID') as reid:
            tracker = ReferenceTracker('reference.jpg', 'person')
            cat_module.CatTracker.assert_not_called()
            self.assertIs(tracker.reid, reid.return_value)
            np.testing.assert_array_equal(reid.return_value.embed.call_args.args[0], image)
            reid.side_effect = RuntimeError('model unavailable')
            with self.assertRaisesRegex(RuntimeError, 'model unavailable'):
                ReferenceTracker('reference.jpg', 'person')
            cat_module.CatTracker.assert_not_called()

    def test_person_threshold_is_point_one_five(self):
        for score, accepted in ((.149, False), (.15, True)):
            selector = PersonConfirmation()
            candidate = dict(box=(0, 0, 80, 200), match_score=score, match_eligible=True)
            for _ in range(3): result = selector.select([candidate])
            self.assertEqual(result is not None, accepted)

    def test_rejected_rival_does_not_hide_ambiguity(self):
        selector = PersonConfirmation()
        best = dict(box=(10, 10, 80, 200), match_score=.86, match_eligible=True)
        rival = dict(box=(150, 10, 80, 200), match_score=.82, match_eligible=False)
        for _ in range(4):
            self.assertIsNone(selector.select([best, rival]))

    def test_cat_mode_keeps_cat_class_and_existing_ambiguity_threshold(self):
        tracker = ReferenceTracker.__new__(ReferenceTracker)
        tracker.mode = 'cat'; tracker.class_id = 15; tracker.lock = threading.Lock()
        def box(cls, x): return Mock(cls=cls, conf=.9, xyxy=[[x, 0, x+60, 100]])
        model = Mock(return_value=[Mock(boxes=[box(0, 0), box(15, 0), box(15, 100)])])
        tracker.engine = Mock(yolo_model=model)
        tracker.yolo_model = model
        tracker.engine.match_cat.side_effect = [.9, .86]
        image = np.zeros((120, 200, 3), np.uint8)
        candidates, target = tracker.detect(image)
        self.assertEqual(len(candidates), 2); self.assertIsNone(target)
        self.assertEqual(model.call_args.kwargs['classes'], [15])
        tracker.engine.match_cat.side_effect = [.9, .7]
        candidates, target = tracker.detect(image)
        self.assertIs(target, candidates[0])

    def test_clothing_color_layout_and_head_exclusion(self):
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        image[:110] = (0, 0, 220); image[110:] = (220, 0, 0)
        reference = clothing_histograms(clothing_crop(image))
        changed_head = image.copy(); changed_head[:40] = (0, 255, 0)
        self.assertAlmostEqual(clothing_similarity(reference, clothing_histograms(clothing_crop(changed_head))), 1.)
        swapped = image.copy(); swapped[:110] = (220, 0, 0); swapped[110:] = (0, 0, 220)
        self.assertLess(clothing_similarity(reference, clothing_histograms(clothing_crop(swapped))), .2)
        self.assertLess(clothing_similarity(reference, clothing_histograms(np.full((100, 50, 3), 255, np.uint8))), .2)

    def test_confirmation_ambiguity_loss_and_spatial_jump(self):
        selector = PersonConfirmation()
        best = dict(box=(10, 10, 80, 200), match_score=.9)
        self.assertIsNone(selector.select([best])); self.assertIsNone(selector.select([best]))
        self.assertIs(selector.select([best]), best)
        rival = dict(box=(150, 10, 80, 200), match_score=.86)
        self.assertIsNone(selector.select([best, rival]))
        self.assertIsNone(selector.select([best]))
        self.assertIsNone(selector.select([]))
        self.assertIsNone(selector.select([dict(best, match_score=.09)]))
        for _ in range(3): selector.select([best])
        self.assertIsNone(selector.select([dict(best, box=(400, 10, 80, 200))]))

    def test_person_reference_requires_one_visible_person(self):
        tracker = ReferenceTracker.__new__(ReferenceTracker)
        tracker.mode = 'person'; tracker.class_id = 0
        image = np.zeros((250, 200, 3), np.uint8)
        def detection(cls, box): return Mock(cls=cls, conf=.9, xyxy=[box])
        model = Mock(return_value=[Mock(boxes=[detection(15, (0, 0, 80, 160))])])
        with self.assertRaisesRegex(ValueError, '仅包含一人'): tracker._person_reference(model, image)
        model.return_value = [Mock(boxes=[detection(0, (0, 0, 80, 160)), detection(0, (100, 0, 180, 160))])]
        with self.assertRaisesRegex(ValueError, '仅包含一人'): tracker._person_reference(model, image)
        model.return_value = [Mock(boxes=[detection(0, (-10, 0, 100, 200))])]
        self.assertEqual(tracker._person_reference(model, image).shape, (200, 100, 3))
        self.assertEqual(model.call_args.kwargs['classes'], [0])


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.markers, self.follow = Mock(), Mock()
        self.session = TrackingSession(self.markers, self.follow, lambda path, mode: object())

    def test_mode_switch_invalidates_reference_and_old_inference(self):
        self.session.register_reference('cat.jpg', 'cat', 0)
        self.session.start('cat.jpg', 'cat', 0)
        sample = self.session.capture({})
        self.session.set_mode('person')
        self.assertFalse(self.session.snapshot()['tracking_active'])
        self.markers.set_target_kind.assert_called_with('person')
        self.follow.stop.assert_called()
        self.session.observe(sample, [1, 2, 3])
        self.markers.observe.assert_not_called()
        with self.assertRaises(ValueError): self.session.start('cat.jpg', 'person', 1)
        with self.assertRaises(ValueError): self.session.register_reference('late.jpg', 'cat', 0)
        self.session.set_mode('cat')
        with self.assertRaises(ValueError): self.session.start('cat.jpg', 'cat', 0)

    def test_slow_initializer_cannot_restore_stopped_or_switched_target(self):
        for switch in (False, True):
            entered, release = threading.Event(), threading.Event()
            errors = []
            def factory(path, mode):
                entered.set(); release.wait(2); return object()
            session = TrackingSession(self.markers, self.follow, factory)
            session.register_reference('ref.jpg', 'cat', 0)
            def start():
                try: session.start('ref.jpg', 'cat', 0)
                except ValueError as exc: errors.append(str(exc))
            worker = threading.Thread(target=start); worker.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertTrue(session.snapshot()['tracking_loading'])
                with self.assertRaises(ValueError): session.start('ref.jpg', 'cat', 0)
                if switch: session.set_mode('person')
                else: session.stop()
            finally: release.set(); worker.join(2)
            self.assertFalse(session.snapshot()['tracking_active'])
            self.assertEqual(len(errors), 1)

    def test_error_is_reported_and_current_person_can_publish(self):
        self.session.set_mode('person')
        self.session.register_reference('person.jpg', 'person', 1)
        self.session.factory = Mock(side_effect=ValueError('参考照片含多人'))
        with self.assertRaisesRegex(ValueError, '多人'): self.session.start('person.jpg', 'person', 1)
        self.assertFalse(self.session.snapshot()['tracking_loading'])
        self.assertIn('多人', self.session.snapshot()['tracking_error'])
        self.session.factory = lambda p, m: object()
        self.session.start('person.jpg', 'person', 1)
        sample = self.session.capture({})
        self.session.observe(sample, [1, 2, 3])
        self.markers.observe.assert_called_once_with([1, 2, 3], sample[3])


if __name__ == '__main__': unittest.main()
