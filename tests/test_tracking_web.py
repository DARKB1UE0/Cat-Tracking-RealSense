"""Flask mode/upload/start behavior using synthetic images and fake trackers."""
import io
import tempfile
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
import web_app
from tracking_session import TrackingSession


class TrackingWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.markers, self.follow = Mock(), Mock()
        self.session = TrackingSession(self.markers, self.follow, lambda p, m: object())
        self.patches = [patch.object(web_app, 'tracking', self.session),
                        patch.object(web_app, 'camera_active', True),
                        patch.object(web_app.cat_markers, 'start'),
                        patch.object(web_app.gimbal_controller, 'start'),
                        patch.dict(web_app.app.config, UPLOAD_FOLDER=self.temp.name)]
        for p in self.patches: p.start(); self.addCleanup(p.stop)
        self.client = web_app.app.test_client()
        self.image = cv2.imencode('.jpg', np.zeros((100,100,3), np.uint8))[1].tobytes()
    def upload(self, mode, generation):
        return self.client.post('/upload', data={'file':(io.BytesIO(self.image),'reference.jpg'),
                                                 'mode':mode,'generation':str(generation)})
    def test_person_switch_requires_fresh_reference_and_stops_follow(self):
        old = self.upload('cat', 0).get_json()['filepath']
        response = self.client.post('/tracking_mode', json={'mode':'person'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.get_json()['generation'],1)
        self.follow.stop.assert_called()
        self.assertEqual(self.upload('cat',0).status_code,400)
        self.assertEqual(self.client.post('/start_tracking',json={'filepath':old,'mode':'person','generation':1}).status_code,400)
        new = self.upload('person',1).get_json()['filepath']
        self.assertNotEqual(old,new)
        self.assertEqual(self.client.post('/start_tracking',json={'filepath':new,'mode':'person','generation':1}).status_code,200)
        state=self.client.get('/status').get_json()
        self.assertEqual(state['mode'],'person');self.assertTrue(state['tracking_active'])
        self.assertEqual(self.client.post('/stop_tracking').status_code,200)
        self.assertFalse(self.client.get('/status').get_json()['tracking_active'])
        self.assertEqual(self.client.post('/tracking_mode',json={'mode':'cat'}).get_json()['mode'],'cat')
    def test_invalid_mode_bad_upload_and_model_error(self):
        self.assertEqual(self.client.post('/tracking_mode',json={'mode':'dog'}).status_code,400)
        self.assertEqual(self.client.post('/tracking_mode',json=[]).status_code,400)
        self.assertEqual(self.client.post('/upload',data={'file':(io.BytesIO(b'bad'),'x.jpg')}).status_code,400)
        path=self.upload('cat',0).get_json()['filepath']
        self.session.factory=Mock(side_effect=ValueError('请上传单人照片'))
        result=self.client.post('/start_tracking',json={'filepath':path,'mode':'cat','generation':0})
        self.assertEqual(result.status_code,400)
        self.assertIn('单人',result.get_json()['message'])
        self.assertFalse(self.client.get('/status').get_json()['tracking_loading'])

    def test_video_displays_rejected_person_scores(self):
        candidate = dict(box=(20, 40, 80, 160), match_score=.09,
                         appearance_score=.1, color_score=0., match_eligible=False)
        tracker = Mock()
        tracker.detect.return_value = ([candidate], None)
        session = Mock()
        session.capture.return_value = (tracker, 0, 'person', None)
        frames = Mock()
        frames.get_color_frame.return_value.get_data.return_value = np.zeros((240, 640, 3), np.uint8)
        with patch.object(web_app, 'tracking', session), \
             patch.object(web_app, 'pipeline'), \
             patch.object(web_app, 'align') as alignment, \
             patch.object(web_app.cv2, 'putText', wraps=cv2.putText) as draw:
            alignment.process.return_value = frames
            stream = web_app.generate_frames()
            try:
                next(stream)  # Detection runs on every second frame.
                next(stream)
            finally:
                stream.close()
            labels = [call.args[1] for call in draw.call_args_list]
            self.assertIn('Other (0.090)', labels)
            self.assertIn('ReID: 0.100 Color: 0.000 LOW', labels)
            self.assertIn('PERSON TEST: NO TARGET', labels)
            session.observe.assert_called_once_with(session.capture.return_value, None)
