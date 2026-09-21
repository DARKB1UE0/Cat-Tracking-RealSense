"""Action callback ordering checks without ROS or hardware."""
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from follow_navigation import FollowNavigation


class NavigationCallbackTests(unittest.TestCase):
    def test_old_cancel_ack_cannot_clear_new_cancel_request(self):
        nav = FollowNavigation()
        old = Future(); old.set_result(SimpleNamespace(goals_canceling=[]))
        current = Future(); current.set_result(SimpleNamespace(goals_canceling=[object()]))
        nav.cancel_future = current
        nav.error = 'current state'
        nav._canceled(old)
        self.assertIs(nav.cancel_future, current)
        self.assertEqual(nav.error, 'current state')
        nav._canceled(current)
        self.assertIsNone(nav.cancel_future)
