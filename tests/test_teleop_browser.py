"""Run keyboard-driving checks in Chrome with a fake rosbridge (no robot traffic).

Run: python3 -m unittest discover -s tests -v
Requires Chrome/Chromium and Jinja2; does not import the camera application.
"""
import functools
import html
import http.server
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest

from jinja2 import Environment, FileSystemLoader


ROOT = Path(__file__).resolve().parents[1]
CHROME = next((shutil.which(name) for name in
               ('google-chrome', 'chromium', 'chromium-browser') if shutil.which(name)), None)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@unittest.skipUnless(CHROME, 'Chrome/Chromium is required')
class TeleopBrowserTest(unittest.TestCase):
    def test_keyboard_driving(self):
        with tempfile.TemporaryDirectory(prefix='cat-teleop-test-') as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'static', root / 'static',
                            ignore=shutil.ignore_patterns('novnc', 'js'))
            template = Environment(loader=FileSystemLoader(ROOT / 'templates')).get_template('index.html')
            page = template.render(
                request=SimpleNamespace(host='127.0.0.1'),
                url_for=lambda endpoint, filename: '/static/' + filename)
            # Prevent all camera, VNC and tracking requests during the test.
            page = re.sub(r'src="http://[^\"]+:6080/[^\"]+"', 'src="about:blank"', page)
            page = page.replace('src="/video_feed"', '')
            page = page.replace('<script src="/static/script.js"></script>', '')
            harness = (ROOT / 'tests' / 'teleop_browser.js').read_text()
            page = page.replace('</head>', '<script>' + harness + '</script></head>')
            (root / 'index.html').write_text(page)
            server = http.server.ThreadingHTTPServer(
                ('127.0.0.1', 0), functools.partial(QuietHandler, directory=directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = subprocess.run([
                    CHROME, '--headless', '--no-sandbox', '--disable-gpu',
                    '--disable-dev-shm-usage', '--no-first-run', '--password-store=basic',
                    '--no-proxy-server', '--disable-background-networking',
                    '--user-data-dir=' + str(root / 'chrome-profile'),
                    '--dump-dom', '--timeout=15000', '--virtual-time-budget=1000',
                    'http://127.0.0.1:' + str(server.server_port)
                ], capture_output=True, text=True, timeout=30)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
            match = re.search(r'<pre id="test-results"[^>]*>(.*?)</pre>', result.stdout, re.S)
            self.assertIsNotNone(match, result.stderr[-3000:] + result.stdout[-3000:])
            report = json.loads(html.unescape(match.group(1)))
            self.assertEqual(report['failures'], [], json.dumps(report, ensure_ascii=False, indent=2))
            self.assertGreaterEqual(len(report['passed']), 20)
            print('\nBrowser checks:', ', '.join(report['passed']))


if __name__ == '__main__':
    unittest.main()
