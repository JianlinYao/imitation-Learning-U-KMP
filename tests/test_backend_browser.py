"""Real browser UI regression using explicitly synthetic process output."""
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import time
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('fastapi', 'uvicorn', 'playwright'))


@unittest.skipUnless(os.environ.get('ROBOT_BROWSER_TESTS') == '1' and AVAILABLE, 'opt-in browser and server dependencies required')
class BackendBrowserTests(unittest.TestCase):
    def test_submit_view_retry_and_mobile_layout(self):
        import uvicorn
        from playwright.sync_api import sync_playwright
        from imitation_learning.server import create_app
        from test_backend import fixture_command
        with tempfile.TemporaryDirectory() as folder:
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            app = create_app(folder, command_factory=fixture_command)
            server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error'))
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            try:
                until = time.monotonic() + 15
                while not server.started and time.monotonic() < until:
                    time.sleep(.05)
                self.assertTrue(server.started)
                with sync_playwright() as runtime:
                    executable = os.environ.get('ROBOT_BROWSER_EXECUTABLE') or shutil.which('google-chrome') or shutil.which('chromium')
                    browser = runtime.chromium.launch(headless=True, args=['--no-sandbox'],
                                                       **({'executable_path': executable} if executable else {}))
                    page = browser.new_page(viewport={'width': 1400, 'height': 1000})
                    page.set_default_timeout(15000)
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(f'http://127.0.0.1:{port}/')
                    page.wait_for_selector('#letters input[value=G]')
                    page.fill('#points', '32')
                    page.click('#submit')
                    page.get_by_text('打开学习报告', exact=True).wait_for()
                    self.assertEqual(app.state.store.list()['total'], 1)
                    self.assertIn('学习记录：1', page.locator('#summary').inner_text())
                    self.assertIn('fixture process done', page.locator('#logs').inner_text())
                    with page.expect_popup() as popup:
                        page.get_by_text('打开学习报告', exact=True).click()
                    report = popup.value
                    report.wait_for_load_state()
                    self.assertIn('Explicit UI test fixture', report.locator('body').inner_text())
                    report.close()
                    page.click('#retry')
                    page.wait_for_function("document.querySelectorAll('#jobs tr[data-id]').length===2")
                    self.assertEqual(app.state.store.list()['total'], 2)
                    page.uncheck('#letters input[value=G]')
                    page.click('#submit')
                    self.assertIn('请至少选择', page.locator('#message').inner_text())
                    self.assertEqual(app.state.store.list()['total'], 2)
                    screenshots = Path(__file__).resolve().parents[1] / 'outputs' / 'browser-tests'
                    screenshots.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(screenshots / 'manager-desktop.png'), full_page=True)
                    page.set_viewport_size({'width': 390, 'height': 844})
                    self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
                    page.screenshot(path=str(screenshots / 'manager-mobile.png'), full_page=True)
                    self.assertEqual(errors, [])
                    browser.close()
            finally:
                server.should_exit = True
                thread.join(15)
                self.assertFalse(thread.is_alive())
