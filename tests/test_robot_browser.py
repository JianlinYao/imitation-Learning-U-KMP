"""Opt-in real Chromium interaction check; uses explicitly synthetic UI fixtures."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from imitation_learning.robot import build_robot_gallery
from robot_fixtures import write_robot_fixture

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("ROBOT_BROWSER_TESTS") == "1", "opt in with ROBOT_BROWSER_TESTS=1 and install Playwright Chromium")
class RobotBrowserTests(unittest.TestCase):
    def test_letter_method_switching_updates_playback_metrics_and_downloads(self):
        from playwright.sync_api import sync_playwright
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_robot_fixture(root / "A" / "KMP", "A", "KMP", .001)
            write_robot_fixture(root / "B" / "KMP", "B", "KMP", .02)
            write_robot_fixture(root / "B" / "GP", "B", "GP", .0005)
            build_robot_gallery(root)
            with sync_playwright() as runtime:
                executable = os.environ.get("ROBOT_BROWSER_EXECUTABLE") or shutil.which("google-chrome") or shutil.which("chromium")
                browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"],
                                                   **({"executable_path": executable} if executable else {}))
                page = browser.new_page(viewport={"width": 1400, "height": 1000})
                page.set_default_timeout(5000)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto((root / "dashboard.html").as_uri())
                self.assertEqual(page.locator("#letter option").all_text_contents(), ["A", "B"])
                self.assertEqual(page.locator("#maximum").inner_text(), "1.000")
                page.click("#play")
                page.wait_for_function("Number(document.getElementById('seek').value)>0")
                page.select_option("#letter", "B")
                self.assertEqual(page.locator("#seek").input_value(), "0")
                self.assertEqual(page.locator("#play").inner_text(), "播放")
                self.assertEqual(page.locator("#status").inner_text(), "仿真未达阈值")
                self.assertEqual(page.locator("#maximum").inner_text(), "20.000")
                self.assertEqual(page.locator("#facts > div").count(), 6)
                self.assertEqual(page.locator("#csv").get_attribute("href"), "B/KMP/robot_trace.csv")
                self.assertIn("failure", page.locator("#note").get_attribute("class"))
                page.select_option("#method", "GP")
                self.assertEqual(page.locator("#status").inner_text(), "仿真跟踪通过")
                self.assertEqual(page.locator("#maximum").inner_text(), "0.500")
                self.assertNotIn("failure", page.locator("#note").get_attribute("class"))
                self.assertEqual(page.locator("#csv").get_attribute("href"), "B/GP/robot_trace.csv")
                self.assertEqual(page.locator("#report").get_attribute("href"), "B/GP/REPORT.md")
                page.select_option("#letter", "A")
                self.assertEqual(page.locator("#method").input_value(), "KMP")
                self.assertEqual(page.locator("#maximum").inner_text(), "1.000")
                self.assertEqual(page.locator("#case-count").inner_text(), "已加载 2 个字母 · 3 个案例")
                screenshots = ROOT / "outputs" / "browser-tests"
                screenshots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshots / "fixture-switching-desktop.png"), full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                page.select_option("#letter", "B")
                self.assertEqual(page.locator("#title").inner_text(), "B · KMP · 机械臂仿真")
                self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"))
                page.screenshot(path=str(screenshots / "fixture-switching-mobile.png"), full_page=True)
                self.assertEqual(errors, [])
                browser.close()
