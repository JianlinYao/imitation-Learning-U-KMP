"""Existing logs can be combined without MuJoCo or changes to source files."""
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from imitation_learning.robot import build_robot_gallery
from robot_fixtures import write_robot_fixture


def read_payload(page):
    html = page.read_text(encoding="utf-8")
    return json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))


class RobotGalleryTests(unittest.TestCase):
    def test_merge_preserves_sources_metrics_and_case_specific_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "仿真结果"
            write_robot_fixture(source / "A", "A")
            write_robot_fixture(source / "B", "B", error_m=.02)
            before = {path: sha256(path.read_bytes()).hexdigest() for path in source.rglob("*") if path.is_file()}
            output = root / "浏览页面" / "combined.html"
            with patch("imitation_learning.robot.rollout", side_effect=AssertionError("must reuse logs")):
                result = build_robot_gallery(source, output)
            data = read_payload(output)
            self.assertEqual(result["letters"], ["A", "B"])
            self.assertFalse(result["all_tracking_passed"])
            self.assertEqual([case["manifest"]["source_case"]["letter"] for case in data["cases"]], ["A", "B"])
            self.assertAlmostEqual(data["cases"][1]["manifest"]["metrics"]["max_tracking_error_m"], .02)
            self.assertEqual(data["cases"][1]["artifacts"]["csv"], "../仿真结果/B/robot_trace.csv")
            self.assertTrue((output.parent / data["cases"][1]["artifacts"]["report"]).is_file())
            self.assertEqual(before, {path: sha256(path.read_bytes()).hexdigest() for path in before})

    def test_multiple_methods_remain_distinct_and_can_rebuild_same_gallery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_robot_fixture(root / "G" / "KMP", "G", "KMP")
            write_robot_fixture(root / "G" / "GP", "G", "GP", error_m=.002)
            build_robot_gallery(root)
            first = read_payload(root / "dashboard.html")
            build_robot_gallery(root)
            self.assertEqual(read_payload(root / "dashboard.html"), first)
            self.assertEqual([case["manifest"]["source_case"]["method"] for case in first["cases"]], ["GP", "KMP"])
            self.assertEqual(first["cases"][1]["artifacts"]["csv"], "G/KMP/robot_trace.csv")

    def test_empty_and_duplicate_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                build_robot_gallery(root)
            write_robot_fixture(root / "first")
            write_robot_fixture(root / "second")
            with self.assertRaisesRegex(ValueError, "duplicate robot case"):
                build_robot_gallery(root)
            self.assertFalse((root / "dashboard.html").exists())

    def test_corrupted_trace_does_not_create_a_misleading_page(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_robot_fixture(root / "A")
            trace_path = root / "A" / "robot_trace.npz"
            with np.load(trace_path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            arrays["tip"] += .01
            np.savez_compressed(trace_path, **arrays)
            with self.assertRaisesRegex(ValueError, "error does not match"):
                build_robot_gallery(root)
            self.assertFalse((root / "dashboard.html").exists())
