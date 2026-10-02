"""Small independent native artifacts protect timing and incomplete-result reporting."""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acceptance import report
from acceptance.report import browser_results


class BrowserReportTests(unittest.TestCase):
    def summarize(self, tests, **extra):
        report = {
            "stats": {"duration": 12},
            "suites": [{"suites": [{"specs": [{"file": "sample.spec.ts", "tests": tests}]}]}],
            **extra,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            return browser_results(path)

    def test_pass_failure_skip_and_retry_are_separate_from_wall_time(self):
        result = self.summarize(
            [
                {"results": [{"status": "passed", "duration": 10, "retry": 0}]},
                {
                    "results": [
                        {"status": "failed", "duration": 8, "retry": 0},
                        {"status": "passed", "duration": 9, "retry": 1},
                    ]
                },
                {
                    "annotations": [{"type": "skip", "description": "reference absent"}],
                    "results": [{"status": "skipped", "duration": 0, "retry": 0}],
                },
            ]
        )
        timing = result["timing"]
        self.assertEqual(timing["wall_ms"], 12)
        self.assertEqual(timing["summed_test_ms"], 27)
        self.assertEqual(timing["attempts"], 4)
        self.assertEqual(timing["retries"], 1)
        self.assertEqual(timing["status_counts"], {"passed": 2, "failed": 1, "skipped": 1})
        self.assertEqual(
            timing["families"],
            {
                "sample.spec.ts": {"invocations": 3, "summed_test_ms": 27},
            },
        )
        self.assertEqual(result["skip_reasons"], ["reference absent"])

    def test_harness_phases_preserve_overlap_and_absent_samples(self):
        body = base64.b64encode(
            json.dumps(
                {
                    "fixtureGenerationMs": 3,
                    "backendSpawnToReadyMs": 7,
                    "browserSetupMs": 2,
                    "testBodyMs": 4,
                    "teardownMs": None,
                }
            ).encode()
        ).decode()
        result = self.summarize(
            [
                {
                    "results": [
                        {
                            "status": "passed",
                            "duration": 15,
                            "attachments": [{"name": "harness-timing", "body": body}],
                        }
                    ]
                }
            ]
        )["timing"]
        self.assertEqual(result["harness_samples"], 1)
        self.assertEqual(
            result["harness_phases"]["fixtureGenerationMs"], {"samples": 1, "summed_ms": 3}
        )
        self.assertEqual(
            result["harness_phases"]["backendSpawnToReadyMs"], {"samples": 1, "summed_ms": 7}
        )
        self.assertEqual(result["harness_phases"]["teardownMs"], {"samples": 0, "summed_ms": None})

    def test_interrupted_unrun_and_global_error_remain_visible(self):
        timing = self.summarize(
            [
                {"results": [{"status": "interrupted", "duration": 5}]},
                {"results": []},
            ],
            errors=[{"message": "worker stopped"}],
        )["timing"]
        self.assertEqual(timing["unrun"], 1)
        self.assertEqual(timing["interrupted"], 1)
        self.assertEqual(timing["global_errors"], [{"message": "worker stopped"}])
        self.assertEqual(timing["status_counts"], {"interrupted": 1})
        self.assertEqual(timing["harness_phases"]["testBodyMs"], {"samples": 0, "summed_ms": None})

    def test_missing_statistics_are_not_an_empty_success(self):
        with self.assertRaisesRegex(ValueError, "Missing browser execution statistics"):
            self.summarize([], stats={})

    def test_missing_artifact_is_not_an_empty_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                browser_results(Path(directory) / "absent.json")


class FullReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lmex-full-report-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Keep evidence outside the checkout to exercise the supplied root.
        self.output = self.root / "evidence"
        self.ui = self.root / "checkout/ui"
        self.current = self.output / "browser/full/report.json"
        self.legacy = self.ui / "test-results/acceptance.json"
        self.write_json(
            self.output / "unit.json",
            {"numTotalTests": 1, "numPassedTests": 1, "numFailedTests": 0, "numPendingTests": 0},
        )
        for name in ("backend.xml", "network.xml"):
            (self.output / name).write_text(
                '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/>'
                "</testsuites>"
            )
        self.write_json(self.ui / "node_modules/@playwright/test/package.json", {"version": "test"})
        self.write_browser(self.output / "component-browser.json", 12)
        self.write_browser(self.current, 123)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def write_browser(self, path, duration, status="passed"):
        self.write_json(
            path,
            {
                "stats": {"duration": duration},
                "suites": [
                    {
                        "specs": [
                            {
                                "file": "sample.spec.ts",
                                "tests": [{"results": [{"status": status, "duration": duration}]}],
                            }
                        ]
                    }
                ],
            },
        )

    def run_report(self):
        # Only checkout location and unrelated environment probes are replaced.
        # main(), path selection, file reads, parsers and report writes stay real.
        with (
            patch.object(report, "__file__", str(self.root / "checkout/acceptance/report.py")),
            patch.object(report.sys, "argv", ["acceptance.report", str(self.output)]),
            patch.object(report.subprocess, "check_output", return_value="test-node\n"),
            patch.object(report.importlib.metadata, "version", return_value="test"),
            patch.object(report.torch.cuda, "is_available", return_value=False),
        ):
            report.main()
        return json.loads((self.output / "report.json").read_text())

    def test_main_reads_current_full_report_from_supplied_root(self):
        result = self.run_report()
        self.assertEqual(result["product_browser"]["timing"]["wall_ms"], 123)
        self.assertEqual(result["product_browser"]["timing"]["status_counts"], {"passed": 1})
        self.assertEqual(result["component_browser"]["timing"]["wall_ms"], 12)
        self.assertFalse(self.legacy.exists())

    def test_main_ignores_stale_legacy_browser_report(self):
        self.write_browser(self.legacy, 999, "failed")
        legacy_bytes = self.legacy.read_bytes()
        result = self.run_report()
        self.assertEqual(result["product_browser"]["timing"]["wall_ms"], 123)
        self.assertEqual(result["product_browser"]["timing"]["status_counts"], {"passed": 1})
        self.assertEqual(self.legacy.read_bytes(), legacy_bytes)

    def test_main_requires_current_full_report_even_with_legacy_present(self):
        self.current.unlink()
        self.write_browser(self.legacy, 999)
        with self.assertRaises(FileNotFoundError) as caught:
            self.run_report()
        self.assertEqual(Path(caught.exception.filename), self.current)
        self.assertFalse((self.output / "report.json").exists())
