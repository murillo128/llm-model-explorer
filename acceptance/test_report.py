"""Small independent native artifacts protect timing and incomplete-result reporting."""

import base64
import json
import tempfile
import unittest
from pathlib import Path

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
