"""Condense native check outputs; preserve SKIP separately from PASS."""

import base64
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import torch

from acceptance.fixtures import SEED


def summarize_junit(path):
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    return {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def browser_results(path):
    report = json.loads(path.read_text())
    # Native list output is an inventory, not a completed execution report.
    if "duration" not in report.get("stats", {}):
        raise ValueError(f"Missing browser execution statistics: {path}")
    measurements, skips, reference_gl, harness = [], [], [], []
    statuses = Counter()
    families = {}
    attempts = retries = unrun = 0

    def visit(suite):
        nonlocal attempts, retries, unrun
        for spec in suite.get("specs", []):
            family = families.setdefault(spec["file"], {"invocations": 0, "summed_test_ms": 0})
            for test in spec["tests"]:
                family["invocations"] += 1
                results = test["results"]
                unrun += not results
                for annotation in test.get("annotations", []):
                    if annotation["type"] == "skip":
                        skips.append(annotation.get("description", "Skipped"))
                for result in results:
                    attempts += 1
                    retries += result.get("retry", 0) > 0
                    statuses[result["status"]] += 1
                    family["summed_test_ms"] += result["duration"]
                    for attachment in result.get("attachments", []):
                        targets = {
                            "measurements": measurements,
                            "reference-webgl": reference_gl,
                            "harness-timing": harness,
                        }
                        if attachment["name"] in targets:
                            target = targets[attachment["name"]]
                            target.append(json.loads(base64.b64decode(attachment["body"])))
        for child in suite.get("suites", []):
            visit(child)

    for suite in report["suites"]:
        visit(suite)
    return {
        "stats": report["stats"],
        "measurements": measurements,
        "skip_reasons": sorted(set(skips)),
        "reference_webgl": reference_gl,
        "timing": {
            "wall_ms": report["stats"]["duration"],
            "summed_test_ms": sum(f["summed_test_ms"] for f in families.values()),
            "attempts": attempts,
            "retries": retries,
            "status_counts": dict(statuses),
            "unrun": unrun,
            "interrupted": statuses["interrupted"],
            "global_errors": report.get("errors", []),
            "families": families,
            "harness_samples": len(harness),
            # Fixture time can overlap spawn-to-ready: these sums are separate.
            "harness_phases": {
                phase: {
                    "samples": len(values),
                    "summed_ms": sum(values) if values else None,
                }
                for phase in (
                    "fixtureGenerationMs",
                    "backendSpawnToReadyMs",
                    "browserSetupMs",
                    "testBodyMs",
                    "teardownMs",
                )
                for values in [[h[phase] for h in harness if h.get(phase) is not None]]
            },
        },
    }


def main():
    output = Path(sys.argv[1])
    ui = Path(__file__).resolve().parents[1] / "ui"
    unit = json.loads((output / "unit.json").read_text())
    report = {
        "fixture_seed": SEED,
        "cuda": {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": torch.cuda.get_device_capability(0)
            if torch.cuda.is_available()
            else None,
        },
        "reference": {
            "supplied": bool(os.environ.get("LMEX_REFERENCE_MODEL_DIR")),
            "device": os.environ.get("LMEX_REFERENCE_DEVICE", "cpu"),
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "node": subprocess.check_output(["node", "--version"], text=True).strip(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "fastapi",
                "starlette",
                "uvicorn",
                "transformers",
                "tokenizers",
                "pytest",
            )
        },
        "playwright": json.loads((ui / "node_modules/@playwright/test/package.json").read_text())[
            "version"
        ],
        "backend": summarize_junit(output / "backend.xml"),
        "network": summarize_junit(output / "network.xml"),
        "ui_unit": {
            key: unit[key]
            for key in ("numTotalTests", "numPassedTests", "numFailedTests", "numPendingTests")
        },
        "component_browser": browser_results(output / "component-browser.json"),
        "product_browser": browser_results(ui / "test-results/acceptance.json"),
        "network_measurements": {
            p.attrib["name"]: p.attrib["value"]
            for p in ET.parse(output / "network.xml").iter("property")
        },
        "network_skip_reasons": [
            s.attrib.get("message", "") for s in ET.parse(output / "network.xml").iter("skipped")
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Acceptance evidence: {output / 'report.json'}")


if __name__ == "__main__":
    main()
