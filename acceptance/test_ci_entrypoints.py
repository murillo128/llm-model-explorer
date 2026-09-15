"""Exercise CI command selection without starting the application or browsers."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB = """import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['CI_COMMAND_LOG'], 'a') as log:
    log.write(json.dumps([name, *args]) + '\\n')
if ' '.join([name, *args]) == os.environ.get('CI_FAIL_COMMAND'):
    sys.exit(7)
if name == 'xvfb-run':
    os.execvp(args[1], args[1:])
"""


class CiEntrypointsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lmex-ci-entrypoints-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory in ("acceptance", "ui", "backend/.venv/bin", "bin"):
            (self.root / directory).mkdir(parents=True)
        shutil.copyfile(
            ROOT / "acceptance/check-integration.sh",
            self.root / "acceptance/check-integration.sh",
        )
        # Deliberately do not create api/.venv: main must not need API tooling.
        for name in ("backend/.venv/bin/python", "backend/.venv/bin/ruff"):
            self.stub(self.root / name)
        for name in ("npm", "git", "xvfb-run", "contract-python"):
            self.stub(self.root / "bin" / name)
        self.log = self.root / "commands.jsonl"
        self.env = {
            **os.environ,
            "PATH": f"{self.root / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "CI_COMMAND_LOG": str(self.log),
            "LMEX_EVIDENCE_DIR": str(self.root / "evidence"),
            "LMEX_CONTRACT_PYTHON": str(self.root / "bin/contract-python"),
        }
        self.env.pop("CI_FAIL_COMMAND", None)

    def stub(self, path):
        path.write_text(f"#!{sys.executable}\n{STUB}")
        path.chmod(0o755)

    def run_gate(self, *args, fail=None):
        env = dict(self.env)
        if fail is not None:
            env["CI_FAIL_COMMAND"] = fail
        result = subprocess.run(
            ["bash", str(self.root / "acceptance/check-integration.sh"), *args],
            # The gate must resolve its checkout independently of caller cwd.
            cwd=self.root / "ui",
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        commands = (
            [json.loads(line) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )
        return result, commands

    def test_main_runs_only_integration_with_all_dprs(self):
        # A stale override must not make main invoke the independent API tools.
        self.env["LMEX_CONTRACT_PYTHON"] = "/missing/contract-python"
        result, commands = self.run_gate("--main-ci")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            commands,
            [
                ["ruff", "check", "--config", "backend/pyproject.toml", "acceptance"],
                [
                    "ruff",
                    "format",
                    "--check",
                    "--config",
                    "backend/pyproject.toml",
                    "acceptance",
                ],
                [
                    "python",
                    "-m",
                    "pytest",
                    "acceptance",
                    "-ra",
                    "-o",
                    "junit_family=legacy",
                    f"--junitxml={self.root / 'evidence/network.xml'}",
                ],
                ["npm", "run", "build"],
                ["xvfb-run", "-a", "npm", "run", "test:acceptance"],
                ["npm", "run", "test:acceptance"],
            ],
        )

    def test_default_preserves_pr_and_epic_checks(self):
        result, commands = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            commands[:3],
            [
                ["contract-python", "api/validate_contract.py"],
                ["contract-python", "api/validate_contract.py", "--write"],
                ["git", "diff", "--exit-code", "--", "api/fixtures"],
            ],
        )
        self.assertEqual(
            [command for command in commands if command[0] == "npm"],
            [
                ["npm", "run", "api:check"],
                ["npm", "run", "api:generate"],
                ["npm", "run", "build"],
                ["npm", "run", "test:acceptance", "--", "--project=dpr1"],
            ],
        )
        self.assertIn(["git", "diff", "--exit-code", "--", "src/api/generated"], commands)

    def test_main_propagates_build_failure(self):
        result, commands = self.run_gate("--main-ci", fail="npm run build")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(commands[-1], ["npm", "run", "build"])

    def test_main_propagates_acceptance_failure(self):
        result, _ = self.run_gate("--main-ci", fail="npm run test:acceptance")
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_unknown_option_is_rejected_before_running_checks(self):
        result, commands = self.run_gate("--typo")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)
        self.assertEqual(commands, [])

    def test_extra_option_is_rejected_before_running_checks(self):
        result, commands = self.run_gate("--main-ci", "--typo")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(commands, [])
