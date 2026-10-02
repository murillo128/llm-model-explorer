"""Exercise CI command selection without starting the application or browsers."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

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
        for directory in (
            "acceptance",
            "ui/tests",
            "ui/acceptance",
            "backend/.venv/bin",
            "bin",
            ".github/scripts",
        ):
            (self.root / directory).mkdir(parents=True)
        shutil.copyfile(
            ROOT / "acceptance/check-integration.sh",
            self.root / "acceptance/check-integration.sh",
        )
        shutil.copyfile(ROOT / "acceptance/check.sh", self.root / "acceptance/check.sh")
        shutil.copyfile(
            ROOT / ".github/scripts/validation_selector.py",
            self.root / ".github/scripts/validation_selector.py",
        )
        for directory in ("ui/tests", "ui/acceptance"):
            for path in (ROOT / directory).glob("*.spec.ts"):
                (self.root / directory / path.name).touch()
        for name in ("ruff", "mypy", "pytest"):
            self.stub(self.root / "backend/.venv/bin" / name)
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

    def plan(self, *paths):
        changed = self.root / "paths.json"
        changed.write_text(json.dumps(paths))
        destination = self.root / "validation-plan.json"
        subprocess.run(
            [
                sys.executable,
                str(self.root / ".github/scripts/validation_selector.py"),
                "--paths",
                str(changed),
                "--output",
                str(destination),
            ],
            check=True,
            capture_output=True,
        )
        return destination

    def stub(self, path):
        path.write_text(f"#!{sys.executable}\n{STUB}")
        path.chmod(0o755)

    def run_gate(self, *args, fail=None, script="check-integration.sh"):
        env = dict(self.env)
        if fail is not None:
            env["CI_FAIL_COMMAND"] = fail
        result = subprocess.run(
            ["bash", str(self.root / "acceptance" / script), *args],
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
                [
                    "ruff",
                    "check",
                    "--config",
                    "backend/pyproject.toml",
                    "acceptance",
                    ".github/scripts/validation_selector.py",
                ],
                [
                    "ruff",
                    "format",
                    "--check",
                    "--config",
                    "backend/pyproject.toml",
                    "acceptance",
                    ".github/scripts/validation_selector.py",
                ],
                [
                    "python",
                    "-m",
                    "pytest",
                    "acceptance",
                    "-ra",
                    "--durations=25",
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
            commands[:1],
            [
                ["contract-python", "api/validate_contract.py"],
            ],
        )
        self.assertEqual(
            [command for command in commands if command[0] == "npm"],
            [
                ["npm", "run", "api:check"],
                ["npm", "run", "build"],
                ["npm", "run", "test:acceptance", "--", "--project=dpr1"],
            ],
        )

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

    def test_main_propagates_timed_http_failure(self):
        command = (
            "python -m pytest acceptance -ra --durations=25 -o junit_family=legacy "
            f"--junitxml={self.root / 'evidence/network.xml'}"
        )
        result, commands = self.run_gate("--main-ci", fail=command)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(" ".join(commands[-1]), command)

    def test_local_keeps_all_layers_and_timing_without_changing_projects(self):
        result, commands = self.run_gate(script="check.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(["mypy"], commands)
        self.assertIn(
            [
                "pytest",
                "--durations=25",
                "-o",
                "junit_family=legacy",
                f"--junitxml={self.root / 'evidence/backend.xml'}",
            ],
            commands,
        )
        self.assertIn(["npm", "run", "test:browser", "--", "--reporter=list,json"], commands)
        self.assertIn(["npm", "run", "test:acceptance"], commands)
        self.assertEqual(
            commands[-1], ["python", "-m", "acceptance.report", str(self.root / "evidence")]
        )

    def test_local_propagates_timed_backend_failure(self):
        command = (
            "pytest --durations=25 -o junit_family=legacy "
            f"--junitxml={self.root / 'evidence/backend.xml'}"
        )
        result, commands = self.run_gate(script="check.sh", fail=command)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(" ".join(commands[-1]), command)

    def test_tokenizer_plan_executes_real_selected_network_and_browser_commands(self):
        plan = self.plan("ui/src/tokenizer/editor.ts")
        result, commands = self.run_gate("--main-ci", "--plan", str(plan))
        self.assertEqual(result.returncode, 0, result.stderr)
        pytest = next(command for command in commands if command[:3] == ["python", "-m", "pytest"])
        self.assertEqual(
            pytest[3:8],
            [
                "acceptance/test_distribution_scales.py",
                "acceptance/test_embeddings.py",
                "acceptance/test_network.py",
                "acceptance/test_polish.py",
                "acceptance/test_reference.py",
            ],
        )
        self.assertIn(
            [
                "npm",
                "run",
                "test:acceptance",
                "--",
                r"bindings\.spec\.ts",
                r"scientific\.spec\.ts",
                r"tokenizer\-layout\.spec\.ts",
                r"transport\.spec\.ts",
            ],
            commands,
        )
        self.assertIn(["npm", "run", "build"], commands)

    def test_http_only_route_executes_pytest_without_build_or_browser(self):
        (self.root / "acceptance/test_report.py").touch()
        plan = self.plan("acceptance/test_report.py")
        result, commands = self.run_gate("--routine", "--main-ci", "--plan", str(plan))
        self.assertEqual(result.returncode, 0, result.stderr)
        selected = next(
            command for command in commands if command[:3] == ["python", "-m", "pytest"]
        )
        self.assertEqual(selected[3], "acceptance/test_report.py")
        self.assertFalse(any(command[0] in {"npm", "xvfb-run"} for command in commands))
        self.log.unlink()
        result, commands = self.run_gate(
            "--routine", "--main-ci", "--plan", str(plan), fail=" ".join(selected)
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(commands[-1], selected)

    def test_shared_api_plan_preserves_complete_acceptance(self):
        plan = self.plan("docs/spec/api/contract.md")
        result, commands = self.run_gate("--plan", str(plan))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(
            any(command[:4] == ["python", "-m", "pytest", "acceptance"] for command in commands)
        )
        browser = next(
            command for command in commands if command[:3] == ["npm", "run", "test:acceptance"]
        )
        self.assertEqual(
            browser[4:],
            [
                r"architecture\.spec\.ts",
                r"bindings\.spec\.ts",
                r"lora\-reference\.spec\.ts",
                r"native\.spec\.ts",
                r"scientific\.spec\.ts",
                r"tokenizer\-layout\.spec\.ts",
                r"transport\.spec\.ts",
                "--project=dpr1",
            ],
        )

    def test_shared_http_helper_runs_consumers_without_browser_setup(self):
        for name in ("network", "architecture", "native_packages", "embeddings", "lora_reference"):
            (self.root / f"acceptance/test_{name}.py").touch()
        plan = self.plan("acceptance/test_network.py")
        result, commands = self.run_gate("--routine", "--main-ci", "--plan", str(plan))
        self.assertEqual(result.returncode, 0, result.stderr)
        selected = next(
            command for command in commands if command[:3] == ["python", "-m", "pytest"]
        )
        self.assertEqual(
            selected[3:8],
            [
                "acceptance/test_architecture.py",
                "acceptance/test_embeddings.py",
                "acceptance/test_lora_reference.py",
                "acceptance/test_native_packages.py",
                "acceptance/test_network.py",
            ],
        )
        self.assertFalse(any(command[0] in {"npm", "xvfb-run"} for command in commands))

    def test_lora_route_runs_reference_tcp_owner_and_propagates_its_failure(self):
        plan = self.plan("ui/acceptance/lora-reference.spec.ts")
        result, commands = self.run_gate("--routine", "--main-ci", "--plan", str(plan))
        self.assertEqual(result.returncode, 0, result.stderr)
        selected = next(
            command for command in commands if command[:3] == ["python", "-m", "pytest"]
        )
        self.assertEqual(
            selected[3:6],
            [
                "acceptance/test_architecture.py",
                "acceptance/test_lora_reference.py",
                "acceptance/test_reference.py",
            ],
        )
        self.log.unlink()
        result, commands = self.run_gate(
            "--routine", "--main-ci", "--plan", str(plan), fail=" ".join(selected)
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(commands[-1], selected)
        self.assertFalse(any(command[0] in {"npm", "xvfb-run"} for command in commands))

    def test_selected_build_pytest_and_browser_failures_propagate(self):
        plan = self.plan("ui/src/tokenizer/editor.ts")
        result, commands = self.run_gate("--main-ci", "--plan", str(plan))
        self.assertEqual(result.returncode, 0, result.stderr)
        failing = [
            command
            for command in commands
            if command[:3]
            in (
                ["python", "-m", "pytest"],
                ["npm", "run", "build"],
                ["npm", "run", "test:acceptance"],
            )
        ]
        for command in failing:
            with self.subTest(command=command):
                self.log.unlink()
                result, commands = self.run_gate(
                    "--main-ci", "--plan", str(plan), fail=" ".join(command)
                )
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual(commands[-1], command)

    def test_empty_required_plan_and_missing_plan_fail_before_checks(self):
        plan = self.plan("ui/src/tokenizer/editor.ts")
        data = json.loads(plan.read_text())
        data["integration"] = []
        plan.write_text(json.dumps(data))
        for path in (plan, self.root / "missing.json"):
            with self.subTest(path=path):
                result, commands = self.run_gate("--plan", str(path))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(commands, [])

    def workflow(self, name):
        # Parse actual invocation/security/event structure rather than matching
        # source strings in place of command execution.
        source = (ROOT / f".github/workflows/{name}.yml").read_text()

        def unique_keys(node):
            if isinstance(node, yaml.MappingNode):
                keys = [key.value for key, _ in node.value]
                self.assertEqual(len(keys), len(set(keys)), f"Duplicate YAML key in {name}")
                for _, child in node.value:
                    unique_keys(child)
            elif isinstance(node, yaml.SequenceNode):
                for child in node.value:
                    unique_keys(child)

        unique_keys(yaml.compose(source, Loader=yaml.BaseLoader))
        return yaml.load(source, Loader=yaml.BaseLoader)

    def test_workflow_selection_invocation_outputs_and_trigger_contract(self):
        for name in ("backend-ci", "api-contract", "ui-ci", "application-acceptance"):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                for event in ("push", "pull_request", "workflow_dispatch"):
                    self.assertIn(event, workflow["on"])
                    self.assertNotIn("paths", workflow["on"][event] or {})
                self.assertEqual(workflow["permissions"], {"contents": "read"})
                job = next(iter(workflow["jobs"].values()))
                self.assertIn("head.repo.full_name == github.repository", job["if"])
                steps = job["steps"]
                checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v4")
                self.assertEqual(checkout["with"]["fetch-depth"], "0")
                self.assertEqual(checkout["with"]["persist-credentials"], "false")
                invocation = next(step for step in steps if step.get("id") == "plan")
                env = self.env | {
                    "GITHUB_OUTPUT": str(self.root / "outputs"),
                    "GITHUB_STEP_SUMMARY": str(self.root / "summary"),
                }
                result = subprocess.run(
                    ["bash", "-euc", invocation["run"]],
                    cwd=self.root,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("integration=true", (self.root / "outputs").read_text())
                self.assertIn("full fallback", (self.root / "summary").read_text())
                plan = json.loads((self.root / "validation-plan.json").read_text())
                self.assertEqual(plan["browser"], ["full"])

    def test_ui_workflow_executes_domain_union_and_failure(self):
        self.plan("ui/src/tokenizer/editor.ts", "ui/src/architecture-explorer/Graph.tsx")
        job = self.workflow("ui-ci")["jobs"]["ui"]
        invocation = next(
            step["run"] for step in job["steps"] if step.get("name") == "Run browser tests"
        )
        env = self.env | {"GITHUB_REF": "refs/heads/feature"}
        result = subprocess.run(
            ["bash", "-euc", invocation],
            cwd=self.root / "ui",
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [json.loads(line) for line in self.log.read_text().splitlines()]
        browser = next(
            command for command in commands if command[:3] == ["npm", "run", "test:browser"]
        )
        self.assertIn(r"tokenizer\-embeddings\.spec\.ts", browser)
        self.assertIn(r"architecture\-camera\.spec\.ts", browser)
        self.assertNotIn(r"matrix\-explorer\.spec\.ts", browser)
        self.assertEqual(browser[-2:], ["--project=desktop", "--project=native-scrollbars"])
        env["CI_FAIL_COMMAND"] = " ".join(browser)
        result = subprocess.run(
            ["bash", "-euc", invocation],
            cwd=self.root / "ui",
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_application_workflow_invokes_selected_main_gate(self):
        self.plan("ui/src/tokenizer/editor.ts")
        job = self.workflow("application-acceptance")["jobs"]["application"]
        invocation = next(
            step["run"] for step in job["steps"] if step.get("name") == "Run application acceptance"
        )
        result = subprocess.run(
            ["bash", "-euc", invocation],
            cwd=self.root,
            env=self.env | {"GITHUB_REF": "refs/heads/main"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertNotIn(["contract-python", "api/validate_contract.py"], commands)
        self.assertIn(
            [
                "npm",
                "run",
                "test:acceptance",
                "--",
                r"bindings\.spec\.ts",
                r"scientific\.spec\.ts",
                r"tokenizer\-layout\.spec\.ts",
                r"transport\.spec\.ts",
            ],
            commands,
        )
