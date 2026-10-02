"""Conservative validation owners and exact Git diff selection (stdlib only)."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
OWNERS = {"backend", "api", "ui", "browser", "integration"}


def normalized(path):
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError(f"Invalid changed path: {path!r}")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != path:
        raise ValueError(f"Non-normalized changed path: {path!r}")
    return path


def select(paths, *, full=False):
    paths = sorted({normalized(path) for path in paths})
    owners, browser, integration, reasons = set(), set(), set(), []
    backend_tests, extended = set(), set()
    compatibility_full = full

    def add(path, why, checks=(), browsers=(), integrations=()):
        owners.update(checks)
        browser.update(browsers)
        integration.update(integrations)
        if "backend" in checks:
            backend_tests.add("tests")
        if integrations:
            extended.add("integration")
        reasons.append(f"{path}: {why}")

    def broad(path, why):
        nonlocal compatibility_full
        compatibility_full = True
        extended.add("backend")
        add(path, why, OWNERS, ("full",), ("full",))

    if full or not paths:
        broad("context", "explicit full mode" if full else "empty diff; full fallback")
    for path in paths:
        if (
            path.startswith("backend/tests/test_")
            and Path(path).parent.as_posix() == "backend/tests"
            and path.endswith(".py")
            and (ROOT / path).is_file()
        ):
            owners.add("backend")
            backend_tests.add(path.removeprefix("backend/"))
            if path == "backend/tests/test_tensor_analysis.py":
                extended.add("backend")
            reasons.append(f"{path}: isolated backend test; its native file owns validation")
        elif path in {
            "ui/acceptance/scientific.spec.ts",
            "ui/acceptance/transport.spec.ts",
            "ui/acceptance/native.spec.ts",
            "ui/acceptance/bindings.spec.ts",
            "ui/acceptance/tokenizer-layout.spec.ts",
            "ui/acceptance/architecture.spec.ts",
            "ui/acceptance/lora-reference.spec.ts",
        }:
            group = Path(path).name.removesuffix(".spec.ts")
            add(
                path,
                "changed integration responsibility and its real HTTP owner",
                ("ui",),
                integrations=(group,),
            )
        elif path == "ui/acceptance/product-harness.ts":
            add(
                path, "per-test service/probe harness consumers", ("ui",), integrations=("product",)
            )
        elif (
            path.startswith("ui/tests/")
            and Path(path).parent.as_posix() == "ui/tests"
            and path.endswith(".spec.ts")
            and (ROOT / path).is_file()
            and Path(path).name.startswith(
                (
                    "architecture",
                    "matrix",
                    "tensor",
                    "tokenizer",
                    "renderer",
                    "distribution",
                    "magnifier",
                )
            )
        ):
            add(
                path,
                "changed native browser file",
                ("ui",),
                ("file:" + path.removeprefix("ui/tests/"),),
            )
        elif path.startswith("ui/src/") and re.search(r"\.test\.[tj]sx?$", path):
            add(path, "UI unit test uses the native unit/type/lint owner", ("ui",))
        elif (
            path.startswith("acceptance/test_") and path.endswith(".py") and (ROOT / path).is_file()
        ):
            add(
                path,
                "changed HTTP test file and its boundary bridge",
                integrations=("network:" + Path(path).name,),
            )
        elif path.startswith(
            ("api/", "docs/spec/api/", "ui/src/api/", "ui/scripts/api-generator/")
        ):
            broad(path, "API schema, framing or generated binding consumers")
        elif path.startswith("docs/spec/"):
            # Normative docs are product inputs, including new specifications.
            broad(path, "normative product specification and its consumers")
        elif (
            path in ("AGENTS.md", "README.md")
            or path.endswith(".md")
            and path.startswith(
                ("docs/", "skills/", "wiki/", "acceptance/", "ui/", "backend/", "examples/")
            )
        ):
            add(path, "operational documentation; existing documentation/infrastructure owners")
        elif path.startswith("backend/tests/"):
            broad(path, "shared, deleted or unknown backend test support")
        elif path.startswith(("backend/", "examples/")):
            add(
                path,
                "backend/native admission/export and real product consumers",
                ("backend", "integration"),
                integrations=("full",),
            )
            # Model/storage/session and other backend dependencies may affect
            # threshold values or the real allocator stream. Be conservative
            # for production/support inputs; isolated tests were handled above.
            if path.startswith("backend/"):
                extended.add("backend")
            if path.startswith("examples/"):
                add(
                    path,
                    "exported bindings consumed by graph and weight modal",
                    ("ui", "browser"),
                    ("architecture",),
                )
        elif path.startswith("ui/src/architecture-explorer/"):
            add(
                path,
                "graph, responsive controls and real native weight modal",
                ("ui", "browser", "integration"),
                ("architecture",),
                ("architecture",),
            )
        elif path.startswith(("ui/src/rendering/", "ui/src/matrix-explorer/")):
            add(
                path,
                "matrix/DPR/native scroll and tensor/embedding/weight-modal consumers",
                ("ui", "browser", "integration"),
                ("matrix", "tokenizer", "architecture"),
                ("full",),
            )
        elif path.startswith("ui/src/tokenizer/"):
            add(
                path,
                "editor, annotations and real tokenization/embedding streams",
                ("ui", "browser", "integration"),
                ("tokenizer",),
                ("transport", "scientific", "tokenizer-layout", "bindings"),
            )
        elif path.startswith(("ui/", "acceptance/", ".github/workflows/")) or path in (
            ".github/scripts/validation_selector.py",
            ".github/scripts/test_validation_selector.py",
        ):
            broad(path, "shared composition, tests, fixtures, configuration or routing")
        elif path == f".github/scripts/{Path(path).name}" and Path(path).name in {
            "prepare_pr_audit.py",
            "devin_runner.py",
            "test_epic_scheduler.py",
            "cleanup_issue_worktrees.py",
            "executor_control.py",
            "test_worktree_cleanup.py",
            "test_codex_profile.py",
            "local_issue_worktree.py",
            "test_executor_routing.py",
            "devin_host_broker.py",
            "devin_host_client.py",
            "codex_profile.py",
            "find_epic_parent.py",
        }:
            # Runner-only infrastructure has its existing dedicated workflow.
            add(path, "runner infrastructure; existing executor-routing checks")
        else:
            broad(path, "unknown surface; conservative full product fallback")
    if any(
        path
        in {
            "backend/uv.lock",
            "backend/pyproject.toml",
            *{
                f"backend/src/llm_model_explorer/{name}.py"
                for name in (
                    "tensor_analysis",
                    "materialization",
                    "tensor_source",
                    "operations",
                    "artifacts",
                    "services",
                    "settings",
                    "app",
                )
            },
        }
        for path in paths
    ):
        extended.add("backend")
    if any(
        path in {"backend/uv.lock", "backend/pyproject.toml"}
        or path.startswith(
            (
                "backend/src/llm_model_explorer/runtime",
                "backend/src/llm_model_explorer/cli",
                "backend/src/llm_model_explorer/app",
            )
        )
        for path in paths
    ):
        compatibility_full = True
    if browser:
        owners.add("browser")
    if integration:
        owners.add("integration")
    return {
        "portfolio": "full" if full else "routine",
        "backend_tests": sorted(backend_tests),
        "extended": sorted(extended),
        "compatibility_full": compatibility_full,
        "paths": paths,
        "owners": sorted(owners),
        "browser": sorted(browser),
        "integration": sorted(integration),
        "reasons": reasons,
    }


def git(*args, root=ROOT):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)


def diff_paths(base, head, *, event, root=ROOT):
    for revision in (base, head):
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or revision == "0" * 40:
            raise ValueError("Missing or invalid full diff revision")
        git("cat-file", "-e", f"{revision}^{{commit}}", root=root)
    merge_base = git("merge-base", base, head, root=root).decode().strip()
    if event == "push" and merge_base != base:
        raise ValueError("Non-forward push; use full validation")
    if event not in ("push", "pull_request"):
        raise ValueError("Unknown diff event")
    diff_base = merge_base if event == "pull_request" else base
    raw = git("diff", "--name-status", "-z", "--find-renames", diff_base, head, "--", root=root)
    if raw and not raw.endswith(b"\0"):
        raise ValueError("Truncated Git diff")
    tokens = raw.split(b"\0")[:-1]
    paths = set()
    while tokens:
        status = tokens.pop(0).decode("ascii")
        if not re.fullmatch(r"[ACDMRTUXB](?:[0-9]+)?", status):
            raise ValueError(f"Invalid Git status: {status}")
        count = 2 if status.startswith(("R", "C")) else 1
        if len(tokens) < count:
            raise ValueError("Truncated Git path record")
        for _ in range(count):
            paths.add(normalized(tokens.pop(0).decode("utf-8")))
    return sorted(paths), diff_base


def github_plan(event, payload, *, root=ROOT):
    context = {
        "event": event,
        "tested_revision": git("rev-parse", "HEAD", root=root).decode().strip(),
    }
    try:
        if event == "workflow_dispatch":
            plan = select([], full=True)
        elif event == "pull_request":
            pr = payload["pull_request"]
            head = pr["head"]["sha"]
            base_ref = pr["base"]["ref"]
            # Use the current fetched base, not a title, file API or single commit.
            base = git("rev-parse", f"refs/remotes/origin/{base_ref}", root=root).decode().strip()
            context.update(head=head, base=base)
            paths, diff_base = diff_paths(base, head, event=event, root=root)
            context["diff_base"] = diff_base
            parents = git("show", "-s", "--format=%P", "HEAD", root=root).decode().split()
            if context["tested_revision"] != head and parents != [base, head]:
                raise ValueError("Tested merge does not match the current head/base pair")
            plan = select(paths)
            if pr["head"]["ref"].startswith("codex/epic-issue-"):
                plan = select([]) | {
                    "paths": paths,
                    "reasons": [
                        "aggregate epic: mandatory routine portfolio and affected extended owners"
                    ],
                }
        elif event == "push":
            base, head = payload["before"], payload["after"]
            context.update(head=head, base=base)
            if context["tested_revision"] != head:
                raise ValueError("Checkout does not match pushed head")
            paths, diff_base = diff_paths(base, head, event=event, root=root)
            context["diff_base"] = diff_base
            plan = select(paths)
        else:
            raise ValueError("Unsupported event")
    except (KeyError, TypeError, ValueError, UnicodeError, subprocess.CalledProcessError) as error:
        plan = select([])
        plan["reasons"] = [f"Invalid/unavailable diff; full fallback: {error}"]
    return plan | {"context": context}


def targets(plan, gate, *, root=ROOT):
    """Resolve existing native files. An empty required group is an error."""
    groups = plan[gate]
    if gate == "browser":
        patterns = {
            "full": ("**/*.spec.ts",),
            "architecture": ("architecture*.spec.ts",),
            "matrix": (
                "matrix*.spec.ts",
                "tensor*.spec.ts",
                "renderer.spec.ts",
                "distribution-scale.spec.ts",
                "magnifier-adaptation.spec.ts",
            ),
            "tokenizer": ("tokenizer*.spec.ts",),
        }
        directory = root / "ui/tests"
    elif gate == "integration":
        patterns = {
            "full": ("**/*.spec.ts",),
            "product": (
                "transport.spec.ts",
                "scientific.spec.ts",
                "native.spec.ts",
                "bindings.spec.ts",
                "tokenizer-layout.spec.ts",
            ),
            **{
                name: (f"{name}.spec.ts",)
                for name in (
                    "transport",
                    "scientific",
                    "native",
                    "bindings",
                    "tokenizer-layout",
                    "architecture",
                    "lora-reference",
                )
            },
        }
        directory = root / "ui/acceptance"
    else:
        raise ValueError(f"Unknown gate: {gate}")
    files = set()
    for group in groups:
        if gate == "browser" and group.startswith("file:"):
            patterns[group] = (normalized(group.removeprefix("file:")),)
        if gate == "integration" and group.startswith("network:"):
            # A changed HTTP test has no browser consumer. Its nonempty native
            # pytest selection is validated independently below.
            continue
        if group not in patterns:
            raise ValueError(f"Unknown required group: {group}")
        matched = {
            file.relative_to(directory).as_posix()
            for pattern in patterns[group]
            for file in directory.glob(pattern)
        }
        if not matched:
            raise ValueError(f"Empty required {gate} group: {group}")
        files.update(matched)
    network_only = (
        gate == "integration"
        and bool(groups)
        and all(group.startswith("network:") for group in groups)
    )
    if gate in plan["owners"] and not files and not network_only:
        raise ValueError(f"Empty required {gate} selection")
    return sorted(files)


def network_targets(plan):
    groups = plan["integration"]
    if "full" in groups:
        return ["acceptance"]
    owners = {
        "transport": ("network", "embeddings"),
        "scientific": ("network", "embeddings", "distribution_scales"),
        "native": ("network", "polish"),
        "tokenizer-layout": ("network", "embeddings"),
        "bindings": ("embeddings", "polish", "reference"),
        "architecture": ("architecture", "native_packages"),
        "lora-reference": ("architecture", "reference"),
        "product": ("network", "embeddings", "polish", "distribution_scales", "reference"),
    }
    files = set()
    for group in groups:
        if group.startswith("network:"):
            name = normalized(group.removeprefix("network:"))
            if "/" in name or not name.startswith("test_") or not name.endswith(".py"):
                raise ValueError("Invalid HTTP test file")
            if not (ROOT / "acceptance" / name).is_file():
                raise ValueError(f"Empty required HTTP file: {name}")
            files.add(f"acceptance/{name}")
        elif group in owners:
            files.update(f"acceptance/test_{name}.py" for name in owners[group])
        else:
            raise ValueError("Unknown required integration selection")
    if not files:
        raise ValueError("Empty required integration selection")
    return sorted(files)


def backend_targets(plan):
    files = plan["backend_tests"]
    if "backend" in plan["owners"] and not files:
        raise ValueError("Empty required backend selection")
    for name in files:
        normalized(name)
        if name != "tests" and (not name.startswith("tests/test_") or not name.endswith(".py")):
            raise ValueError("Invalid backend test file")
    return ["tests"] if "tests" in files else files


def validate(plan):
    if plan["portfolio"] not in {"routine", "full"} or set(plan["extended"]) - {
        "backend",
        "integration",
    }:
        raise ValueError("Invalid portfolio/extended owners")
    if not isinstance(plan["compatibility_full"], bool):
        raise ValueError("Invalid compatibility mode")
    backend_targets(plan)
    if set(plan["owners"]) - OWNERS:
        raise ValueError("Unknown validation owner")
    for gate in ("browser", "integration"):
        targets(plan, gate)
        if plan[gate] and gate not in plan["owners"]:
            raise ValueError(f"Unowned {gate} selection")
    if "integration" in plan["owners"]:
        network_targets(plan)


def run_browser(plan, gate, *, main=False, extended=False):
    validate(plan)
    if gate not in plan["owners"]:
        print(f"{gate}: explicitly non-applicable")
        return
    files = targets(plan, gate)
    if not files:
        print("integration browser: explicitly non-applicable; selected HTTP tests still required")
        return
    if extended:
        if gate not in plan["extended"] or plan["portfolio"] == "full":
            print(f"{gate}: extended explicitly non-applicable/already included")
            return
        # Known logical files have no extended classification. Unknown/nested
        # native files stay selected conservatively, including future extended tags.
        files = [name for name in files if name not in {"transport.spec.ts", "scientific.spec.ts"}]
        if not files:
            print(f"{gate}: no classified extended consumer")
            return
    projects = (
        []
        if main
        else (
            ["--project=desktop", "--project=native-scrollbars"]
            if gate == "browser"
            else ["--project=dpr1"]
        )
    )
    if extended:
        projects = ["--project=dpr1"]
    command = [
        "xvfb-run",
        "-a",
        "npm",
        "run",
        "test:browser" if gate == "browser" else "test:acceptance",
        "--",
        # Playwright CLI file filters are regular expressions, not literal paths.
        *map(re.escape, files),
        *projects,
    ]
    print(json.dumps(command), flush=True)
    environment = os.environ | {
        "LMEX_TEST_PORTFOLIO": "extended" if extended else plan["portfolio"]
    }
    subprocess.run(command, cwd=ROOT / "ui", env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--github", action="store_true")
    source.add_argument("--paths", type=Path, help="complete normalized JSON path array")
    source.add_argument("--full", action="store_true")
    source.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--run", choices=("browser", "integration"))
    parser.add_argument("--network-targets", action="store_true")
    parser.add_argument("--backend-targets", action="store_true")
    parser.add_argument("--integration-browser-needed", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--main", action="store_true")
    args = parser.parse_args()
    try:
        if args.github:
            try:
                payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
                plan = github_plan(os.environ["GITHUB_EVENT_NAME"], payload)
            except (OSError, ValueError, KeyError) as error:
                plan = select([])
                plan["reasons"] = [f"Unavailable event; full fallback: {error}"]
        elif args.plan:
            plan = json.loads(args.plan.read_text())
        elif args.paths:
            paths = json.loads(args.paths.read_text())
            if not isinstance(paths, list):
                raise ValueError("Changed paths must be a JSON array")
            plan = select(paths)
        else:
            plan = select([], full=True)
        validate(plan)
        if args.output:
            args.output.write_text(json.dumps(plan, indent=2) + "\n")
        if args.github_output:
            with args.github_output.open("a") as output:
                for owner in sorted(OWNERS):
                    output.write(f"{owner}={str(owner in plan['owners']).lower()}\n")
                output.write(f"compatibility_full={str(plan['compatibility_full']).lower()}\n")
                output.write(f"portfolio={plan['portfolio']}\n")
                output.write(
                    f"integration_browser={str(bool(targets(plan, 'integration'))).lower()}\n"
                )
                output.write(f"backend_extended={str('backend' in plan['extended']).lower()}\n")
        if args.summary:
            with args.summary.open("a") as summary:
                summary.write(
                    "### Validation selection\n\n```json\n" + json.dumps(plan, indent=2) + "\n```\n"
                )
        if args.network_targets:
            print("\n".join(network_targets(plan)))
        elif args.integration_browser_needed:
            print(str(bool(targets(plan, "integration"))).lower())
        elif args.backend_targets:
            print("\n".join(backend_targets(plan)))
        elif args.run:
            run_browser(plan, args.run, main=args.main, extended=args.extended)
        else:
            print(json.dumps(plan, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Invalid validation plan: {error}\n")
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode)


if __name__ == "__main__":
    main()
