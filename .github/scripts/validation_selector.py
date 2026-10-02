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

    def add(path, why, checks=(), browsers=(), integrations=()):
        owners.update(checks)
        browser.update(browsers)
        integration.update(integrations)
        reasons.append(f"{path}: {why}")

    def broad(path, why):
        add(path, why, OWNERS, ("full",), ("full",))

    if full or not paths:
        broad("context", "explicit full mode" if full else "empty diff; full fallback")
    for path in paths:
        if path.startswith(("api/", "docs/spec/api/", "ui/src/api/", "ui/scripts/api-generator/")):
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
        elif path.startswith(("backend/", "examples/")):
            add(
                path,
                "backend/native admission/export and real product consumers",
                ("backend", "integration"),
                integrations=("full",),
            )
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
                ("full",),
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
                ("product",),
            )
        elif path.startswith(("ui/", "acceptance/", ".github/workflows/")) or path in (
            ".github/scripts/validation_selector.py",
            ".github/scripts/test_validation_selector.py",
        ):
            broad(path, "shared composition, tests, fixtures, configuration or routing")
        elif path.startswith((".github/scripts/", "scripts/")) and Path(path).name in {
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
    if browser:
        owners.add("browser")
    if integration:
        owners.add("integration")
    return {
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
            plan = select(paths, full=pr["head"]["ref"].startswith("codex/epic-issue-"))
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
        plan = select([], full=True)
        plan["reasons"] = [f"Invalid/unavailable diff; full fallback: {error}"]
    return plan | {"context": context}


def targets(plan, gate, *, root=ROOT):
    """Resolve existing native files. An empty required group is an error."""
    groups = plan[gate]
    if gate == "browser":
        patterns = {
            "full": ("*.spec.ts",),
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
        patterns = {"full": ("*.spec.ts",), "product": ("product.spec.ts",)}
        directory = root / "ui/acceptance"
    else:
        raise ValueError(f"Unknown gate: {gate}")
    files = set()
    for group in groups:
        if group not in patterns:
            raise ValueError(f"Unknown required group: {group}")
        matched = {file.name for pattern in patterns[group] for file in directory.glob(pattern)}
        if not matched:
            raise ValueError(f"Empty required {gate} group: {group}")
        files.update(matched)
    if gate in plan["owners"] and not files:
        raise ValueError(f"Empty required {gate} selection")
    return sorted(files)


def network_targets(plan):
    if "full" in plan["integration"]:
        return ["acceptance"]
    if plan["integration"] == ["product"]:
        return [
            f"acceptance/test_{name}.py"
            for name in (
                "network",
                "embeddings",
                "polish",
                "distribution_scales",
                "reference",
                "ci_entrypoints",
                "report",
                "validation_selector",
            )
        ]
    raise ValueError("Empty or unknown required integration selection")


def validate(plan):
    if set(plan["owners"]) - OWNERS:
        raise ValueError("Unknown validation owner")
    for gate in ("browser", "integration"):
        targets(plan, gate)
        if plan[gate] and gate not in plan["owners"]:
            raise ValueError(f"Unowned {gate} selection")
    if "integration" in plan["owners"]:
        network_targets(plan)


def run_browser(plan, gate, *, main=False):
    validate(plan)
    if gate not in plan["owners"]:
        print(f"{gate}: explicitly non-applicable")
        return
    files = targets(plan, gate)
    projects = (
        []
        if main
        else (
            ["--project=desktop", "--project=native-scrollbars"]
            if gate == "browser"
            else ["--project=dpr1"]
        )
    )
    command = [
        "xvfb-run",
        "-a",
        "npm",
        "run",
        "test:browser" if gate == "browser" else "test:acceptance",
        "--",
        *files,
        *projects,
    ]
    print(json.dumps(command), flush=True)
    subprocess.run(command, cwd=ROOT / "ui", check=True)


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
    parser.add_argument("--main", action="store_true")
    args = parser.parse_args()
    try:
        if args.github:
            try:
                payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
                plan = github_plan(os.environ["GITHUB_EVENT_NAME"], payload)
            except (OSError, ValueError, KeyError) as error:
                plan = select([], full=True)
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
        if args.summary:
            with args.summary.open("a") as summary:
                summary.write(
                    "### Validation selection\n\n```json\n" + json.dumps(plan, indent=2) + "\n```\n"
                )
        if args.network_targets:
            print("\n".join(network_targets(plan)))
        elif args.run:
            run_browser(plan, args.run, main=args.main)
        else:
            print(json.dumps(plan, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Invalid validation plan: {error}\n")
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode)


if __name__ == "__main__":
    main()
