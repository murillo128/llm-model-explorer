"""Conservative validation owners and exact Git diff selection (stdlib only)."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
OWNERS = {"backend", "api", "ui", "browser", "integration"}
# These native test modules also export fixtures/oracles to other tests and
# support modules. Keep the finite classification aligned with their imports;
# a test_ filename alone does not establish isolation.
SHARED_BACKEND_TESTS = {
    f"backend/tests/test_{name}.py"
    for name in (
        "architecture_analysis",
        "artifacts",
        "dense_architecture",
        "embeddings",
        "lmex",
        "model_defined_service",
        "models",
        "operations",
        "peft_adapters",
        "quantized_models",
        "qwen35_architecture",
        "streaming",
        "tensor_analysis",
        "tensor_data",
        "tokenization",
        "vjepa2_architecture",
    )
}

# Positive leaf classifications reviewed at 83843e3. New/deleted/nested inputs
# stay conservative; update these and consumer groups when relationships change.
BACKEND_LEAVES = {
    "backend/tests/test_app.py",
    "backend/tests/test_architecture_artifacts.py",
    "backend/tests/test_architecture_grouping.py",
    "backend/tests/test_architecture_service.py",
    "backend/tests/test_architecture_templates.py",
    "backend/tests/test_bitsandbytes_nf4.py",
    "backend/tests/test_catalogue_cache.py",
    "backend/tests/test_cli.py",
    "backend/tests/test_clm_export.py",
    "backend/tests/test_compact_architecture.py",
    "backend/tests/test_deepseek_v2_architecture.py",
    "backend/tests/test_embedding_analysis.py",
    "backend/tests/test_glm4_moe_lite_architecture.py",
    "backend/tests/test_kev_export.py",
    "backend/tests/test_model_defined_architecture.py",
    "backend/tests/test_model_defined_templates.py",
    "backend/tests/test_model_defined_templates_service.py",
    "backend/tests/test_operation_cards.py",
    "backend/tests/test_quantized_decoding.py",
    "backend/tests/test_quantized_flows.py",
    "backend/tests/test_session_operations.py",
    "backend/tests/test_sessions.py",
    "backend/tests/test_settings.py",
}
BROWSER_LEAVES = {
    "ui/tests/acceptance-probe.spec.ts",
    "ui/tests/api-transport.spec.ts",
    "ui/tests/architecture-browser.spec.ts",
    "ui/tests/architecture-camera.spec.ts",
    "ui/tests/architecture-card-actions.spec.ts",
    "ui/tests/architecture-components.spec.ts",
    "ui/tests/architecture-connections.spec.ts",
    "ui/tests/architecture-controls.spec.ts",
    "ui/tests/architecture-inspection.spec.ts",
    "ui/tests/architecture-interfaces.spec.ts",
    "ui/tests/architecture-isolation.spec.ts",
    "ui/tests/architecture-notifications.spec.ts",
    "ui/tests/architecture-overview.spec.ts",
    "ui/tests/architecture-shell.spec.ts",
    "ui/tests/architecture-summaries.spec.ts",
    "ui/tests/architecture.spec.ts",
    "ui/tests/distribution-scale.spec.ts",
    "ui/tests/magnifier-adaptation.spec.ts",
    "ui/tests/matrix-centering.spec.ts",
    "ui/tests/matrix-explorer.spec.ts",
    "ui/tests/matrix-inspection.spec.ts",
    "ui/tests/matrix-overlay-scrollbars.spec.ts",
    "ui/tests/matrix-zoom-pixels.spec.ts",
    "ui/tests/matrix-zoom-selection.spec.ts",
    "ui/tests/matrix-zoom.spec.ts",
    "ui/tests/model-diagnostics.spec.ts",
    "ui/tests/pane-loading.spec.ts",
    "ui/tests/renderer.spec.ts",
    "ui/tests/sessions.spec.ts",
    "ui/tests/shell-feedback.spec.ts",
    "ui/tests/shell-viewport.spec.ts",
    "ui/tests/shell.spec.ts",
    "ui/tests/tensor-explorer-scrollbars.spec.ts",
    "ui/tests/tensor-explorer.spec.ts",
    "ui/tests/tensor-header.spec.ts",
    "ui/tests/tensor-inventory.spec.ts",
    "ui/tests/tensor-navigation.spec.ts",
    "ui/tests/tokenizer-embeddings.spec.ts",
    "ui/tests/tokenizer-layout.spec.ts",
    "ui/tests/tokenizer-model-switch.spec.ts",
    "ui/tests/tokenizer-prompt-regression.spec.ts",
    "ui/tests/tokenizer.spec.ts",
}
HTTP_LEAVES = {
    "acceptance/test_architecture.py",
    "acceptance/test_ci_entrypoints.py",
    "acceptance/test_distribution_scales.py",
    "acceptance/test_embeddings.py",
    "acceptance/test_lora_reference.py",
    "acceptance/test_native_packages.py",
    "acceptance/test_polish.py",
    "acceptance/test_reference.py",
    "acceptance/test_report.py",
    "acceptance/test_validation_selector.py",
}
UI_UNIT_LEAVES = {
    "ui/src/api/architecture-client.test.ts",
    "ui/src/api/architecture-schema.test.ts",
    "ui/src/api/architecture-validation.test.ts",
    "ui/src/api/client.test.ts",
    "ui/src/api/embedding-analysis.test.ts",
    "ui/src/api/embeddings.test.ts",
    "ui/src/api/lmex-decoder.test.ts",
    "ui/src/api/lora-architecture.test.ts",
    "ui/src/api/runtime-config.test.ts",
    "ui/src/api/testing/wire-conformance.test.ts",
    "ui/src/app/App.test.tsx",
    "ui/src/app/Bootstrap.test.tsx",
    "ui/src/app/ModelDiagnostics.test.tsx",
    "ui/src/app/ToastHost.test.tsx",
    "ui/src/app/feedback.test.ts",
    "ui/src/app/model-diagnostics.test.ts",
    "ui/src/app/session-controller.test.ts",
    "ui/src/architecture-explorer/ArchitectureControls.test.tsx",
    "ui/src/architecture-explorer/ArchitectureExplorer.test.tsx",
    "ui/src/architecture-explorer/ArchitectureInspection.test.tsx",
    "ui/src/architecture-explorer/ArchitectureWorkspace.test.tsx",
    "ui/src/architecture-explorer/CardSummary.test.tsx",
    "ui/src/architecture-explorer/auto-layout.test.ts",
    "ui/src/architecture-explorer/browser-model.test.ts",
    "ui/src/architecture-explorer/card-actions.test.ts",
    "ui/src/architecture-explorer/component-actions.test.tsx",
    "ui/src/architecture-explorer/connection-hit.test.ts",
    "ui/src/architecture-explorer/graph.test.ts",
    "ui/src/architecture-explorer/interfaces.test.ts",
    "ui/src/architecture-explorer/invariants.test.ts",
    "ui/src/architecture-explorer/model-defined-shared.test.ts",
    "ui/src/architecture-explorer/overview.test.ts",
    "ui/src/architecture-explorer/projection.test.ts",
    "ui/src/architecture-explorer/repeated-layout.test.ts",
    "ui/src/architecture-explorer/routing-clearance.test.ts",
    "ui/src/architecture-explorer/routing-compare.test.ts",
    "ui/src/architecture-explorer/scope-navigation.test.ts",
    "ui/src/architecture-explorer/scope.test.ts",
    "ui/src/architecture-explorer/shared-structure.test.ts",
    "ui/src/architecture-explorer/useCanvasCallback.test.tsx",
    "ui/src/components/TensorHeader.test.tsx",
    "ui/src/components/TensorTree.test.tsx",
    "ui/src/components/TensorWorkspace.test.tsx",
    "ui/src/explorers/stream-words.test.ts",
    "ui/src/matrix-explorer/MatrixExplorer.test.tsx",
    "ui/src/rendering/camera-history.test.ts",
    "ui/src/rendering/chroma.test.ts",
    "ui/src/rendering/distribution-scale.test.ts",
    "ui/src/rendering/geometry.test.ts",
    "ui/src/rendering/inspection-layout.test.ts",
    "ui/src/rendering/matrix-camera-navigation.test.ts",
    "ui/src/rendering/matrix-row-selection.test.ts",
    "ui/src/rendering/matrix-scrollbars.test.ts",
    "ui/src/rendering/wheel-gesture.test.ts",
    "ui/src/rendering/zoom-selection-geometry.test.ts",
    "ui/src/test/pixel-comparison.test.ts",
    "ui/src/tokenizer/annotations.test.ts",
    "ui/src/tokenizer/embedding-controller.test.ts",
}

# Cross-boundary closure includes function-local/embedded imports and fixture JSON.
CROSS_BOUNDARY_BACKEND = {
    "backend/tests/architecture_assertions.py": ("architecture",),
    "backend/tests/architecture_grouping_cases.py": ("architecture",),
    "backend/tests/cache_helpers.py": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/clm_fixtures.py": (
        "network:test_native_packages.py",
        "network:test_architecture.py",
    ),
    "backend/tests/dense_fixtures.py": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/fixtures/architecture-semantics-baseline.json": ("architecture",),
    "backend/tests/fixtures/dense-reference-metadata.json": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/fixtures/kimi-linear-reference.json": ("architecture",),
    "backend/tests/fixtures/quantized-configs.json": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/fixtures/qwen35-reference.json": ("architecture",),
    "backend/tests/fixtures/qwen35-tiny.json": ("architecture",),
    "backend/tests/fixtures/vjepa2-reference.json": ("architecture",),
    "backend/tests/fixtures/vjepa2-tiny.json": ("architecture",),
    "backend/tests/kev_fixtures.py": (
        "network:test_native_packages.py",
        "network:test_architecture.py",
    ),
    "backend/tests/quantized_oracles.py": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/test_clm_export.py": (
        "network:test_native_packages.py",
        "network:test_architecture.py",
    ),
    "backend/tests/test_kev_export.py": (
        "network:test_native_packages.py",
        "network:test_architecture.py",
    ),
    "backend/tests/test_deepseek_v2_architecture.py": ("architecture",),
    "backend/tests/test_glm4_moe_lite_architecture.py": ("network:test_architecture.py",),
    "backend/tests/test_dense_architecture.py": ("architecture",),
    "backend/tests/test_kimi_linear_architecture.py": ("architecture",),
    "backend/tests/test_lmex.py": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/test_lora_architecture.py": (
        "network:test_native_packages.py",
        "network:test_architecture.py",
    ),
    "backend/tests/test_models.py": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/test_operations.py": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/test_quantized_models.py": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/test_qwen35_architecture.py": ("architecture",),
    "backend/tests/test_streaming.py": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/test_tensor_data.py": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "backend/tests/test_vjepa2_architecture.py": ("architecture",),
}

INTEGRATION_SUPPORT = {
    "acceptance/architecture_fixtures.py": (
        "architecture",
        "network:test_polish.py",
        "network:test_lora_reference.py",
    ),
    "acceptance/kimi_linear_fixture.py": ("architecture",),
    "acceptance/polish_fixtures.py": ("network:test_polish.py",),
    "acceptance/architecture_reference.py": (
        "architecture",
        "lora-reference",
        "network:test_polish.py",
    ),
    "acceptance/quantized_reference.py": (
        "architecture",
        "lora-reference",
        "bindings",
        "network:test_polish.py",
    ),
    "acceptance/reference.py": ("bindings", "network:test_reference.py"),
}


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
        if path in CROSS_BOUNDARY_BACKEND:
            add(
                path,
                "backend fixture/import closure and real TCP/browser consumers",
                ("backend",),
                integrations=CROSS_BOUNDARY_BACKEND[path],
            )
            if path in SHARED_BACKEND_TESTS or path in {
                "backend/tests/cache_helpers.py",
                "backend/tests/dense_fixtures.py",
                "backend/tests/quantized_oracles.py",
                "backend/tests/fixtures/quantized-configs.json",
            }:
                extended.add("backend")
        elif path in INTEGRATION_SUPPORT:
            add(
                path,
                "shared integration fixture/oracle and real consumers",
                integrations=INTEGRATION_SUPPORT[path],
            )
        elif path in SHARED_BACKEND_TESTS:
            add(path, "shared backend test fixtures/oracles and native consumers", ("backend",))
            extended.add("backend")
        elif path == "acceptance/test_network.py":
            add(
                path,
                "shared real CLI/service/stream helpers and all HTTP consumers",
                integrations=(
                    "architecture",
                    *(
                        "network:" + file.name
                        for file in sorted((ROOT / "acceptance").glob("test_*.py"))
                    ),
                ),
            )
        elif path in BACKEND_LEAVES and (ROOT / path).is_file():
            owners.add("backend")
            backend_tests.add(path.removeprefix("backend/"))
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
        elif path in BROWSER_LEAVES and (ROOT / path).is_file():
            add(
                path,
                "changed native browser file",
                ("ui",),
                ("file:" + path.removeprefix("ui/tests/"),),
            )
        elif path in UI_UNIT_LEAVES and (ROOT / path).is_file():
            add(path, "UI unit test uses the native unit/type/lint owner", ("ui",))
        elif path in HTTP_LEAVES and (ROOT / path).is_file():
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
                ("architecture", "lora-reference"),
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
        "backend_tests": ["tests"] if "tests" in backend_tests else sorted(backend_tests),
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
        "lora-reference": ("architecture", "lora_reference", "reference"),
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


def effective_plan(plan):
    """Resolve reference promotion at the plan/config seam, before phase routing."""
    requested = plan.get("requested_portfolio", plan["portfolio"])
    if requested not in {"routine", "full"}:
        raise ValueError("Invalid requested portfolio")
    references = [
        name
        for name in (
            "LMEX_REFERENCE_MODEL_DIR",
            "LMEX_ARCHITECTURE_REFERENCES",
            "LMEX_LORA_REFERENCE_MODEL_ROOT",
        )
        if os.environ.get(name)
    ]
    if os.environ.get("LMEX_REQUIRE_ARCHITECTURE_REFERENCES") == "1":
        references.append("LMEX_REQUIRE_ARCHITECTURE_REFERENCES")
    return plan | {
        "requested_portfolio": requested,
        "portfolio": "full"
        if requested == "full" or plan["portfolio"] == "full" or references
        else "routine",
        "reference_inputs": references,
    }


def phase_targets(plan, gate, *, extended=False):
    files = targets(plan, gate)
    if gate != "integration":
        return [] if extended else files
    if extended:
        if plan["portfolio"] == "full" or gate not in plan["extended"]:
            return []
        # Reviewed files with no extended cases. Unknown files remain required.
        return [name for name in files if name not in {"transport.spec.ts", "scientific.spec.ts"}]
    if plan["portfolio"] == "routine":
        # This one declared owner contains only extended cases. No other empty
        # discovery result is accepted as successful/non-applicable execution.
        return [name for name in files if name != "lora-reference.spec.ts"]
    return files


def run_browser(plan, gate, *, main=False, extended=False):
    plan = effective_plan(plan)
    validate(plan)
    if gate not in plan["owners"]:
        print(f"{gate}: explicitly non-applicable")
        return
    files = phase_targets(plan, gate, extended=extended)
    if not files:
        phase = "extended" if extended else plan["portfolio"]
        print(f"{gate} {phase} browser: explicitly non-applicable; selected owners remain required")
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
    phase = "extended" if extended else plan["portfolio"]
    environment = os.environ | {"LMEX_TEST_PORTFOLIO": phase, "LMEX_TEST_PHASE": phase}
    if environment.get("LMEX_EVIDENCE_DIR"):
        environment["LMEX_EVIDENCE_DIR"] = str(Path(environment["LMEX_EVIDENCE_DIR"]).resolve())
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
        plan = effective_plan(plan)
        validate(plan)
        if args.output:
            args.output.write_text(json.dumps(plan, indent=2) + "\n")
        if args.github_output:
            routine_browser = bool(phase_targets(plan, "integration"))
            extended_browser = bool(phase_targets(plan, "integration", extended=True))
            integration_browser = routine_browser or extended_browser
            network = network_targets(plan) if "integration" in plan["owners"] else []
            # The TCP architecture owner runs the independent semantic oracle
            # with Node's native TypeScript support, without npm or Chromium.
            integration_node = integration_browser or bool(
                set(network) & {"acceptance", "acceptance/test_architecture.py"}
            )
            with args.github_output.open("a") as output:
                for owner in sorted(OWNERS):
                    output.write(f"{owner}={str(owner in plan['owners']).lower()}\n")
                output.write(f"compatibility_full={str(plan['compatibility_full']).lower()}\n")
                output.write(f"portfolio={plan['portfolio']}\n")
                output.write(f"integration_browser={str(integration_browser).lower()}\n")
                output.write(f"integration_routine={str(routine_browser).lower()}\n")
                output.write(f"integration_extended={str(extended_browser).lower()}\n")
                output.write(f"integration_node={str(integration_node).lower()}\n")
                output.write(f"backend_extended={str('backend' in plan['extended']).lower()}\n")
        if args.summary:
            with args.summary.open("a") as summary:
                summary.write(
                    "### Validation selection\n\n```json\n" + json.dumps(plan, indent=2) + "\n```\n"
                )
        if args.network_targets:
            print("\n".join(network_targets(plan)))
        elif args.integration_browser_needed:
            print(
                str(
                    bool(
                        phase_targets(plan, "integration")
                        or phase_targets(plan, "integration", extended=True)
                    )
                ).lower()
            )
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
