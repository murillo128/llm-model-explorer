"""Behavioral selection/diff oracles; no application boundary is mocked as proof."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "validation_selector",
    Path(__file__).resolve().parents[1] / ".github/scripts/validation_selector.py",
)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


class SelectionTest(unittest.TestCase):
    def test_responsibility_table(self):
        cases = [
            (
                "ui/src/architecture-explorer/Graph.tsx",
                {"ui", "browser", "integration"},
                ["architecture"],
                ["architecture", "lora-reference"],
            ),
            (
                "ui/src/tokenizer/editor.ts",
                {"ui", "browser", "integration"},
                ["tokenizer"],
                ["bindings", "scientific", "tokenizer-layout", "transport"],
            ),
            (
                "ui/src/rendering/renderer.ts",
                {"ui", "browser", "integration"},
                ["architecture", "matrix", "tokenizer"],
                ["full"],
            ),
            (
                "ui/src/matrix-explorer/MatrixExplorer.tsx",
                {"ui", "browser", "integration"},
                ["architecture", "matrix", "tokenizer"],
                ["full"],
            ),
            (
                "backend/src/llm_model_explorer/sessions.py",
                {"backend", "integration"},
                [],
                ["full"],
            ),
            (
                "examples/clm/export.py",
                {"backend", "ui", "browser", "integration"},
                ["architecture"],
                ["full"],
            ),
            (
                "examples/kev/export.py",
                {"backend", "ui", "browser", "integration"},
                ["architecture"],
                ["full"],
            ),
            ("skills/test-quality/references/repository-testing.md", set(), [], []),
        ]
        for path, owners, browser, integration in cases:
            with self.subTest(path=path):
                plan = selector.select([path])
                self.assertEqual(set(plan["owners"]), owners)
                self.assertEqual(plan["browser"], browser)
                self.assertEqual(plan["integration"], integration)
                self.assertTrue(plan["reasons"])
                selector.validate(plan)

    def test_shared_unknown_and_normative_inputs_are_broad(self):
        for path in (
            "api/fixtures/tensor.json",
            "docs/spec/api/binary-streaming.md",
            "docs/spec/ui/tokenizer-explorer.md",
            "docs/spec/backend/models.md",
            "docs/spec/product.md",
            "ui/scripts/api-generator/generate.mjs",
            "ui/src/api/generated/schema.ts",
            "ui/src/app/App.tsx",
            "ui/src/components/Button.tsx",
            "ui/src/explorers/TensorExplorer.tsx",
            "ui/src/app/global.css",
            "ui/tests/new.spec.ts",
            "ui/tests/scalar-oracle.ts",
            "ui/acceptance/probe.ts",
            "acceptance/fixtures.py",
            "ui/package-lock.json",
            "ui/playwright.config.ts",
            ".github/workflows/ui-ci.yml",
            ".github/scripts/validation_selector.py",
            "new-product/module.py",
            ".github/scripts/new_product_input.py",
            ".github/scripts/product/codex_profile.py",
            "scripts/codex_profile.py",
            "package-lock.json",
        ):
            with self.subTest(path=path):
                plan = selector.select([path])
                self.assertEqual(set(plan["owners"]), selector.OWNERS)
                self.assertEqual(plan["browser"], ["full"])
                self.assertEqual(plan["integration"], ["full"])

    def test_operational_markdown_is_explicitly_non_applicable(self):
        for path in (
            "acceptance/README.md",
            "acceptance/test-cost-audit.md",
            "ui/evidence/domain-test-ownership.md",
            "backend/README.md",
            "examples/clm/evidence.md",
            "examples/kev/README.md",
        ):
            with self.subTest(path=path):
                plan = selector.select([path])
                self.assertEqual(plan["owners"], [])
                self.assertEqual(plan["browser"], [])
                self.assertEqual(plan["integration"], [])

    def test_union_and_new_native_cases(self):
        plan = selector.select(["ui/src/tokenizer/a.ts", "ui/src/architecture-explorer/a.ts"])
        self.assertEqual(plan["browser"], ["architecture", "tokenizer"])
        self.assertEqual(
            plan["integration"],
            [
                "architecture",
                "bindings",
                "lora-reference",
                "scientific",
                "tokenizer-layout",
                "transport",
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ui/tests").mkdir(parents=True)
            for name in ("architecture-new.spec.ts", "tokenizer-new.spec.ts", "new.spec.ts"):
                (root / "ui/tests" / name).touch()
            self.assertEqual(
                selector.targets(plan, "browser", root=root),
                ["architecture-new.spec.ts", "tokenizer-new.spec.ts"],
            )
            self.assertEqual(
                selector.targets(selector.select([], full=True), "browser", root=root),
                ["architecture-new.spec.ts", "new.spec.ts", "tokenizer-new.spec.ts"],
            )

    def test_changed_native_files_and_backend_test_are_isolated(self):
        plan = selector.select(["ui/tests/matrix-zoom.spec.ts"])
        self.assertEqual(plan["integration"], [])
        self.assertEqual(selector.targets(plan, "browser"), ["matrix-zoom.spec.ts"])
        backend = selector.select(["backend/tests/test_session_operations.py"])
        self.assertEqual(backend["owners"], ["backend"])
        self.assertEqual(selector.backend_targets(backend), ["tests/test_session_operations.py"])
        self.assertEqual(backend["extended"], [])
        scientific = selector.select(["ui/acceptance/scientific.spec.ts"])
        self.assertEqual(selector.targets(scientific, "integration"), ["scientific.spec.ts"])
        self.assertEqual(
            selector.network_targets(scientific),
            [
                "acceptance/test_distribution_scales.py",
                "acceptance/test_embeddings.py",
                "acceptance/test_network.py",
            ],
        )

    def test_shared_backend_test_modules_reach_all_native_consumers_and_thresholds(self):
        # Existing imports include quantized_flows -> tensor_analysis and
        # embedding_analysis -> embeddings -> tokenization -> models. A helper
        # module remains shared even though its filename starts with test_.
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
        ):
            with self.subTest(module=name):
                plan = selector.select([f"backend/tests/test_{name}.py"])
                cross_boundary = name in {
                    "dense_architecture",
                    "lmex",
                    "models",
                    "operations",
                    "quantized_models",
                    "qwen35_architecture",
                    "streaming",
                    "tensor_data",
                    "vjepa2_architecture",
                }
                self.assertEqual(
                    plan["owners"], ["backend", "integration"] if cross_boundary else ["backend"]
                )
                self.assertEqual(selector.backend_targets(plan), ["tests"])
                self.assertEqual(
                    plan["extended"], ["backend", "integration"] if cross_boundary else ["backend"]
                )
                selector.validate(plan)

    def test_shared_http_test_module_reaches_real_model_root_and_stream_consumers(self):
        plan = selector.select(["acceptance/test_network.py"])
        self.assertEqual(plan["owners"], ["integration"])
        self.assertEqual(selector.targets(plan, "integration"), ["architecture.spec.ts"])
        files = selector.network_targets(plan)
        for name in (
            "network",
            "architecture",
            "native_packages",
            "embeddings",
            "distribution_scales",
            "polish",
            "reference",
            "lora_reference",
        ):
            self.assertIn(f"acceptance/test_{name}.py", files)
        self.assertIn("integration", plan["extended"])
        selector.validate(plan)

    def test_cross_boundary_fixture_closure_selects_actual_native_consumers(self):
        cases = [
            *[
                (
                    f"backend/tests/{name}.py",
                    {"acceptance/test_native_packages.py", "acceptance/test_architecture.py"},
                    set(),
                )
                for name in (
                    "test_lora_architecture",
                    "clm_fixtures",
                    "kev_fixtures",
                    "test_clm_export",
                    "test_kev_export",
                )
            ],
            (
                "backend/tests/test_glm4_moe_lite_architecture.py",
                {"acceptance/test_architecture.py"},
                set(),
            ),
            *[
                (
                    f"backend/tests/{name}",
                    {"acceptance/test_architecture.py"},
                    {"architecture.spec.ts"},
                )
                for name in (
                    "architecture_assertions.py",
                    "architecture_grouping_cases.py",
                    "test_dense_architecture.py",
                    "test_deepseek_v2_architecture.py",
                    "test_kimi_linear_architecture.py",
                    "test_qwen35_architecture.py",
                    "test_vjepa2_architecture.py",
                    "fixtures/architecture-semantics-baseline.json",
                    "fixtures/kimi-linear-reference.json",
                    "fixtures/qwen35-reference.json",
                    "fixtures/qwen35-tiny.json",
                    "fixtures/vjepa2-reference.json",
                    "fixtures/vjepa2-tiny.json",
                )
            ],
            *[
                (
                    f"backend/tests/{name}",
                    {
                        "acceptance/test_architecture.py",
                        "acceptance/test_native_packages.py",
                        "acceptance/test_polish.py",
                    },
                    {"architecture.spec.ts"},
                )
                for name in (
                    "dense_fixtures.py",
                    "quantized_oracles.py",
                    "test_quantized_models.py",
                    "test_models.py",
                    "test_tensor_data.py",
                    "test_streaming.py",
                    "test_operations.py",
                    "test_lmex.py",
                    "cache_helpers.py",
                    "fixtures/dense-reference-metadata.json",
                    "fixtures/quantized-configs.json",
                )
            ],
        ]
        for path, http, browser in cases:
            with self.subTest(path=path):
                plan = selector.select([path])
                self.assertIn("backend", plan["owners"])
                self.assertIn("integration", plan["owners"])
                self.assertTrue(http <= set(selector.network_targets(plan)))
                self.assertTrue(browser <= set(selector.targets(plan, "integration")))
                self.assertFalse(plan["compatibility_full"])
                selector.validate(plan)

    def test_integration_support_selects_its_fixture_and_reference_consumers(self):
        cases = [
            (
                "architecture_fixtures.py",
                {"test_architecture.py", "test_native_packages.py", "test_polish.py"},
                {"architecture.spec.ts"},
            ),
            ("kimi_linear_fixture.py", {"test_architecture.py"}, {"architecture.spec.ts"}),
            ("polish_fixtures.py", {"test_polish.py"}, set()),
            (
                "architecture_reference.py",
                {"test_architecture.py", "test_lora_reference.py", "test_polish.py"},
                {"architecture.spec.ts", "lora-reference.spec.ts"},
            ),
            (
                "quantized_reference.py",
                {"test_architecture.py", "test_lora_reference.py", "test_polish.py"},
                {"architecture.spec.ts", "lora-reference.spec.ts", "bindings.spec.ts"},
            ),
            ("reference.py", {"test_reference.py"}, {"bindings.spec.ts"}),
        ]
        for name, http, browser in cases:
            with self.subTest(path=name):
                plan = selector.select([f"acceptance/{name}"])
                self.assertEqual(plan["owners"], ["integration"])
                self.assertTrue(
                    {f"acceptance/{file}" for file in http} <= set(selector.network_targets(plan))
                )
                self.assertEqual(set(selector.targets(plan, "integration")), browser)
                selector.validate(plan)

    def test_existing_unclassified_test_names_do_not_establish_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in [
                "backend/tests/test_future_helper.py",
                "ui/tests/matrix-future.spec.ts",
                "ui/src/api/future.test.ts",
                "acceptance/test_future_helper.py",
            ]:
                file = root / path
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
                with self.subTest(path=path), patch.object(selector, "ROOT", root):
                    plan = selector.select([path])
                    self.assertIn("integration", plan["owners"])
                    self.assertNotEqual(plan["integration"], [])

    def test_cross_boundary_union_is_deterministic_and_preserves_full_owner(self):
        paths = [
            "backend/tests/test_session_operations.py",
            "backend/tests/test_quantized_models.py",
            "backend/tests/test_lora_architecture.py",
        ]
        first = selector.select(paths)
        self.assertEqual(first, selector.select([*reversed(paths), paths[0]]))
        self.assertEqual(first["backend_tests"], ["tests"])
        files = selector.network_targets(first)
        self.assertEqual(files, sorted(set(files)))
        self.assertTrue(
            {
                "acceptance/test_native_packages.py",
                "acceptance/test_polish.py",
                "acceptance/test_architecture.py",
            }
            <= set(files)
        )

    def test_architecture_ui_selects_lora_browser_and_independent_tcp(self):
        plan = selector.select(["ui/src/architecture-explorer/ArchitectureControls.tsx"])
        self.assertIn("lora-reference.spec.ts", selector.targets(plan, "integration"))
        self.assertIn("acceptance/test_lora_reference.py", selector.network_targets(plan))

    def test_extended_only_routine_and_reference_full_do_not_launch_duplicate_phases(self):
        plan = selector.select(["ui/acceptance/lora-reference.spec.ts"])
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(selector.subprocess, "run") as command,
        ):
            selector.run_browser(plan, "integration", main=True)
            command.assert_not_called()
            selector.run_browser(plan, "integration", main=True, extended=True)
            command.assert_called_once()
        with (
            patch.dict(
                "os.environ",
                {"LMEX_LORA_REFERENCE_MODEL_ROOT": "/invalid/supplied/reference"},
                clear=True,
            ),
            patch.object(selector.subprocess, "run") as command,
        ):
            selector.run_browser(plan, "integration", main=True)
            self.assertEqual(command.call_args.kwargs["env"]["LMEX_TEST_PORTFOLIO"], "full")
            selector.run_browser(plan, "integration", main=True, extended=True)
            command.assert_called_once()

    def test_lora_browser_selects_its_independent_reference_tcp_owner(self):
        plan = selector.select(["ui/acceptance/lora-reference.spec.ts"])
        self.assertEqual(selector.targets(plan, "integration"), ["lora-reference.spec.ts"])
        self.assertEqual(
            selector.network_targets(plan),
            [
                "acceptance/test_architecture.py",
                "acceptance/test_lora_reference.py",
                "acceptance/test_reference.py",
            ],
        )
        self.assertIn("integration", plan["extended"])
        selector.validate(plan)

    def test_http_only_edit_has_no_unchanged_browser_consumer(self):
        plan = selector.select(["acceptance/test_report.py"])
        self.assertEqual(plan["owners"], ["integration"])
        self.assertEqual(selector.targets(plan, "integration"), [])
        self.assertEqual(selector.network_targets(plan), ["acceptance/test_report.py"])
        selector.validate(plan)
        with self.assertRaises(ValueError):
            selector.validate(plan | {"integration": ["network:test_missing_file.py"]})
        with self.assertRaises(ValueError):
            selector.validate(plan | {"integration": []})

    def test_helper_unknown_deletion_and_portfolio_truth(self):
        harness = selector.select(["ui/acceptance/product-harness.ts"])
        self.assertEqual(len(selector.targets(harness, "integration")), 5)
        deleted = selector.select(["ui/tests/matrix-deleted.spec.ts"])
        self.assertEqual(deleted["browser"], ["full"])
        for path in [
            "future/unknown.py",
            "ui/tests/future/new.spec.ts",
            "backend/tests/conftest.py",
            "backend/tests/test_deleted.py",
        ]:
            plan = selector.select([path])
            self.assertEqual(plan["portfolio"], "routine")
            self.assertEqual(set(plan["extended"]), {"backend", "integration"})
        dependency = selector.select(["backend/src/llm_model_explorer/model_files.py"])
        self.assertIn("backend", dependency["extended"])
        self.assertEqual(selector.select([], full=True)["portfolio"], "full")
        with self.assertRaises(ValueError):
            selector.validate(selector.select([]) | {"portfolio": "quiet"})

    def test_matrix_retains_all_consumers_and_native_scroll(self):
        plan = selector.select(["ui/src/rendering/a.ts"])
        files = selector.targets(plan, "browser")
        for name in (
            "renderer.spec.ts",
            "matrix-zoom-pixels.spec.ts",
            "tensor-explorer-scrollbars.spec.ts",
            "tokenizer-embeddings.spec.ts",
            "architecture-inspection.spec.ts",
        ):
            self.assertIn(name, files)

    def test_full_owner_includes_new_nested_spec_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory, name in (
                ("ui/tests/future", "component.spec.ts"),
                ("ui/acceptance/future", "product.spec.ts"),
            ):
                path = root / directory / name
                path.parent.mkdir(parents=True)
                path.touch()
            plan = selector.select([], full=True)
            self.assertEqual(
                selector.targets(plan, "browser", root=root), ["future/component.spec.ts"]
            )
            self.assertEqual(
                selector.targets(plan, "integration", root=root), ["future/product.spec.ts"]
            )

    def test_empty_or_invalid_required_selection_fails(self):
        plan = selector.select(["ui/src/tokenizer/a.ts"])
        for value in ([], ["typo"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                selector.validate(plan | {"browser": value})
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            selector.targets(plan, "browser", root=Path(temporary))
        for path in ("../ui/a.ts", "/ui/a.ts", "ui//a.ts", "ui\\a.ts", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                selector.select([path])
        self.assertEqual(set(selector.select([])["owners"]), selector.OWNERS)


class GitDiffTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lmex-selector-git-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Selector tests")
        self.git("config", "user.email", "selector@example.invalid")
        self.write("ui/src/tokenizer/old.ts", "original content\n" * 10)
        self.write("ui/src/rendering/deleted.ts", "deleted\n")
        self.base = self.commit()

    def git(self, *args):
        return selector.git(*args, root=self.root).decode().strip()

    def write(self, path, content):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "test input")
        return self.git("rev-parse", "HEAD")

    def test_all_pushed_commits_rename_and_deletion(self):
        self.write("ui/src/architecture-explorer/new.ts", "new\n")
        self.commit()
        self.git("mv", "ui/src/tokenizer/old.ts", "ui/src/architecture-explorer/renamed.ts")
        (self.root / "ui/src/rendering/deleted.ts").unlink()
        head = self.commit()
        paths, diff_base = selector.diff_paths(self.base, head, event="push", root=self.root)
        self.assertEqual(
            paths,
            [
                "ui/src/architecture-explorer/new.ts",
                "ui/src/architecture-explorer/renamed.ts",
                "ui/src/rendering/deleted.ts",
                "ui/src/tokenizer/old.ts",
            ],
        )
        self.assertEqual(diff_base, self.base)
        plan = selector.github_plan("push", {"before": self.base, "after": head}, root=self.root)
        self.assertEqual(plan["paths"], paths)
        self.assertEqual(plan["context"]["tested_revision"], head)

    def test_pr_uses_complete_merge_base_diff_and_records_tested_merge(self):
        self.git("checkout", "-qb", "feature")
        self.write("ui/src/tokenizer/first.ts", "first")
        self.commit()
        self.write("ui/src/tokenizer/last.ts", "last")
        head = self.commit()
        self.git("checkout", "main")
        self.write("docs/operations.md", "base advanced")
        base = self.commit()
        self.git("update-ref", "refs/remotes/origin/main", base)
        self.git("merge", "--no-ff", "-qm", "tested merge", head)
        plan = selector.github_plan(
            "pull_request",
            {"pull_request": {"head": {"sha": head, "ref": "feature"}, "base": {"ref": "main"}}},
            root=self.root,
        )
        self.assertEqual(plan["paths"], ["ui/src/tokenizer/first.ts", "ui/src/tokenizer/last.ts"])
        self.assertEqual(plan["context"]["base"], base)
        self.assertEqual(plan["context"]["head"], head)
        self.assertEqual(plan["context"]["diff_base"], self.base)
        self.assertEqual(plan["context"]["tested_revision"], self.git("rev-parse", "HEAD"))

    def test_invalid_missing_force_push_unrelated_and_event_fallback(self):
        self.write("docs/a.md", "new")
        head = self.commit()
        for before, after in (("0" * 40, head), ("1" * 40, head), (head, self.base)):
            with self.subTest(before=before, after=after):
                plan = selector.github_plan(
                    "push", {"before": before, "after": after}, root=self.root
                )
                self.assertEqual(set(plan["owners"]), selector.OWNERS)
                self.assertIn("fallback", plan["reasons"][0])
        self.git("checkout", "--orphan", "unrelated")
        self.git("rm", "-rf", ".")
        self.write("README.md", "unrelated")
        unrelated = self.commit()
        with self.assertRaises(subprocess.CalledProcessError):
            selector.diff_paths(head, unrelated, event="push", root=self.root)
        for event, payload in (("pull_request", {}), ("push", {}), ("unknown", {})):
            self.assertEqual(
                set(selector.github_plan(event, payload, root=self.root)["owners"]), selector.OWNERS
            )

    def test_manual_and_final_epic_pr_are_full(self):
        self.git("update-ref", "refs/remotes/origin/main", self.base)
        self.write("skills/test-quality/references/repository-testing.md", "docs")
        head = self.commit()
        plan = selector.github_plan(
            "pull_request",
            {
                "pull_request": {
                    "head": {"sha": head, "ref": "codex/epic-issue-278"},
                    "base": {"ref": "main"},
                }
            },
            root=self.root,
        )
        self.assertEqual(set(plan["owners"]), selector.OWNERS)
        self.assertEqual(
            set(selector.github_plan("workflow_dispatch", {}, root=self.root)["owners"]),
            selector.OWNERS,
        )

    def test_malformed_diff_records_fail_explicitly(self):
        for raw in (b"M\0ui/src/a.ts", b"R100\0old.ts\0", b"invalid\0file\0"):
            with (
                self.subTest(raw=raw),
                patch.object(selector, "git", side_effect=[b"", b"", self.base.encode(), raw]),
                self.assertRaises(ValueError),
            ):
                selector.diff_paths(self.base, self.base, event="push", root=self.root)


if __name__ == "__main__":
    unittest.main()
