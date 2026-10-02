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
                ["full"],
            ),
            (
                "ui/src/tokenizer/editor.ts",
                {"ui", "browser", "integration"},
                ["tokenizer"],
                ["product"],
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
            "package-lock.json",
        ):
            with self.subTest(path=path):
                plan = selector.select([path])
                self.assertEqual(set(plan["owners"]), selector.OWNERS)
                self.assertEqual(plan["browser"], ["full"])
                self.assertEqual(plan["integration"], ["full"])

    def test_union_and_new_native_cases(self):
        plan = selector.select(["ui/src/tokenizer/a.ts", "ui/src/architecture-explorer/a.ts"])
        self.assertEqual(plan["browser"], ["architecture", "tokenizer"])
        self.assertEqual(plan["integration"], ["full", "product"])
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
