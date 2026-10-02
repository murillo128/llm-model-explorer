# Validation ownership and change routing

`.github/scripts/validation_selector.py` is the single conservative map used by
the four product workflows and the selected integration entrypoint. It reads
complete normalized paths, returns finite owners and reasons, and resolves native
Playwright files. Test titles and commit messages never determine selection.

Full discovery includes nested spec files. Acceptance uses the native
`**/*.spec.ts` matcher, with new untagged cases owned by DPR 1. Selected CLI file
filters escape literal paths because Playwright interprets them as regular
expressions; special filename characters cannot silently omit a selected file.

| Surface | Fast owners | Expensive owners |
| --- | --- | --- |
| `ui/src/architecture-explorer/` | Full UI check | Every `architecture*.spec.ts`; complete TCP/product acceptance including native weight modal, CLM, Kev and LoRA |
| `ui/src/rendering/`, `ui/src/matrix-explorer/` | Full UI check | Matrix/tensor/renderer/distribution/magnifier, tokenizer/embedding and architecture component files; complete integration |
| `ui/src/tokenizer/` | Full UI check | Every `tokenizer*.spec.ts`; real network/embeddings/polish/distribution/reference HTTP owners and complete `product.spec.ts` |
| Shared UI app/components/explorers/styles/runtime config | Full backend/API/UI checks | Complete component browser and integration |
| `backend/` | Existing full backend check, including wheel smoke | Complete integration; no fragile backend filename narrowing |
| `api/`, `docs/spec/api/`, generated binding/generator inputs | Full API/backend/UI checks | Complete component browser and integration |
| `examples/` | Full backend/native exporter checks and UI check | Architecture component files and complete real integration |
| Tests, fixtures, helpers, workflow/config/locks, all normative specifications | Full API/backend/UI checks | Complete product gates |
| Operational Markdown and known runner infrastructure | Existing documentation/executor infrastructure checks | Explicit product non-applicability |
| Unknown input or invalid/unavailable diff | Full API/backend/UI checks | Complete product gates |

Selections are unions. Broad groups dominate by resolving to a deduplicated file
set. New ordinary component cases run in desktop; narrow remains `@responsive`,
and native scrollbars keep their dedicated headed project. Product DPR 1 owns all
ordinary cases, DPR 2 keeps `@density`. PRs retain their existing desktop/native
and DPR-1 policy; main/full local runs retain all configured projects. Renderer
files keep their independently configured fractional DPR cases. No assertions,
timeouts, worker budgets, retries or optional capability rules change.

The coarse backend, example, shared and architecture integration selections are
intentional: shared startup and native weight streams cross those boundaries.
Routing does not claim that a filename-level map can isolate individual semantics.

## Revision and failure contract

All four workflows run cheap selection before conditional test steps, without
path filters that could prevent inspection. Epic-child base exclusions and the
same-repository guard before self-hosted jobs remain. Checkouts use full Git
history with no persisted credentials. PR selection diffs the current fetched
base's merge-base against the exact PR head and records the tested head/merge.
Push selection diffs actual `before` through `after`, including every pushed
commit, deletions and both sides of renames. No file-list API truncation applies.

Missing commits, non-forward pushes, unrelated history, malformed paths/events or
mismatched merge context select the full product gate with a visible reason.
Invalid explicit plan files and empty required groups fail before checks. Native
pytest/Playwright still fail empty execution. No `--pass-with-no-tests` is used.

Each historical logical check remains one job. Selected mandatory commands run
under failure-propagating shell/Python execution; a failed/cancelled job cannot
be aggregated into success. There is no conditional-job aggregator. An unselected
owner is explicitly non-applicable in the normal job summary, which includes the
plan, reasons and exact revision context. This does not claim tests executed.

## Commands and full mode

From the repository root after the documented dependency setup:

```sh
# Complete local proof and aggregate evidence, independent of routing:
bash acceptance/check.sh
# Complete integration only (main policy keeps all configured DPR projects):
bash acceptance/check-integration.sh --main-ci
# Focused routes from a complete operator-supplied path list:
printf '%s\n' '["ui/src/tokenizer/TokenizerExplorer.tsx"]' > /tmp/changed-paths.json
python3 .github/scripts/validation_selector.py --paths /tmp/changed-paths.json --output /tmp/validation-plan.json
(cd ui && npm run check)
python3 .github/scripts/validation_selector.py --plan /tmp/validation-plan.json --run browser --main
bash acceptance/check-integration.sh --main-ci --plan /tmp/validation-plan.json
# Cheap orchestration tests, including actual workflow shell invocations:
api/.venv/bin/python -m unittest discover -s acceptance -p test_validation_selector.py
api/.venv/bin/python -m unittest discover -s acceptance -p test_ci_entrypoints.py
```

Omit `--main`/`--main-ci` to reproduce PR project policy. The path JSON is an
explicit complete input, not a way to validate a partial list as a whole PR.
`--github` consumes the Actions event and exact Git objects; `--full` produces a
complete plan. Every workflow has `workflow_dispatch`, which always selects full
coverage. A final PR whose head is `codex/epic-issue-*` also forces full coverage,
including when only operational files changed. Main integration continues to
delegate API/component checks to their independent workflows.

Reference SmolLM2 Base, complete architecture references, LoRA pairs, CLM/Kev
native reference availability and CUDA remain governed by the existing acceptance
documentation. Missing optional capabilities are reported as skips, never passes.
