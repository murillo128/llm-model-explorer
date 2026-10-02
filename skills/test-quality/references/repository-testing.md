# LLM Model Explorer testing boundaries

The accepted contract lives under `docs/spec/`; tests establish observed behavior,
not permission to override that contract. Inspect the relevant backend, API and
UI specification plus current package/workflow configuration before choosing
checks. Keep tests with the component that owns the behavior.

## Backend and API

The Python backend uses pytest. From `backend/`, a focused invocation is
`uv run --locked pytest tests/<test_file>.py -q --durations=25`; replace the path
with an existing selected file. The primary Python 3.12 CI owns selected routine tests plus
affected extended proof, Ruff, mypy and isolated installed-wheel coverage.
Ordinary main Python 3.14 uses a real eighteen-case installed-wheel/CLI/runtime
smoke; runtime/dependency/packaging changes and manual mode select fuller proof.
This is reduced default cross-version confidence, not a supported-version change. The wheel test outside the source tree protects distribution and
must not be removed as a duplicate import test.

Use small deterministic tensors, temporary model directories/cache roots and
independent expected values. Test dtype, shape, indexing, numerical tolerances,
stream failure timing, cancellation and shared-consumer lifetime where required
by the accepted contract, not just where source branches happen to exist. A fake
reader may isolate an unrelated cost but cannot establish real storage/quantization
compatibility. Do not download full models or require CUDA for ordinary unit tests;
CPU fixture success is not optional CUDA or full local-model acceptance.

`api/validate_contract.py` and checked fixtures protect the independent public
contract; `ui/scripts/api-generator/generate.mjs --check` protects generated UI
bindings. Use the configured API environment, not whichever system Python is
available. Generation followed by a clean diff proves reproducibility, not that
both consumers implement the contract. Check consumers and integration when a
normative API, binary format, fixture or generator changes. Do not silently infer
that a missing workflow run makes consumer validation unnecessary.

## UI and real boundaries

From `ui/`, `npm test -- <test-file>` selects Vitest tests;
`npm run typecheck`, `npm run lint` and `npm run build` supply their distinct checks.
`npm run check` already aggregates API binding checks, types, lint, unit tests and
build: avoid repeating its components without a reason. `npm run test:browser --
<spec-file>` selects Playwright browser tests; `npm run test:acceptance --
<spec-file>` selects the separate acceptance configuration. Confirm current files,
projects, ports and backend prerequisites before running either browser command.

Pure transforms and component interactions belong in focused Vitest tests when
that boundary is faithful. Use a real browser for layout, native scrollbars,
pointer/zoom behavior and WebGL resource/rendering claims that jsdom cannot prove.
A line-count, CSS-string or mocked canvas assertion is not real rendering proof.
Expected cell identity or numerical values must not come from the renderer under
test. Retain distinct browser/DPR/platform coverage unless an authorized audit
shows that its independent risk remains covered.

[Integration acceptance](../../../acceptance/check-integration.sh) exercises real
backend/network and production-browser boundaries. Read
[acceptance guidance](../../../acceptance/README.md) for environments and capability
limits; do not replace this path with mocked API responses. For shared-operation
cancellation, for example, protect the invariant at its backend owner and keep a
transport test when disconnect handling presents an additional risk. Do not replay
the whole domain matrix in every browser scenario.

## CI selection, isolation and evidence

The existing workflows separately own backend, API, UI, application acceptance and
Skillforge runner checks. The product workflows share
[the repository-local selector](../../../.github/scripts/validation_selector.py);
selection runs before conditional steps with complete exact Git diffs. See
[the ownership map and invocation contract](../../../acceptance/validation-routing.md).
Normative specs, API/generated inputs, examples and unknown/shared configuration
reach their consumers. Pure operational Markdown uses documentation/infrastructure
owners. Invalid diff context falls back to the full mandatory routine portfolio and
potentially affected extended owners; invalid plans or
empty required groups fail. Fast checks remain full for each selected owner.

Narrow test routes use explicit reviewed leaves, not filename/existence alone.
Shared helper imports can cross from backend tests into real TCP/browser fixture
preparation, including function-local/embedded imports and reference oracles.
Keep the finite consumer map current and union all owners. Known Python-only
native-package inputs stay browser-free; shared network support also serves the
architecture browser oracle. New unclassified existing test paths remain broad.

Preserve accepted epic-child local validation and the final aggregate CI boundary.
Native configuration separates routine, affected extended and explicit full
portfolios; ordinary untagged tests remain routine. `acceptance/check.sh` is always
complete. `check-integration.sh --routine --main-ci` is the combined routine gate;
without `--routine` or a plan it keeps full integration. CI's `--main-ci` delegates
API/generated drift to their independent applicable exact-target workflows on
PRs and main; missing selected gates cannot be treated as success. Aggregate epic
PRs select all mandatory routine and affected extended proof; manual dispatch is
full. Focused commands consume an explicit complete path list, never a commit title.

Both Playwright defaults are trace-free, with zero retries, native JSON/failure
screenshots/logs and compact diagnostics. Diagnose one named failure with native
`--trace on`; do not replay a suite automatically. Full/manual execution retains
all non-retired `@extended` cases and configured reference/CUDA checks. Backend
`pytest -m extended` retains genuine 2**24+1 quantiles and Linux allocator proof;
analysis/materialization/dependency or unknown impact must select them. Long
architecture lifetime retains sixteen cycles in its targeted extended owner.

The plan resolves reference-enabled effective full before setup/phase selection;
do not run its extended cases a second time. The declared extended-only LoRA file
has no routine browser command, while its TCP and extended owners remain required.
Native reports/output use separate routine/extended/full/diagnostic directories
under the owned evidence root (or `ui/test-results/acceptance/` locally). A later
failure must preserve earlier finalized reports. Explicit diagnostic trace runs
use the diagnostic destination. Unexpected empty native discovery still fails.

Issue #277 explicitly authorizes risk-based low-value retirement. Record exact
removed permutations, retained independent owners or plausible lost defect
detection and measured/unmeasured cost in the existing ownership/cost reports.
Keep known failures, invalid-input/privacy/data-corruption guards, genuine
threshold sizes and numerical tolerances. Do not infer equivalent coverage from
case counts, mocks or equality between two subject outputs.

Use isolated caches, ports and artifact paths for concurrent checks. Do not edit a
checkout while tests run or reuse another revision's UI build as acceptance proof.
Reproduce asynchronous behavior with controlled events/conditions, not arbitrary
sleeps or retries that conceal defects. Keep large model files and generated
artifacts outside Git. Runtime, retries and setup costs need comparable evidence
before claiming CI savings.

Infrastructure tests use the existing `.github/scripts/test_*.py` unittest files
and executor-routing workflow. Inspect the current revision before selecting
suites: runner synchronization is merged; verify current host capability separately
from repository code. Fake Devin tests do not verify the user's installed CLI or
OS sandbox. Documentation-only
skill changes need scoped diff, link/frontmatter/routing and scenario review, not a
new product suite or source-keyword tests pretending to measure agent behavior.

Mutation testing and broad cleanup remain optional separately scoped work. This
reference introduces no new dependency, required percentage or scheduled job.
