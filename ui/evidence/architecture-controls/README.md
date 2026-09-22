# Architecture control organization

Issue #197 moves graph commands out of View options. The header retains selection
and global commands, contextual controls retain stack/instance/focus navigation,
and a fixed lower-left dock owns zoom and fit. Active filters remain in the
existing header row to preserve narrow canvas height. The preferences panel contains only
persistent display settings. Derived MLP eligibility comes from the existing
source-graph detector independently of its enabled state.

Graph identity, declared scope and coverage join the existing Model information
and diagnostics destination. The source browser with explicit Center retains
outside-isolation navigation. No projection recognition or camera algorithm changed.

## Reproduction

From `ui/`, with the repository's pinned Node version and installed dependencies:

```sh
npm run check
PLAYWRIGHT_WORKERS=2 UI_TEST_PORT=24227 npx playwright test \
  tests/architecture-controls.spec.ts tests/architecture-overview.spec.ts \
  tests/model-diagnostics.spec.ts tests/architecture-isolation.spec.ts \
  tests/architecture-interfaces.spec.ts tests/architecture-connections.spec.ts \
  tests/architecture-browser.spec.ts tests/architecture-camera.spec.ts \
  tests/architecture.spec.ts --project=desktop --project=narrow
```

Checks use deterministic local graph fixtures at 1440×900 and 390×844 with Chromium
and SwiftShader. They do not establish full local reference-checkpoint acceptance.

## Evidence

The control suite observes worker requests/replies and camera transforms, checks
source topology and parameter preservation, and exercises keyboard focus, outside
interaction, menu dismissal, MLP eligibility and reversible grouping. The overview
suite retains exact single collapse-and-reframe completion coverage. The diagnostics
suite uses the production application to verify complete rediscovery and graph
metadata without refetch or numeric work. Isolation, interfaces, connections and
browser checks cover outside-source reveal, exact shared-instance binding, hidden
selections, state focus and provenance.

Screenshots ([desktop controls](desktop-controls.png), [narrow controls](narrow-controls.png),
[desktop derived MLP](desktop-derived-mlp.png), [narrow derived MLP](narrow-derived-mlp.png))
are compact captures from the control suite: default preferences and
the eligible derived-MLP preferences in desktop and narrow viewports. Trace/log
outputs stay outside Git.

The broader run exposed a pre-existing setup assumption in the 24-instance
connection scenario: it expected a repeated stack in the initial one-level Model
view. Both desktop and narrow runs fail at that same first assertion on pinned
base `b7fb81349831506a9242cb33ce526344db4d3d80`. The scenario now explicitly opens
the source Language model and fits its compact stack before its existing
connection/state/exhaustive assertions. The initial overview remains covered by
the dedicated overview suite.

The shared Chromium cache disappeared between projects during the broader run.
Interrupted checks were rerun with the pinned Chromium headless shell installed
under a task-local temporary `PLAYWRIGHT_BROWSERS_PATH`; no repository or CI
configuration changed.
