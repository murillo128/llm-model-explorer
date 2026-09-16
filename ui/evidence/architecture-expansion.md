# Component expansion validation

Validated on 2026-09-16 with Node 24.14.0, Chromium/SwiftShader, desktop
1440×900 and narrow 390×844. These authored fixtures establish UI behavior,
not installed-checkpoint acceptance.

## Observed results

- `npm run check`: API bindings, TypeScript, lint, 963 tests across 40 files,
  and production build passed (`VITEST_MAX_WORKERS=2`).
- `architecture-card-actions.spec.ts`: 16 browser cases passed, including real
  pointer double-click sequences and contraction after Show all operations.
- `architecture-inspection.spec.ts`: 16 browser cases passed, preserving exact
  progressive values, native matrix camera, cancellation and resource cleanup.
- Architecture graph, controls, connections, camera, isolation, components and
  shell suites: 92 browser cases passed.

Selection on title/body leaves layout count, camera, expansion, scope and focus
unchanged. The integrated shell's fetch counter remains unchanged through card
selection/expansion and explicit metadata inspection; no numeric request starts.
Each component/control double-click changes layout count by exactly one when
expandable, zero for leaves/empty groups, and never opens inspection implicitly.
The nested contraction check retains zoom and the acted-on origin within one
CSS pixel, without a Fit view command during the measured transition.

Hidden source selection remains exact through parent contraction and explorer
remount. Nested expansion returns on reopening; Back restores the prior scope.
Two mounted test consumers share selection and toggles through `useGraphView`
and `component-actions`, including actions issued before projection replacement.
Unit tests also retain sibling/derived state, distinguish ranges from concrete
instances, invalidate replacement graphs/models and restore exhaustive topology.

## Reproduction

From `ui/`, with an unused port pair:

```sh
VITEST_MAX_WORKERS=2 npm run check
UI_TEST_PORT=45161 PLAYWRIGHT_WORKERS=1 npm run test:browser -- tests/architecture*.spec.ts --project=desktop --project=narrow
```

One initial narrow test setup attempted an offscreen card gesture. It now uses
the existing explicit Fit view control before measurement; the corrected case
passed on both viewports without changing the anchoring assertions.
