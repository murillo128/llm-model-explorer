# Camera initialization repair

The original UI CI failure compared identical source graphs with and without
optional template metadata. Overview and exhaustive cameras differed only by
10 pixels vertically; their boxes, ports, routes and explicitly fitted isolated
views matched. The failing head was `f818862ac684a1e9239312c7c6d1115fa6bbc21f`,
with base `c8ead9ca8485347a74097f0253d51e47991c36c3`.

## Causal reproduction

On both revisions, a browser probe held the real React Flow viewport
ResizeObserver notification until after the layout reply and initial fit. It
recorded source graph/layout generation, loading presence, actual DOM size,
consumed renderer size, node readiness, and requested/completed camera actions.
The worker and camera math remained real. Both template variants behaved alike:

| Event | Actual viewport | Renderer viewport | Camera |
| --- | --- | --- | --- |
| Loading | 1440 × 718 | 1440 × 718 | Initial |
| Initial fit completed after loader removal | 1440 × 738 | 1440 × 718 | (177, 41), zoom 1 |
| Resize notification released | 1440 × 738 | 1440 × 738 | Still (177, 41), zoom 1 |

[Compact before measurements](initialization-before.json) retain all four
base/head × template-variant results. The displacement is exactly half the
20-pixel loader displacement; fitting with current geometry gives (177, 51).
This is a production initialization race present on the base, not merely an
assertion sampling an unfinished but otherwise correct fit.

The committed `architecture-camera.spec.ts` adds controlled worker, viewport
resize and camera-promise delivery. Its four template/remount tests fail against
the archived base with the test-only harness/probe applied: `aria-busy` becomes
false while the renderer size is stale. They pass with the repair. Tests use
Chromium/SwiftShader, two workers, no automatic retries, and actual 1440 × 900
and 390 × 844 project viewports (no forced desktop override).

## Correction and oracles

The loading label overlays the Architecture canvas rather than taking flex
space. A local lifecycle helper waits for the current layout's fixed boxes and
current nonzero DOM/renderer viewport dimensions to agree. The existing layout
box dimensions are read from React Flow's existing authored width/height even
for culled nodes, avoiding a wait for offscreen cards to mount when changing
scope. No layout coordinates,
routes, projection algorithms or ELK inputs change.

Lifecycle fits use React Flow's existing fixed-node bounds and
`getViewportForBounds` with the same padding/zoom limits, followed by its
zero-duration `setViewport`. This avoids React Flow's deferred `fitView` node
queue applying an obsolete generation. Readiness awaits completion, and cleanup
fences old completions. Saved camera/Back restoration has precedence; explicit
camera interaction supersedes pending initialization. A bounded, recoverable
failure releases readiness if geometry/completion cannot become usable.

Tests cover saved-camera remounts, delayed size and completion, scope replacement,
Back, model replacement, user zoom, and unchanged camera/layout on
selection/hover/search/menu/ordinary resize. The optional-template test still
compares exact cameras, boxes, ports, routes and layout counts for overview,
exhaustive expansion and isolation. It neither adds a Fit to the first two
states nor removes or relaxes camera equality. Its readiness now observes the
layout/camera lifecycle; animation frames only flush temporary pointer emphasis.

The first corrected CI head (`c694368c6a75f8142186dc5cfed1eaecd76df35b`)
passed the exact camera equivalence check but exposed a separate regression in
the existing nested-expansion browser test: the `linear0:out → layer1:in`
connection was absent after exhaustive expansion. The retained trace reported
React Flow's missing `target:in` handle while the card's input port was present.
Supplying `measured` box dimensions had preserved a stale internal handle list
when ports changed without changing a card's size. Readiness now uses the
already-authored width/height and leaves `measured` to React Flow, preserving its
normal handle invalidation. The existing connection assertion remains intact
and is included in repeated desktop/narrow validation.

Reproduce from `ui/`:

```sh
UI_TEST_PORT=46860 PLAYWRIGHT_WORKERS=2 npm run test:browser -- \
  --project=desktop --project=narrow architecture-camera.spec.ts \
  architecture-connections.spec.ts architecture.spec.ts \
  --grep 'current viewport|cold fit|pending scope|late camera|optional template metadata|nested expansion' \
  --repeat-each=3
npm run check
```

The test attachments retain requested/completed transforms and dimensions.
Generated browser reports and traces remain outside Git. Fixture evidence does
not establish model checkpoint or GPU acceptance.
