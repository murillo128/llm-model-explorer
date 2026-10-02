# Domain matrix ownership and compact navigation fixtures

This is observed test evidence for issue #274, relative to
`db37bf250f6b2d07671fceee22a23d3cf25fee28` after #273's project selection.
The independent temporal oracle is the Tensor Explorer specification's
consecutive-event 180 ms rule. `TensorViewport` now uses `WheelGesture` for that
decision; camera application, wheel units/modifiers, native scrolling, public
viewport methods and all six interruption/reset sites keep their behavior.
`CameraHistory` remains the camera-state owner. No mocked canvas claims rendering.

## Exact old-to-new proof map

| Original case / inputs | Primary owner and retained browser proof | Defect detection |
| --- | --- | --- |
| `matrix-zoom.spec.ts`: `wheel gesture timing: 179 ms gaps coalesce across 358 ms`, `180 ms gaps coalesce across 360 ms`, `181 ms gap splits after the second event`, each at DPR 1/1.25/2 | Three literal schedules in `wheel-gesture.test.ts`, using the runtime policy and real `CameraHistory` with independently authored scale/origin transitions. One browser `wheel and trackpad pinch coalesce into separate gesture-level camera states` bridge retains DOM delivery, changed real camera and exact Escape restoration. | Inclusive/exclusive threshold, measuring from gesture start, missed intermediate undo state. Nine browser invocations become three unit cases. |
| Same file: `wheel and trackpad pinch coalesce into separate gesture-level camera states`, DPR 1/1.25/2 | One DPR-1 browser case additionally delivers native `page.mouse.wheel` and restores it with actual focused Escape. Scheduled wheel `[0,40,80]` and ctrl-wheel `[300,340,380]` still traverse viewport/renderer/history. Literal 220 ms gap and interruption/reset unit cases supplement it. | Separate burst origins, viewport-to-policy wiring, local restore. The modifier itself does not split a gesture (historical `4aa9aef` evidence). Three browser invocations become one; actual touch ownership remains in the touch browser case. |
| `architecture-browser.spec.ts`: long-label tail of `browser and Model cards share compact chrome while contextual controls remain available`, desktop+narrow | Same browser assertions, existing four-instance explicit graph; `layer-31.attention.Q` becomes nonzero last `layer-3.attention.Q`. Both are full-attention instances with the same Q/K/V, state and nested-boundary structure. | Wrong clicked identity/parent, truncated disclosure, header displacement. Keep real DOM width, pointer and title assertions. |
| Same file: `section disclosure restores focus and pre-search presentation with empty and long rows bounded`, desktop+narrow | Same browser assertions and four-instance graph/last Q target. | Lost focus on section collapse, lost pre-search section state, clipped long row. |
| `architecture-controls.spec.ts`: `browser disclosure reveals only the hidden exact instance and preserves the compact window` and `...multi-instance window`, desktop | Six-instance explicit graph; exact hidden `layer-10` becomes last `layer-5`, outside the four-instance window. Both are linear-attention instances. Keep generated layout, window equality, exact visible IDs, expansion and source-record/edge traceability assertions. | Historical `46ff8e7` regression: revealing one instance must not replace a retained window. |

Compact graphs still contain first, nonzero and last identities, repeated labels,
distinct parameters, asymmetric branches, state and nested module boundaries.
No generated geometry is used as a fixture oracle. Mixed stacks and independently
retained windows, same-shaped ports, K/V, fan-out, shared/concrete bindings and
adapted projections retain their existing families unchanged.

The following size-sensitive browser families remain on 48 instances:
`pane collapse/restore and bounded resize preserve mounted canvas, filter and
scroll; storage failure is safe`, `clearing search restores tree scroll and
retains explicit filtered expansion`, and `unrelated model growth leaves a small
isolated component readable and its bounds unchanged`. Full-size reference,
expert completeness, memory/layout and first/last-at-scale cases are unchanged.
Fractional-DPR/pixel, native scroll, pointer capture/cancel, focus, context-loss
and GL lifetime assertions remain at their browser/renderer owners.
At the #274 revision, #269/#270 and CLM/Kev/formula/port-order cases were unchanged. This audit found
no faithful cheap replacement for their native layout/event claims.
The later user-authorized #277 consolidation separates real package/composition
TCP proof from generic graph presentation and removes the same-generator dense
28-layer stress duplicate while retaining the larger 30-layer case. Current
component collection is 520 (370 desktop, 138 narrow, 12 native scrollbars).
See the [current ownership map](../../acceptance/integration-test-ownership.md#architecture-and-package-families);
the historical #274 measurements below remain unchanged.

## Negative controls

All faults were applied only in `/tmp/issue-274-fault`, then restored; none is in
published source. The unmodified focused comparison passed first.

| Temporary production fault | Intended failing assertion |
| --- | --- |
| Cutoff 179 instead of 180 | The 180 ms unit case fails. |
| Cutoff 181 instead of 180 | The 181 ms unit case fails. |
| Keep gesture-start time instead of advancing last-event time | Both long 179/180 ms bursts fail. |
| Pass a fresh object from the viewport for each wheel event | Retained browser bridge fails its exact three-camera Escape sequence. |
| Reintroduce `46ff8e7`'s prior exact-instance window replacement at current browser toggle | Both compact/multi-instance browser cases fail retained-window equality: actual start 5/count 1. |
| Omit pre-search section restoration in the real browser component | Compact section case fails after Clear search: Expand Model section is absent. |
| Resolve the clicked Q row to its containing attention group | Compact long-label case fails: selected title is Full attention instead of the exact long Q label. |

An initial identity-fault edit had a syntax error; that diagnostic is excluded.
The corrected fault reaches the intended selection/title assertion above.

## Measured cost

Same host, Node 24.14.0/npm 11.9.0, locked Playwright 1.63/Chromium 153,
SwiftShader, one worker, zero retries, unchanged browser arguments and isolated
ports/output directories. Both measurements use the four affected graph cases
with their existing project selection, plus all old timing cases or their one
retained bridge. Both runs passed without skips/flakes. The single locked
dependency installation is shared; it is outside both comparisons.

| Phase | Before | After |
| --- | ---: | ---: |
| Production build wall | 0.43 s | 0.37 s |
| Browser wall, including servers/setup/teardown | 43.64 s (18 invocations) | 29.16 s (7 invocations) |
| Sum of native browser attempt durations | 41.272 s | 27.064 s |
| Before Hooks sum | 3.099 s | 2.773 s |
| After Hooks sum | 1.206 s | 0.247 s |
| Body remainder excluding those hooks | 36.967 s | 24.044 s |
| Vitest wall including startup/environment/setup | 0.96 s (6 existing cases) | 1.02 s (11 cases) |
| Vitest test-function duration sum | 0.022 s | 0.021 s (new temporal file 0.004 s) |
| Build + browser + unit wall | **45.03 s** | **30.55 s** |

These single-run scoped measurements include the new unit cost and the stronger
native bridge; they are not a whole-suite/pipeline speedup estimate. Native
collection changes 532 to 521 invocations: desktop 382 to 371, narrow 138 to 138,
headful native-scrollbars 12 to 12. Only temporal multiplication is removed;
#273's prior project savings are not counted again.

## Reproduction and validation

From `ui/`, with the locked Node PATH:

```sh
npm test -- wheel-gesture.test.ts camera-history.test.ts matrix-camera-navigation.test.ts \
  geometry.test.ts zoom-selection-geometry.test.ts projection.test.ts \
  scope-navigation.test.ts auto-layout.test.ts repeated-layout.test.ts interfaces.test.ts
npm test -- CardSummary.test.tsx
npm run typecheck
npm run lint
npm run build
UI_TEST_PORT=29540 PLAYWRIGHT_WORKERS=1 npm run test:browser -- \
  matrix-zoom.spec.ts matrix-zoom-selection.spec.ts matrix-zoom-pixels.spec.ts \
  matrix-overlay-scrollbars.spec.ts matrix-centering.spec.ts renderer.spec.ts \
  architecture-browser.spec.ts architecture-controls.spec.ts architecture-isolation.spec.ts \
  architecture-connections.spec.ts architecture-interfaces.spec.ts architecture-card-actions.spec.ts
```

Focused unit validation: 117 passed plus 22 CardSummary cases. Typecheck, lint
and production build pass. All 218 selected desktop/narrow browser cases passed
in 600.439 s, with zero skips, retries or flakes. Validated source/test hashes
match the published implementation; the completed evidence update changes only
this report.
The full ordinary browser suite and real checkpoint acceptance were not rerun;
unchanged size-sensitive coverage is not claimed as freshly executed wholesale.
Raw native reports, listings, hook accounting and fault outputs stay outside Git
under `/tmp/issue-274-evidence/`.
