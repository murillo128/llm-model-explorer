# Synchronized architecture browser

This evidence covers the combined Architecture Explorer UX on the epic integration
base, including component navigation, shared expansion, metadata cards and direct
matrix inspection. It uses the existing deterministic fixture graphs and streams;
it does not establish full local reference-checkpoint acceptance.

The browser derives ordered Model rows and Shared families from the received graph.
Names select source identities without reveal; disclosure and double-click use the
canvas expansion controller. Center opens required ancestors and exact repetition
windows. Outside-scope commands preserve a Back snapshot. A common operation in
structure-only mode is distinguished from a concrete source selection without
selecting instance zero or enabling weight actions.

Component disclosure preserves existing repetition windows. Expanding a hidden
instance adds only that exact instance alongside the current window; contraction
uses the same canvas projection and restores the prior compact context. Regression
coverage compares browser and canvas disclosure from a four-instance desktop
window (two on narrow screens), and opens/closes instance 10 from both a compact
overview and an existing window. It checks retained siblings, exact source/port
provenance and the restored projection. These three cases reproduced the window
overwrite before the correction.

The pane retains its mounted browser/canvas, query, separate tree/search scroll and
preferred width across collapse. Width/visibility use optional local preferences;
graph-backed state follows the existing backend/model/graph view lifetime. At
narrow widths the expanded browser occupies a short upper row; collapse restores
the same 40-pixel rail beside a full-height canvas.

## Validation

Local checks used an isolated snapshot on integration base
`bb556491c5ba420aa49895bdb831f166c76572b4`, with identical candidate UI source and
test files. No model download or GPU was required.

| Check | Result |
| --- | --- |
| `npm run check` | Passed: generated API bindings, typecheck, lint, 43 unit-test files / 985 tests, production build |
| Initial complete Architecture Explorer and Tensor inventory browser suite, narrow | 76 passed |
| Initial desktop camera, family, repetition, shared-instance and isolation revalidation | Passed |
| Final browser/control regression suite, desktop and narrow (including six repetition-window cases) | 38 passed |
| Initial built UI + real backend, four architecture fixtures and cancellation/shared-instance scenarios, DPR 1 and 2 | 12 passed |
| Final built-shell safety at 390/1178/1440 pixels and shared-instance lifetime checks, DPR 1 and 2 | 8 passed |

The PR's UI check supplies complete desktop and native-scrollbar regression
coverage on the published commit; its application check runs the integrated
product gates. Both must pass on the current commit before the review-ready
handoff. The local narrow run also verifies the complete synthetic graphs for
SmolLM2, Qwen3, Qwen3.5 and V-JEPA2, independent of checkpoint acceptance.

The first local production rerun encountered a port collision before DPR 2 model
selection: the backend port served HTML instead of `/models` JSON. The run was
stopped and repeated with the unchanged snapshot on a verified free port range.

The production lifetime check performs eight concrete-instance switches after
weight inspection/cancellation. Each final sample has zero active workers, two
total worker/layout invocations, two retained source graphs and one retained
layout; switching leaves layout count and camera unchanged and sends no requests.
Heap diagnosis identified stale Canvas closures in detached toolbar props and
the worker effect's cleanup. Toolbar/browser event proxies and the separately
scoped worker hook release those references without relaxing the memory bound.
Temporary heap instrumentation and snapshots are not repository artifacts.

## Reproduction

Use the repository's locked Node 24.14/npm dependencies and existing Python 3.12
backend environment. One heavy suite runs at a time with isolated ports.

```sh
cd ui
npm run check
UI_TEST_PORT=4320 npm run test:browser -- 'tests/architecture.*spec.ts' \
  tests/tensor-inventory.spec.ts --project desktop --project narrow
UI_TEST_PORT=4370 npm run test:browser -- tests/architecture-browser.spec.ts \
  tests/architecture-controls.spec.ts --project desktop --project narrow
```

The production acceptance uses the existing harness, real HTTP backend, built UI,
SwiftShader, and local synthetic model fixtures:

```sh
UI_TEST_PORT=4340 xvfb-run -a npm run test:acceptance -- architecture.spec.ts \
  --project dpr1 --project dpr2 \
  --grep 'deterministic production.*full graph|close during progressive|shared structure production'
UI_TEST_PORT=4490 xvfb-run -a npm run test:acceptance -- \
  --project dpr1 --project dpr2 \
  --grep 'architecture safety baseline|shared structure production'
```

On this host Xvfb requires the Mesa EGL vendor selection:
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json`.
The local runs set `PYTHONPATH` to the same snapshot's `backend/src`. Model files,
caches, complete logs, screenshots and traces stay outside Git except the selected
compact screenshots linked below.

## Captured interaction states

The fixture harness keeps its own fixture picker above the workspace. Desktop
captures use 1440 × 900; narrow captures use 390 × 844. Separate built-shell tests
cover the real model/session controls and explorer tabs, including 280-pixel width.

| State | Desktop | Narrow |
| --- | --- | --- |
| Expanded browser; no selection | [Image](browser-expanded-desktop.png) | [Image](browser-expanded-narrow.png) |
| Collapsed 40-pixel restore rail | [Image](browser-collapsed-desktop.png) | [Image](browser-collapsed-narrow.png) |
| Filtered exact source selection; unchanged canvas | [Image](browser-filtered-selection-desktop.png) | [Image](browser-filtered-selection-narrow.png) |
| Shared family with ordered concrete instances | [Image](browser-shared-instances-desktop.png) | [Image](browser-shared-instances-narrow.png) |
| Nested isolated component | [Image](browser-nested-isolation-desktop.png) | [Image](browser-nested-isolation-narrow.png) |
| Metadata card and direct matrix action | [Image](metadata-card-desktop.png) | [Image](metadata-card-narrow.png) |
| Native matrix inspection with exact indexed magnifier | [Image](matrix-desktop.png) | [Image](matrix-narrow.png) |

The tiny matrix fixture has six real scalar cells at native scale; the magnifier
shows indexed values without changing Matrix Explorer geometry. The graph stays
mounted behind the modal.
