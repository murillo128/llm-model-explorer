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
| Complete Architecture Explorer and Tensor inventory browser suite, narrow | 76 passed |
| Desktop camera, family, repetition, shared-instance and isolation revalidation | Passed |
| Built UI + real backend, four architecture fixtures and cancellation/shared-instance scenarios, DPR 1 and 2 | 12 passed |

The PR's UI check supplies complete desktop and native-scrollbar regression
coverage on the published commit; its application check runs the integrated
product gates. Both must pass on the current commit before the review-ready
handoff. The local narrow run also verifies the complete synthetic graphs for
SmolLM2, Qwen3, Qwen3.5 and V-JEPA2, independent of checkpoint acceptance.

## Reproduction

Use the repository's locked Node 24.14/npm dependencies and existing Python 3.12
backend environment. One heavy suite runs at a time with isolated ports.

```sh
cd ui
npm run check
UI_TEST_PORT=4320 npm run test:browser -- 'tests/architecture.*spec.ts' \
  tests/tensor-inventory.spec.ts --project desktop --project narrow
```

The production acceptance uses the existing harness, real HTTP backend, built UI,
SwiftShader, and local synthetic model fixtures:

```sh
UI_TEST_PORT=4340 xvfb-run -a npm run test:acceptance -- architecture.spec.ts \
  --project dpr1 --project dpr2 \
  --grep 'deterministic production.*full graph|close during progressive|shared structure production'
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
