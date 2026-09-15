# Architecture contextual controls — issue 120

## Baseline and boundary

The activation base is `7a8d1f2bf6375a9d90be385ea6282e8d8f050945`, including
[issue 119's safety baseline](architecture-safety.md) and the accepted post-#114
shell and scientific inspection. The owning behavior is specified in
[Architecture Explorer](../docs/spec/ui/architecture-explorer.md#contextual-navigation-controls).

The change replaces the permanent engineering controls with two bounded rows:
Model/containment breadcrumbs, Find component, Fit view and View options, followed
by model entry choices or the active stack's concrete instance/variant and visible
range. Selection has concise inspect/center/clear actions. Advanced expansion,
collapse, zoom, dimensions, interfaces, context, MLP and state controls remain in
the overflow. Search examines all received nodes and reveals the exact selected
source record without numeric loading. Each stack retains its independent window.

Production changes are confined to Architecture controls, scoped CSS, their canvas
handler wiring and model/graph-keyed navigation state. Projection, presentation
aliases, derived grouping, layout/ELK, routing, hit testing, graph retrieval and
numeric inspection implementations are unchanged. The preservation receipt records
exact SHA-256 equality for those source files against the activation base.

## Validation design

The existing semantic, geometry, hover and numeric lifetime suites remain active;
tests use the new picker/overflow locations through a small browser action helper.
The old select inventory assertion now enumerates the search listbox, still
requiring exactly every source node ID in source order.

`architecture-controls.spec.ts` captures real worker requests and results without
replacing layout or its options. Menu opening, text editing, keyboard movement,
details disclosure and inspection compare complete source/projection records,
boxes, ports, routes, camera, canvas identity and layout invocation count. The
shared source oracle also checks received projections. Because worker messages
and Playwright serialize objects, the adapter proves exact record/edge content
before restoring references for the in-process identity oracle. Controlled
mutations to a node label and a source-edge endpoint prove the adapter cannot hide
changed content.

Additional browser cases cover hidden Q projections with repeated labels,
parent/instance disambiguation, exact later-instance parameter names, nonperiodic
variants, first/last endpoints, opening the displayed representative, independent
encoder/predictor windows, breadcrumb return and explorer remount, edge
pin/inspect/clear without navigation changes, partial/no-repetition graphs,
keyboard focus restoration and panel-bounded popovers. Existing native numeric
inspection tests continue to verify exact tensor bindings, values and lifetimes.
The built-shell test additionally checks that opening/typing/dismissing controls
causes no new backend requests and retains the canvas element.

## Built UI and visual evidence

The production acceptance capture uses the same four-layer structural graph and
live numeric/tokenizer fixture service as the safety baseline. It captures Model,
stack, selected node and selected connection at 1178×900, 1440×900 and 390×900.
Toolbar geometry is deliberately changeable; the entire exterior of the panel and
the populated Tensor/Tokenizer views remain the preservation targets.

The capture test checks fixed shell geometry/styles, bounded document dimensions,
no permanent instance selector at model level, reachable contextual controls and
at most two compact control rows. Screenshots and the comparison receipt accompany
this report. These authored graph fixtures establish UI behavior, not actual
checkpoint support. Real checkpoint acceptance remains separate in
[the architecture safety evidence](architecture-safety.md).

All 12 Architecture captures were inspected. Desktop controls occupy 40 px and
36 px rows; the canvas starts at y=144.5 instead of y=201.5, reclaiming 57 px.
At 390 px, each constrained row has one native horizontal scrollbar and the
document remains bounded. Graph labels retain the existing scale.

| Width | Model | Stack | Selected node | Selected connection |
| --- | --- | --- | --- | --- |
| 390 | [Capture](evidence/architecture-controls/safety-architecture-390.png) | [Capture](evidence/architecture-controls/controls-stack-390.png) | [Capture](evidence/architecture-controls/controls-node-390.png) | [Capture](evidence/architecture-controls/controls-edge-390.png) |
| 1178 | [Capture](evidence/architecture-controls/safety-architecture-1178.png) | [Capture](evidence/architecture-controls/controls-stack-1178.png) | [Capture](evidence/architecture-controls/controls-node-1178.png) | [Capture](evidence/architecture-controls/controls-edge-1178.png) |
| 1440 | [Capture](evidence/architecture-controls/safety-architecture-1440.png) | [Capture](evidence/architecture-controls/controls-stack-1440.png) | [Capture](evidence/architecture-controls/controls-node-1440.png) | [Capture](evidence/architecture-controls/controls-edge-1440.png) |

The [comparison receipt](evidence/architecture-controls/visual-comparison.json)
records **zero differing pixels** for the entire populated Tensor and Tokenizer
captures and the entire Architecture exterior at both baseline desktop widths.
Shell geometry and styles also match exactly. Pixel comparison uses RGB channel
differences with no tolerance, excluding only the explicitly recorded Architecture
interior rectangle. Identical Tensor/Tokenizer images reuse the committed baseline
instead of duplicating them. The new narrow captures have no prior pixel baseline;
their shell preservation and bounded geometry are checked within the same run.

## Observed checks

- UI check: binding generation, typecheck, lint, **710 unit tests**, production build.
- Architecture browser suites: **58 passed** across desktop and narrow projects.
- Final production visual checks: **3 passed**, covering all 12 captures above.
- Final typecheck, lint and build: passed after the narrow scrollbar adjustment.

The [check receipt](evidence/architecture-controls/checks.json) records preserved
source hashes, the validated source snapshot and the timing of final CSS/capture
adjustments. Required application and full UI CI results belong to the exact PR
head and are recorded on the PR.

## Cross-stack canvas navigation regression

Revealing an encoder MLP operation, then a predictor operation, leaves both
stack windows available on the global canvas. Activating the encoder's derived
MLP label must switch both its breadcrumb and contextual instance controls to the
encoder. The shared navigation handler now persists focus and active repetition
together, resolving derived MLPs through their source owner. Stack overview uses
its explicit repetition because its focal node is the common stack parent.
Selection and each stack's expansion/window state remain independent.

The new browser regression first reproduced the missing Encoder picker on
`db9edc66905f1a91f4901df2c60b94ce2c1b8faa`. It exercises pointer and keyboard
activation through the actual canvas, verifies the concrete variant, both
navigation endpoints, independent windows and unchanged selected source, then
remounts the explorer and follows Open instance and Previous/Next. The original
12 visual captures and their source receipt describe the initial controls
implementation; this follow-up changes only navigation wiring and adds this
regression. Follow-up source hashes and validation are recorded separately in
the check receipt. The full UI check passed again (710 unit tests), all four
pointer/keyboard regression cases passed at desktop/narrow widths, and the 12
existing controls cases passed on the same production source.

## Reproduction

Use the locked Node/Python/Chromium environments and an isolated `UI_TEST_PORT`.
Run one heavy suite at a time. The UI and production browser gates are unchanged.

```sh
npm --prefix ui run check
npm --prefix ui run test:browser -- architecture-controls.spec.ts architecture.spec.ts architecture-connections.spec.ts architecture-inspection.spec.ts architecture-shell.spec.ts
# From ui/; built UI and live fixture backend:
xvfb-run -a npm run test:acceptance -- product.spec.ts --grep 'architecture safety baseline' --project=dpr1
xvfb-run -a npm run test:acceptance -- architecture.spec.ts --project=dpr1
```

The PR's application gate additionally runs `acceptance/check-integration.sh`
(contract validation, HTTP acceptance and the complete DPR 1 product suite).
Full-size authored graph fixtures and absent local references must never be
reported as real-model acceptance. Large logs, worker outputs, traces and local
model/cache contents remain outside Git.
