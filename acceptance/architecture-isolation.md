# Architecture component isolation — issue 122

## Projection checkpoint

Activation base: `5d3522272cae1a7da8afcdf47017702b7698b36c` on
`codex/epic-issue-124`, including the #119 conservation baseline, #120 controls
and #121 authored component boundaries. The issue branch preserves the existing
default-branch runner-maintenance commit through a merge of that pinned base.

`scope.ts` resolves one source group/operation or the existing justified derived
MLP into source membership. `projection.ts` visits only those roots, with no
ancestor layout boxes. An external alias represents one exact original endpoint
in one crossing direction. Its ports and connection paths retain original
records; shapes and labels never establish signal identity. Forwarding remains
transparent only inside the selected scope, stopping at computational operations
and external endpoints. Exits and re-entries are separate external dependencies;
wholly external bypasses are explicitly excluded. The source graph is unchanged.

The projection exposes source membership and excluded source node/edge IDs.
Existing represented/hidden/filtered edge accounting remains a disjoint complete
partition. Declared source interfaces remain available for inspection. Consumption
is evaluated against the full source connectivity, independently of visibility.
An absent scope takes the unchanged global projection path.

At this checkpoint, the focused scope/projection/invariant/automatic-layout suite
passed **78 tests**. TypeScript, targeted ESLint and whitespace checks passed.
Tests reuse #119's source-reference and directed-path oracle and the actual ELK
geometry assertions. They cover authored/derived groups, leaf operations,
independent encoder/predictor components, partial/no-repetition graphs, equal-shaped
inputs, fan-out, separate K/V state, auxiliaries, residual exclusion and external
re-entry. Increasing surrounding layers from 4 to 48 leaves isolated Attention,
MLP and leaf-operation boxes, ports, routes and bounds identical.

```sh
cd ui
npm exec vitest run src/architecture-explorer/scope.test.ts \
  src/architecture-explorer/projection.test.ts \
  src/architecture-explorer/invariants.test.ts \
  src/architecture-explorer/auto-layout.test.ts
```

This checkpoint established structural fixture evidence. Navigation followed only
after the issue-declared independent semantic checkpoint passed.

The first independent checkpoint found that external context aliases bypassed
`showContext: false`. The correction classifies those aliases using the original
source node kind while keeping exact endpoint identity. A new API-validated
context fixture proves identical reversible filtering globally and in isolation.
The external re-entry fixture now also passes normal API validation and retains
both external computation segments as excluded records. The revised focused
suite passed **79 tests**, plus TypeScript and targeted lint. A separate exact
comparison of 24 global graph/view combinations against the activation source
produced identical projection records, ELK boxes/ports/routes and bounds (excluding
elapsed layout time). No global expectations were changed.

## Reversible navigation and state

The corrected semantic checkpoint received an independent **PASS** at
`256691ce6e1466a1b3b5303ea93ccfc797f9fed8`, safe to proceed with navigation,
final-capable **no**. The [checkpoint verdict](https://github.com/murillo128/llm-model-explorer/pull/146#issuecomment-5687713381)
records the fixed context filter and independent valid re-entry/group-free probes.

`scope-navigation.ts` captures only view options, IDs and the viewport. Back
restores camera, node/connection selection, expansion, filters and independent
repetition windows. History retains at most 16 entries plus the global fallback;
backend-owned `GraphViews` retains at most eight model/graph views and removes
superseded graph identities. No source graph, numeric resource or layout copy is
stored in a history entry. Existing session-keyed mounting and worker cancellation
reject obsolete results. A failed scope layout retains Back and Retry access.

The isolated state and concrete source breadcrumbs remain visible on the same
canvas. Explore component is separate from source inspection and expansion; the
existing derived MLP exposes it in group inspection. Search labels results outside
the current scope and reveals them in model context. View in model restores the
global view before revealing/centering the chosen source component. Expand
component expands only scope membership; Show all operations in model exits
isolation and retains true exhaustive access. Fit includes scope context aliases;
new scopes use an initial camera with zoom at least 0.8 and retain normal pan.

## Validation observations

The navigation candidate `9adaf7d291028a226dfb6c5f545aa73d5b008e57` passed the
repository UI check: generated bindings, TypeScript, ESLint, **730 unit tests**
and production build. The complete Architecture browser suite passed **80 tests**
(**40 desktop + 40 narrow**, no skips or retries), including nested Back, exact
cameras/options, source/derived scopes, global
exhaustive access, model growth, cross-stack search, explorer/model replacement,
failed layout retry and queued late-result rejection. The actual pointer tests
verify exact external fan-out, shared trunk, exclusive branch and endpoint/pin
sets while preserving generated geometry, camera and layout invocation count.
The executor inspected its boundary-port/shared-trunk captures. An additional
API-validated group-free leaf regression passed in the **13-test scope suite**;
the final candidate therefore contains 731 unit cases.

Two earlier attempts to run the UI check from a linked-dependency archive were
invalidated by snapshot setup: a missing nested workspace dependency, followed by
Vite denying an external worker asset. An archive with its own locked `npm ci`
installation fixes that environment without source/test/CI configuration changes.
These were not product-test failures or waived checks.

The unchanged backend at `bd787a137e13c71ecb3c007dd48c6f48ddbf9885` passed all
**8 TCP Architecture tests, no skips**, including the independent operation-level
contract and all four approved complete local checkpoints. The compact
[reference receipt](evidence/architecture-isolation/references.json) records exact
source identity, model fingerprints, graph identities, counts, cold/warm readiness
and observed process memory. No weights, model paths, raw graphs or cache entries
are retained in Git. CPU execution is not CUDA evidence; these tests do not claim
dynamic inference equivalence.

The built acceptance snapshot `cd09e738b8f2072d89e3d1917eadfd59d4e0950b`
passed **9 production-browser tests, no skips or retries**: four deterministic
fixtures, all four complete local references, and repeated cancellation/resource
release from an isolated operation. Scope entry and pointer inspection issue no
backend requests. Native weight values and modal return/lifetime checks passed.
The **3 shell safety cases** passed at 390, 1178 and 1440 pixels. At both desktop
widths, whole Tensor/Tokenizer captures and the pixels outside the Architecture
interior (ordinary and isolated views) exactly match the post-#114 baseline:
**zero different pixels in all eight comparisons**. Camera restoration and
unchanged numeric texture/reader counts also passed.

A final comparison on that exact acceptance snapshot again matched **24 global
views** against the activation baseline, including every projection record,
ELK box/port/route, source edge identity and layout bound. The final commit adds
only the group-free regression and this evidence after the tested runtime.
The compact [check receipt](evidence/architecture-isolation/checks.json) records
source heads, counts, isolated geometry, exact shell measurements, pixel
comparison masks and capture hashes. Large reports, traces, raw graphs, caches
and local model paths remain outside Git.

### Built reference captures

The real dense-Qwen Attention and MLP captures at 1178 and 1440 pixels show both
initial navigation and explicit Fit. MLP fits as a readable complete component
with its shared gate/up input. Attention has a genuinely wide operation chain:
the initial 0.8-scale camera keeps labels readable and requires horizontal pan;
explicit Fit includes every external input/output and makes the complete chain
smaller. Isolation does not rearrange or replace the established layout engine.
Boundary-port and shared-trunk captures retain the exact Q/K/V or gate/up fan-out.
The inspected V-JEPA2 encoder captures also preserve the separate positional
input, Q/K/V fan-out and real residual destination. Its MLP retains its three
original operations. No unrelated stack contributes to the isolated bounds.

| Complete reference component | 1178 pixels | 1440 pixels |
| --- | --- | --- |
| Qwen3 Attention | [Initial](evidence/architecture-isolation/isolated-qwen3-attention-1178-initial.png), [boundary port](evidence/architecture-isolation/isolated-qwen3-attention-1178-boundary.png), [Fit + trunk](evidence/architecture-isolation/isolated-qwen3-attention-1178.png) | [Initial](evidence/architecture-isolation/isolated-qwen3-attention-1440-initial.png), [boundary port](evidence/architecture-isolation/isolated-qwen3-attention-1440-boundary.png), [Fit + trunk](evidence/architecture-isolation/isolated-qwen3-attention-1440.png) |
| Qwen3 MLP | [Initial](evidence/architecture-isolation/isolated-qwen3-mlp-1178-initial.png), [Fit + trunk](evidence/architecture-isolation/isolated-qwen3-mlp-1178.png) | [Initial](evidence/architecture-isolation/isolated-qwen3-mlp-1440-initial.png), [Fit + trunk](evidence/architecture-isolation/isolated-qwen3-mlp-1440.png) |
| V-JEPA2 encoder Attention | [Initial](evidence/architecture-isolation/isolated-vjepa2-attention-1178-initial.png), [Fit](evidence/architecture-isolation/isolated-vjepa2-attention-1178.png) | [Initial](evidence/architecture-isolation/isolated-vjepa2-attention-1440-initial.png), [Fit](evidence/architecture-isolation/isolated-vjepa2-attention-1440.png) |

### Reproduction

Use locked UI/backend dependencies and the approved local-only reference manifest
from [Architecture acceptance](architecture.md). Each suite uses one worker and
its own output directory/ports. On the acceptance host, Xvfb's GLX extension is
disabled to avoid the previously recorded NVIDIA EGL startup crash; Chromium
continues using the suite's configured SwiftShader WebGL2 path.

```sh
cd ui
npm run check
UI_TEST_PORT=19620 npm run test:browser -- tests/architecture \
  --project=desktop --project=narrow
# Supply LMEX_ARCHITECTURE_REFERENCES and LMEX_REQUIRE_ARCHITECTURE_REFERENCES=1.
# Set PYTHONPATH to the tested snapshot's backend/src; keep model access offline.
UI_TEST_PORT=20620 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xvfb-run -a --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- architecture.spec.ts --project=dpr1
UI_TEST_PORT=21620 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xvfb-run -a --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- product.spec.ts \
  --grep 'architecture safety baseline' --project=dpr1
```
