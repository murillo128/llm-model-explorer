# Reviewed Architecture components — issue 121

## Contract and baseline

The activation baseline is `a82a1033b18eb575b41f13c6782cee496bd5b1a6` on
`codex/epic-issue-124`, after #119 and #120. This change makes reviewed Attention
and MLP components explicit in every actual layer; it preserves the accepted
logical native/GPTQ/NVFP4 inspection contract. It introduces no new model family,
inference path, tensor geometry, graph schema or layout algorithm.

Dense Qwen3/Llama receive `.self_attn` and `.mlp` module groups; V-JEPA receives
`.attention` and `.mlp` groups in both stacks. Existing Qwen3.5 `.self_attn` and
`.linear_attn` groups are reused, with an additional `.mlp` group. All are real
modules in the already reviewed source revisions documented in
`backend/evidence/`. V-JEPA's head/rotary/GELU operation keys retain their original
identity even though those keys are not lexically inside the corresponding
module prefix. Membership comes from the authored source description.

The affected description revisions become `2`. Reviewed implementation revisions
are unchanged. New graph and record IDs are expected; tensor inventory IDs and
parameter names are unchanged. `semantic_role` and concise operation labels add
navigation information; description provenance retains each semantic source key.
Norms/residual additions remain in the containing layer. Qwen3.5's prior/next
KV, convolution and delta states remain within their original Attention owner.

## Operation-level conservation checkpoint

Before source changes, `architecture_grouping_cases.py` exported observed graphs
from the activation baseline. Its inputs reuse independently authored metadata
fixtures, not the producer's parameter-shape generator. The exporter records the
builder's explicit semantic keys as injective ID correspondence; no label/shape
matching or tensor-ID remapping occurs.

`ui/scripts/check-architecture-semantics.mjs` reuses #119's independent
`semanticSnapshot` oracle. The committed compact baseline contains SHA-256 values
for each semantic section, not generated weights or whole graphs. Seven cases
cover dense Qwen3, Llama, biased Llama, partial Llama, hybrid full/linear Qwen3.5,
and V-JEPA encoder/predictor with and without QKV bias. The comparison excludes
only the expressly added `semantic_role` attribute. Every existing operation,
formula/attribute, directed port dependency, multiplicity, shape, parameter
binding/storage/alias/inspection identity, repetition/instance owner, open
interface and diagnostic remains in the independent oracle.

The baseline can be reproduced by exporting the same cases with the three
producer modules and graph core from the pinned activation commit; the export
helper itself is test-only and compatible with that commit. Expected hashes must
not be refreshed from a changed producer to resolve a computational mismatch.
For detailed diagnosis, compare the before/after `semanticSnapshot` sections;
the full exports stay outside Git.

```sh
PYTHONPATH=backend/src:backend/tests backend/.venv/bin/python \
  backend/tests/architecture_grouping_cases.py /tmp/architecture-grouping
node ui/scripts/check-architecture-semantics.mjs /tmp/architecture-grouping
```

The hand-authored family topology assertions still run, now allowing exact
transparent group forwarding. Their set-based lookup is only a convenience for
individual expected connections; the separate multiset comparison remains the
conservation oracle. New component tests enumerate dense/V-JEPA membership,
per-layer interface and residual/norm ownership, exact module references,
Qwen3.5 state ownership, unused declared layer ports, repeat generation and
revision changes. Existing #119 positive/negative controls cover nested/unused
interfaces, residual drops, swapped equal-shaped ports, incorrect instance
weights, merged K/V state, duplicated signals and independent stacks. Missing
semantic-role metadata alone still validates without changing coverage.

Graph-count growth is accounted for independently of semantic equality:
dense adds two groups and seven forwarding edges per layer; Qwen3.5 adds one
group and two edges per layer; V-JEPA adds two groups and five edges per layer.
No non-group node or parameter is added or removed.

## Local checkpoint checks

The focused family suite passed 109 tests. Grouping/cache/service passed 113
tests, including warm retrieval with generation forbidden, post-readiness cache
loss, per-producer revision invalidation and immutable publication. Backend lint,
formatting, strict typing and generated-record drift checks passed. The seven
independent conservation cases passed against the activation baseline. All 29
issue-119 oracle controls passed. Independent API validation passed OpenAPI 3.1,
283 reference resolutions, 88 instance cases, 144 architecture cases and 76
reproducible wire fixtures.

These observations are synthetic/source-metadata evidence. Complete local
checkpoint and built-browser evidence is recorded separately below; fixture
success is not a claim of actual reference acceptance or inference equivalence.


## Intermediate independent review and complete local checkpoints

The independent read-only [checkpoint review](https://github.com/murillo128/llm-model-explorer/pull/144#issuecomment-5686215639)
returned **PASS**, safe to integrate UI preference, for
`a82a1033b18eb575b41f13c6782cee496bd5b1a6..a83783ef06e1330cbb778f069b4223ffee1ef2bc`.
It was not final-capable. The reviewer additionally removed only newly added
groups and independently checked original ordered hierarchy, interfaces,
directed-edge multiplicity and forwarding shapes. No material findings.

The full backend suite passed 1,224 tests with 19 explicit CUDA-only skips.
The existing TCP Architecture acceptance passed all seven tests with no skips,
including all four approved local reference checkpoints. It ran from an isolated
archive of the reviewed backend commit, with CPU native threads bounded to one,
the existing approved reference manifest, offline model access and fresh temporary
caches. The [reference receipt](evidence/architecture-grouping/references.json)
records fingerprints, inventories, upstream and producer revisions, cold/warm
readiness, memory and exact graph identities. Startup measurements include model
root discovery/hashing; they are not isolated per-model compute benchmarks.

| Actual local checkpoint | Nodes / edges | Cold / warm readiness |
| --- | ---: | ---: |
| SmolLM2-135M Base | 1,031 / 1,596 | 21.237 / 10.304 s |
| Qwen3-0.6B GPTQ Int4 | 1,019 / 1,546 | 21.671 / 9.594 s |
| Qwen3.5-0.8B NVFP4 | 972 / 1,377 | 18.269 / 10.033 s |
| V-JEPA2 ViT-L | 854 / 1,259 | 19.923 / 10.694 s |

The four fixture checkpoints remain a separate test input. No checkpoint weights,
cache payloads or private model paths are committed.

## UI behavior and regression coverage

Navigation prefers explicit source groups and retains their source IDs for
selection, inspection, breadcrumbs and Focus MLP. Authored role labels remain
visible, including Qwen3.5's joint query/gate projection. Original operation paths
remain searchable through the designated semantic-key provenance; generic
description names are not component-path aliases. Legacy group-less MLP fixtures
retain the existing bounded derivation and reversible raw detail.

Projection tests compare authored boundaries to the legacy graph with the
independent operation-level oracle. Layout tests check horizontal nested groups
at two widths. Browser tests exercise the existing renderer's exact source,
destination, branch, shared-trunk and group-port hover, preserve camera/layout
during inspection, and check explicit components without a duplicate derived MLP.

The normal UI check passed generated API bindings, TypeScript, lint, all 713 unit
tests and a production build. The source-path adjustment also passed TypeScript,
lint, targeted projection/invariant tests and a fresh production build. The
independent seven-case conservation check is now part of the existing TCP
Architecture acceptance entry point, including CI.


### Browser reproduction

Use the existing approved local-reference manifest described in
[architecture.md](architecture.md); it must resolve all four complete checkpoints.
From a built checkout, run the Architecture suite and shell regression separately:

```sh
export PYTHONPATH="$PWD/backend/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LMEX_REQUIRE_ARCHITECTURE_REFERENCES=1
# LMEX_ARCHITECTURE_REFERENCES points to the operator's approved manifest.
cd ui
UI_TEST_PORT=16620 xvfb-run -a --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- architecture.spec.ts --project=dpr1
UI_TEST_PORT=17620 xvfb-run -a --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- product.spec.ts --grep 'architecture safety baseline' --project=dpr1
```

On this host, Xvfb's default GLX initialization crashed inside the NVIDIA EGL
library before Chromium could launch. Disabling that X-server extension restored
the display; a separate headed Chromium capability check confirmed WebGL2. The
suite retains its configured SwiftShader renderer and all numerical assertions.
This is a local invocation setting; repository runner/workflow configuration is
unchanged.


### Observed built-browser results

The main Architecture suite passed **9 tests, no skips**, on isolated commit
`64e765361506c1c217f813c998eb3b87dcbddab3`: four deterministic checkpoint fixtures,
all four complete local checkpoints, and cancellation/resource release during
progressive weight streaming. The runtime source is identical to `41211aa`; later
commits only refine acceptance coverage and retain evidence. These checks verify
exhaustive source/edge traceability, concrete instance bindings, native rank-1 and
rank-2 samples, fixture GPTQ/NVFP4 samples, modal camera/resource restoration,
unknown inspection honesty, and no tokenizer call for V-JEPA.

The component-browser coverage includes 48 unaffected connection, inspection,
lifecycle and shell scenarios at `18c391d`, plus all 18 affected component/control
scenarios passing at `41211aa` on desktop and narrow viewports. The initial new
component check incorrectly expected offscreen DOM; it now verifies complete
projection identity and reveals the node before DOM inspection. The path-search
regression was fixed by limiting aliases to semantic-key provenance. No source
semantics or numerical expectations changed to resolve these failures.

Actual Qwen3 and Qwen3.5 reference cases took about six minutes each because they
exercise complete large native embedding matrices. These timings include the
existing weight-modal acceptance workload; they are not graph-layout timings.


### Retained component captures

All linked component captures use complete approved local checkpoints, the built
UI, real backend, 1440 × 1000 viewport and DPR 1. They are not UI-authored graphs.
The executor visually inspected compact and expanded views: component membership
is distinct, flow remains horizontal, norms/residuals remain outside, and V-JEPA's
encoder and predictor have independent navigation context. Expanded Attention
views fit wide graphs into the viewport; native zoom/inspection retains readable
per-operation detail. Screenshots establish presentation, not computational
conservation (which is checked separately above).

| Checkpoint / instance | Compact layer | Expanded Attention | Expanded MLP |
| --- | --- | --- | --- |
| Qwen3 layer 0 | [Compact](evidence/architecture-grouping/reference-qwen3-0-0-compact.png) | [Attention](evidence/architecture-grouping/reference-qwen3-0-0-attention.png) | [MLP](evidence/architecture-grouping/reference-qwen3-0-0-mlp.png) |
| Qwen3.5 layer 0, linear | [Compact](evidence/architecture-grouping/reference-qwen35-0-0-compact.png) | [Attention](evidence/architecture-grouping/reference-qwen35-0-0-attention.png) | [MLP](evidence/architecture-grouping/reference-qwen35-0-0-mlp.png) |
| Qwen3.5 layer 3, full | [Compact](evidence/architecture-grouping/reference-qwen35-0-3-compact.png) | [Attention](evidence/architecture-grouping/reference-qwen35-0-3-attention.png) | [MLP](evidence/architecture-grouping/reference-qwen35-0-3-mlp.png) |
| V-JEPA encoder layer 0 | [Compact](evidence/architecture-grouping/reference-vjepa2-0-0-compact.png) | [Attention](evidence/architecture-grouping/reference-vjepa2-0-0-attention.png) | [MLP](evidence/architecture-grouping/reference-vjepa2-0-0-mlp.png) |
| V-JEPA predictor layer 0 | [Compact](evidence/architecture-grouping/reference-vjepa2-1-0-compact.png) | [Attention](evidence/architecture-grouping/reference-vjepa2-1-0-attention.png) | [MLP](evidence/architecture-grouping/reference-vjepa2-1-0-mlp.png) |


The final per-variant Qwen3.5 pair passed **2 tests, no skips**, on
`508b17e9323bb48737de73d6a36787d3577a5d89`. It checks the first instance of each
actual variant rather than the first two layers: the approved model starts with
linear attention and first uses full attention at layer 3. Both expanded forms
are fitted and checked horizontally, including the linear mask-input path.

The existing shell safety baseline passed **3 tests, no skips**, at 390, 1178 and
1440 pixels on `64e7653`. It preserves shell geometry, Tensor/Tokenizer state and
Architecture navigation; the executor also inspected representative captures.
The [browser receipt](evidence/architecture-grouping/browser.json) retains exact
source snapshots, per-case outcomes, browser/viewport details, observed layout
measurements and shell geometry. Local runs use one worker with independent
ports; values are single-run observations rather than performance guarantees.

All local gates are complete. CPU-only backend skips remain explicit; no CUDA
acceptance or dynamic inference-equivalence claim is made. Final-head CI and
integration freshness are verified on the PR before the normal audit handoff.
