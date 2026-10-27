# Neuron experiment UI evidence

Captured on 2026-10-27 from source/test commit
`a2d83500a9e0fdf5d25627babfa9c78d3ace5809`, based on epic integration
`5663203fe773ced2f90c45976669c1ecf20d76ac`. The subsequent evidence commit
adds only this record and the eight original browser screenshots.

These are measured results from the complete local **HuggingFaceTB/SmolLM2-135M
Base** checkpoint, upstream revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, through the real production backend,
HTTP/LMEX client and built UI. They are neither API mocks nor the design reference.
The run reused the isolated checkpoint copy documented in
[the inspector evidence](../neuron-inspector/README.md); original weights and
checkpoint files were not changed. Python 3.12.14, PyTorch 2.14.0+cpu,
Transformers runtime 4.57.6, adapter `smollm2_native` version 1, float32 CPU;
Node 24.14.0 and the locked Playwright Chromium at DPR 1 under Xvfb/SwiftShader.
No CUDA claim is made.

| Viewport | Signed activations | Shared-context probabilities | Separate continuations | Cancellation |
| --- | --- | --- | --- | --- |
| 1440 × 960 | [Capture](neuron-activation-1440.png) | [Compare](neuron-comparison-1440.png) | [Branches](neuron-continuations-1440.png) | [Cancelled](neuron-cancelled-1440.png) |
| 760 × 960 | [Capture](neuron-activation-760.png) | [Compare](neuron-comparison-760.png) | [Branches](neuron-continuations-760.png) | [Cancelled](neuron-cancelled-760.png) |

The narrow captures use the existing **Collapse browser** control to expose the
complete probability table; the test restores the browser afterward. Inspector,
experiment and tables use ordinary internal scroll, with fixed global bars and
no scientific surface scaling. Screenshots intentionally contain only a public,
test-authored prompt and its measured results. Runtime model files, caches,
private manifests, raw streams and large result arrays are not committed.

## Observed journey

The test selects concrete layer 7 / intermediate coordinate 427, enters
`The capital of France is` in the reused inline tokenizer, and verifies that the
activation POST contains exactly that tokenization's ordered IDs, the pinned
revision and this one neuron. Five prompt positions return signed float32
activations, accessible by keyboard and hover. Invalid index editing and a switch
to another concrete module preserve the editor, disable invalid runs and mark the
previous result stale, without starting another inference.

Comparison executes baseline, zero and scale ×2 at scope `all`, top-k 5,
greedy generation limited to 8 new tokens. The table displays the supplied union
in token-ID order, probabilities from the complete vocabulary and deltas in
percentage points. Each continuation is a separate branch. All three measured
continuations happen to be identical for this small intervention; no divergence
or semantic interpretation is fabricated. **Run again** sends the same recipe.

A subsequent 64-new-token request is cancelled after receiving its operation ID;
the browser verifies the matching DELETE and the distinct cancelled state. The
previous completed result remains explicitly stale. A fresh table-only run
(`max_new_tokens=0`) then completes with no-generation status for each branch.
An explicit **Export configuration** click produces the bounded manifest with the
exact token/neuron identities and options, without opt-in text or continuations;
the test reads it in memory and deletes its temporary download. Graph camera
identity and viewport bounds are asserted throughout.

## Comparison with the design

The [accepted illustrative reference](https://raw.githubusercontent.com/murillo128/llm-model-explorer/48fc9f029b705c1a082f873a97f3a6761e8aee90/docs/design/neuron-explorer/reference.svg)
guides the light, neutral experiment panel below the graph/inspector. Existing
graph and browser composition are preserved. Intentional differences:

- Activations and Compare are separate local modes. The scalar readout uses exact
  signed values rather than a decorative interpolated curve or a new renderer.
- Conditions expose exact operation, factor/delta and scope in labelled controls.
  Baseline is immutable. Expanded controls and result sections scroll internally.
- Candidates use the IDs actually supplied by this API; the UI does not invent
  decoded candidate strings from the mockup. Continuation text is backend-supplied.
- Layer 7 and neuron 427 are the chosen real target. The illustrated percentages,
  activation curve and phrases were not used as acceptance values.

## Validation and test ownership

The API fixture's independent signed values `[-1, 2, 0, -3]` and probability rows
own UI association/readout oracles. UI tests do not reproduce transformer math.

- `neuron-client.test.ts`: typed POSTs, immutable echo association, contract
  association cases, rejecting reordered neurons before delivery, and abort plus
  one operation DELETE before META. Existing decoder/conformance tests retain
  ownership of framing, ranking, finite values and terminal completeness.
- `neuron-experiment.test.ts`: immutable snapshots, dirty identity, invalid
  finite/range inputs, explicit scale/add/scope, table-only configuration,
  percentage-point units and the bounded public export allowlist.
- `neuron-experiment-controller.test.ts`: duplicate-submit prevention, split
  words, completion only after EOF, cancelled/late replies, session/prompt
  replacement and incomplete-body failure without accepting partial results.
- `NeuronExperiment.test.tsx`: signed position/ID association, accessible exact
  decimals, unnormalized subset probabilities and separate continuation blocks.
- `architecture.spec.ts`: real tokenization, runtime and production UI journey,
  browser lifecycle/geometry, cancellation recovery and explicit download.

Validation results, with revisions kept explicit:

| Check | Revision | Result |
| --- | --- | --- |
| API generated-binding check | `fe70ca9` | Passed; no contract/generated changes |
| Typecheck | `fe70ca9`, `a2d8350` | Passed |
| Full lint; final acceptance-file lint | `fe70ca9`; `a2d8350` | Passed |
| UI unit suite | `fe70ca9` | 69 files, 1,298 tests passed |
| Production build | `dd22166` | Passed; existing large-chunk advisory only |
| Inspector + shell browser checks | `fe70ca9` | 9 passed; one real narrow-layout overlap found |
| Previously failing delayed-capability/scroll case | `dd22166` | Passed after graph minimum-height repair |
| Complete SmolLM2 experiment scenario | `a2d8350` | 1 passed, 0 skipped, 0 retries; 21.3 seconds total |

The existing browser test failed on `fe70ca9` because the camera dock intercepted
the Neurons button at 760 px. `dd22166` restores a 160 px minimum graph pane and
adds experiment scroll padding; the unchanged regression then passes. Product
TypeScript is identical from `fe70ca9` through `a2d8350`; subsequent changes are
this focused CSS repair and browser evidence/download verification. Therefore the
unit portfolio was not needlessly repeated. The final browser run uses the same
product build as `dd22166`. An initial new-test assertion incorrectly treated
decorated editor DOM text as raw source; it was corrected to verify retained
annotations, while the independent POST assertion continues to verify exact IDs.

## Reproduction and remaining boundary

Use the repository's locked UI/backend environments, a built UI, and a private
`LMEX_ARCHITECTURE_REFERENCES` manifest naming the complete local checkpoint.
From the repository root, with that manifest configured:

```sh
npm --prefix ui run api:check
npm --prefix ui run typecheck
npm --prefix ui run lint
npm --prefix ui test
npm --prefix ui run build
UI_TEST_PORT=14380 npm --prefix ui run test:browser -- \
  neuron-inspector.spec.ts architecture-shell.spec.ts --project=desktop
PYTHONPATH="$PWD/backend/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  LMEX_REQUIRE_ARCHITECTURE_REFERENCES=1 \
  LMEX_EVIDENCE_DIR=/tmp/neuron-experiment-evidence UI_TEST_PORT=14370 \
  xvfb-run -a --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm --prefix ui run test:acceptance -- architecture.spec.ts --project=dpr1 \
  --grep 'complete local reference.*smollm2.*neuron experiment'
```

The existing product workflow branch filters intentionally defer epic-child PR
CI for `codex/epic-issue-*`; executor-routing CI paths are unchanged. This is
local scoped validation, not a claim that aggregate application CI or the final
epic acceptance has run. The aggregate epic PR retains its mandatory gates.
Full checkpoint comparison mathematics, broader reference acceptance and the
final integration report remain with their existing backend/integration owners.
