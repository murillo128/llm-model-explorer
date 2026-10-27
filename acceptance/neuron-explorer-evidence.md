# Neuron Explorer: complete SmolLM2 Base acceptance

Acceptance implementation source: `b8565a6cf515bbb2add9a58e4da28c4bc1c87d5a`, built on the
canonical integration base `94877f94d08af846842ef704a6d6b7cdb18327be`.
Observed on 2026-10-27. This report distinguishes full-checkpoint execution from
the synthetic mathematical/protocol fixtures and the illustrative design mockup.
The publication commit adds only this report, usage/routing documentation and
captured screenshots. The PR records that exact head and its hosted CI disposition.

## Use and limits

Open a complete local **HuggingFaceTB/SmolLM2-135M Base** checkpoint with native
Safetensors and its tokenizer. Base completes text; it is not an Instruct chat
model. The application does not download weights or apply a chat template.
Execution is admitted independently from graph, tensor and tokenizer inspection.
Instruct, other families, quantized checkpoints and LoRA execution are outside
this adapter; an unavailable execution capability leaves independent inspection
usable. CPU float32 is the validated baseline; CUDA is optional.

In Architecture Explorer choose **Neurons**, a concrete MLP and an intermediate
index. The inspected coordinate is
`a[j] = SiLU(gate(x)[j]) * up(x)[j]`, immediately before `down_proj`, not a weight
cell or a semantic label. Gate/up open the complete weight at row `j`; down opens
it at column `j`. When the optional graph binding is absent, the explicit concrete-module selector
provides the entry (as observed for this checkpoint). Closing a weight preserves
the graph camera and selected coordinate.

Enter a short prompt and wait for its current tokenization. **Run** observes
signed float32 activations by prompt position, forwarding those exact token IDs.
**Compare** keeps baseline first, with zero (replace by 0), scale (multiply by a
factor), or add (offset by a delta). Each condition starts from the same unmodified model and exact prompt; an
intervention uses the original forward value at each position. Weights are never changed. Prompt/all/continuation scopes refer to
absolute input positions. Continuation-only scope cannot change the first
next-token distribution, which comes from the last prompt position.

The probability table compares the same prompt context, uses full-vocabulary
probabilities without renormalizing the displayed candidate union, and labels
differences in percentage points. Greedy continuations occupy separate branches;
their later positions are not a controlled shared-context comparison. A changed
probability need not change the argmax or generated text. An effect on one prompt
does not establish that a neuron owns a semantic concept. The controls display
the effective request/context limits; there is no sampling or temperature control.

Cancel explicitly, then run a fresh configuration. Incomplete bytes are not a
completed result. Editing the prompt/selection, replacing the session or model
revision, and closing the view release obsolete work. Experiment content stays
in memory, with no automatic prompt, output, activation or comparison history
on disk. **Export configuration** is an explicit small recipe download; token
IDs disclose prompt information, and optional inclusion of user text is sensitive.
User text is excluded by default. Result arrays and generated continuations are
always excluded. A saved opaque revision
token does not prove content equality across restarts or authorize a stale session.
The accepted [API](../docs/spec/api/neuron-explorer.md),
[runtime](../docs/spec/backend/neuron-execution.md), and
[view](../docs/spec/ui/neuron-explorer.md) specifications own the detailed rules.

## Evidence owners and bounded additions

| Contract / defect detected | Owner and independent expected result |
| --- | --- |
| Wrong tap, sign, selected coordinate, operator, absolute scope, softmax or greedy result | Existing `backend/tests/test_neuron_{activations,comparisons,execution}.py`: explicit scalar math and separate seeded Transformers oracles; all 91 tests passed in the aggregate run. No E2E parameter-matrix duplication. |
| Full local checkpoint loads incorrectly or intervention modifies resident weights | New `acceptance/test_neuron_reference.py`: ordinary uninstrumented local `LlamaForCausalLM.from_pretrained` baseline against real TCP probabilities/top-k/greedy IDs, whole resident-parameter digest, same runtime instance before/after zero/scale and a clean baseline, plus unchanged source-shard SHA-256. |
| Wrong IDs, float32 position/condition mapping or displayed continuations | Existing production-browser neuron scenario now matches actual LMEX bytes to every displayed activation/probability cell and continuation, preserving request/session/module/recipe identity. Backend tests retain mathematical ownership. |
| Cancellation contaminates a subsequent baseline | Existing browser cancellation scenario now removes interventions and checks exact baseline probabilities against its pre-cancel baseline. Existing backend tests own controlled interruption, two-session isolation and hook/buffer teardown. |
| Old supported response repopulates a replacement revision | New full-checkpoint browser case holds a real capability response, atomically replaces only a temporary config with unsupported nonzero dropout, and observes native SSE/session replacement. After releasing the old response, unavailable state remains, with no module/experiment controls; graph and native weight inspection still work. |
| New reference check silently loses CI consumers | Selector tests route architecture UI/browser and shared reference-oracle changes to `test_neuron_reference.py`; its own leaf remains HTTP-only. Shared service/runtime changes retain broad integration. |

The numerical bridge reuses `Service`, `Frames` and the existing reference
manifest/inventory checks. Its loopback-only acceptance-server observer reads
resident weight digests and hook/task counts; it exposes no prompt or result.
The production CLI has no `/__test` endpoint. Browser checks use the production
CLI, a fresh production UI build, Chromium/SwiftShader and real HTTP, without
fabricated inference responses. A passive, one-request browser observer copies bytes as the application's existing
reader consumes them; it does not clone or drain a second stream. The only held
response is real capability JSON
in the revision race. The operator checkpoint stays read-only: an atomic rename
breaks the temporary config hard link before revision mutation.

Privacy, reader release, no experiment-cache publication and independent-session
cleanup remain owned by the existing backend tests, including
`test_tcp_headers_precede_computation_and_complete_comparison_layout`,
`test_incomplete_conditions_release_hooks_and_buffers`,
`test_consumer_lifecycle_drains_worker_and_other_session_stays_clean`, and
`test_comparison_reader_releases_output_on_content_change`.

## Tested checkpoint and environment

- Upstream: `HuggingFaceTB/SmolLM2-135M` Base, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` declared by the operator manifest.
- Content fingerprint: `e10c7a1b7533e2ed6b2471bb3bee9b837947c80c7a200c299d5794c2d4ff94ff`.
- Complete native BF16 storage, 272 tensors; effective execution float32 on CPU.
- Linux 6.8, Intel Core i7-11700K, Python 3.12.14, PyTorch 2.14.0+cpu,
  Transformers 4.57.6, `smollm2_native` adapter version 1.
- Node 24.14.0/npm 11.9.0; locked Chromium, SwiftShader, DPR 1, headed Xvfb.
- `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`; test process also bounds PyTorch to
  two threads. CUDA was unavailable. No CUDA execution acceptance is claimed.

The fixed public prompt is `The capital of France is`, at layer 7 / neuron 427.
The browser compares baseline/zero/scale×2 with eight greedy tokens; the direct
numerical bridge uses two tokens and top-k 5. The independently loaded baseline
uses normal library execution rather than the product's instrumented forward.
The three displayed eight-token continuations were identical in this observed
prompt, while probabilities changed. Probability tolerance is `rtol=1e-5, atol=1e-7`, allowing float32 reduction and
attention-kernel roundoff; the exact resident weight digest and greedy IDs must
match. Existing small mathematical oracles retain their tighter tolerances.

| Observation | Measured value |
| --- | --- |
| Backend startup/readiness | 5.303 s |
| Initial observation including model loading | 1.163 s |
| Warm three-condition comparison, two greedy tokens | 0.390 s |
| Backend RSS / high-water RSS | 962856 kB / 1072828 kB |
| Maximum baseline probability absolute error | 1.57952309e-06 |
| Maximum intervention probability absolute delta | 0.000364392996 |

Resident parameter SHA-256 before/after and independent direct loader:
`988be794bfc03d1396742ed5c6512f40dedf48fdd8ae89bddde47094f5d63e82`. The runtime instance and hook/task-free
state also remained identical; on-disk shard digests were unchanged.

These are single observed runs, not performance thresholds. Startup includes
process launch, fingerprinting and architecture preparation. The first activation
request includes model loading; comparison is warm. Linux backend VmRSS/VmHWM
includes the test observer's temporary digest copies and is not a production-only
memory benchmark. No private model path or model weights are published.

## Validation and reproduction

| Check | Result and source revision |
| --- | --- |
| Independent API conformance | Passed at `fcce33c`: 468 references, 130 instance cases, 279 architecture cases, 81 neuron cases, eight neuron payloads, 138 reproducible wire fixtures |
| Backend Ruff/format/mypy and full pytest | Passed at `fcce33c`: 2,122 passed, 20 CUDA skips; 1,369.96 seconds for pytest |
| Real HTTP acceptance | Passed at `9b4eec7`: 89 passed, eight optional skips; 339.92 seconds |
| UI binding/type/lint/unit/build | Passed at `9b4eec7`: 69 unit files, 1,298 tests |
| Full component-browser portfolio | Passed at `9b4eec7`: 460 tests, two workers, zero retries; 10.6 minutes |
| Production-browser portfolio | Passed at `b8565a6`: 45 tests, 14 explicit optional skips, one worker, zero retries; 13.6 minutes |
| Isolated installed-wheel smoke | Passed at `89cb4d1`: 18 tests; 66.68 seconds, isolated installed package |

The initial aggregate HTTP phase exposed three pre-existing launcher-test
isolation failures under the supplied reference environment. Their synthetic
routine/extended plans inherited reference inputs and became full plans. The
bounded repair gives plan creation, metadata inspection and gate execution one
explicit test-owned environment; the explicit reference-promotion test remains.
All 49 launcher/selector cases and 124 subtests passed with the outer reference
variables present, followed by the complete green HTTP rerun. The failed report
was retained separately. See the [resolved validation contract](https://github.com/murillo128/llm-model-explorer/issues/322#issuecomment-6443150370).

`9b4eec7` changes only that launcher test relative to `fcce33c`; backend, API,
selector, product UI and the new numerical/browser tests are identical between
those two commits. All checks in this table were run during #322 execution.
A bounded command-runner helper collected the wheel result; the executor ran
the other suites. Passed evidence is attributed to its actual source revision. Empty
Git diffs verified `fcce33c..b8565a6` for `backend/` and `api/`, and
`9b4eec7..b8565a6` for acceptance Python/selector inputs, `ui/src/`, `ui/tests/`,
package/lock files and component Playwright configuration.
`89cb4d1` changes only the browser byte observer: the earlier full production
run at `9b4eec7` had 44 passes, 14 optional skips and one failure because Chromium
released a response body before Playwright could retrieve it through CDP. The
repair observes the existing reader without changing response bytes, assertions
or cancellation behavior. A second full run at `89cb4d1` reached the same CDP
retention limitation earlier, while reading capability JSON after graph preparation
(44 passes, 14 optional skips, one failure). The resolved metadata read reuses the
native-weight test's ordinary GET of the observed session-bound capability URL.
The focused run then detected an empty observer capture: ApiClient had bound
fetch before the hook was installed. `b8565a6` installs the passive observer in
the existing before-navigation hook and arms it only for successful inference
requests. All three focused neuron scenarios passed in 59.1 seconds, with
type/lint checks passing. The complete production portfolio was rerun afterward.
Failed and interrupted reports were retained separately; see the
[browser-observation clarification](https://github.com/murillo128/llm-model-explorer/issues/322#issuecomment-6444389183). The
first one-worker component run was interrupted to use the existing CI two-worker
setting; its partial report is not counted as a pass. Production browser uses
one worker. Traces remain off and retries remain zero. Missing CUDA and unrelated
Qwen/V-JEPA/LoRA reference prerequisites are skips, not certification.

Provision dependencies using [the native acceptance instructions](README.md).
Supply an existing private architecture-reference manifest with a `smollm2`
selection; use the provisioned checkpoint, without downloading in the application.
Focused commands from the repository root:

```sh
export LMEX_ARCHITECTURE_REFERENCES=/private/reference-manifest.json
export LMEX_REQUIRE_ARCHITECTURE_REFERENCES=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
backend/.venv/bin/python -m pytest acceptance/test_neuron_reference.py -q \
  -o junit_family=legacy --junitxml=/tmp/neuron-reference.xml
npm --prefix ui run build
UI_TEST_PORT=14430 LMEX_EVIDENCE_DIR=/tmp/neuron-browser \
  xvfb-run -a --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm --prefix ui run test:acceptance -- architecture.spec.ts \
  --project=dpr1 --grep 'complete local reference.*smollm2.*neuron'
```

The initial aggregate run used `acceptance/check.sh`, the same manifest, the Base
directory in `LMEX_REFERENCE_MODEL_DIR`, and an isolated `LMEX_EVIDENCE_DIR`/
`UI_TEST_PORT`. Strict architecture-reference mode is appropriate for the focused
SmolLM2 invocation, not the all-family aggregate command when unrelated Qwen,
V-JEPA, MoE or LoRA checkpoints were not supplied. Missing optional references
remain explicit skips. The aggregate epic PR still owns its mandatory hosted CI;
child PRs targeting `codex/epic-issue-*` have the repository's declared product-CI
deferral. No check is replaced by the launcher or by absent check badges.

After the documented harness corrections, completed phases used these native
commands with the same reference/thread environment and isolated evidence roots:

```sh
unset LMEX_REQUIRE_ARCHITECTURE_REFERENCES
export LMEX_REFERENCE_MODEL_DIR=/private/SmolLM2-135M
export LMEX_TEST_PORTFOLIO=full
backend/.venv/bin/python -m pytest acceptance -ra --durations=25 \
  -o junit_family=legacy --junitxml=/tmp/neuron-final/network.xml
npm --prefix ui run check
PLAYWRIGHT_JSON_OUTPUT_FILE=/tmp/neuron-final/component.json UI_TEST_PORT=14450 \
  xvfb-run -a --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm --prefix ui run test:browser -- --workers=2 --reporter=list,json
LMEX_TEST_PORTFOLIO=full LMEX_EVIDENCE_DIR=/tmp/neuron-final UI_TEST_PORT=14550 \
  xvfb-run -a --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm --prefix ui run test:acceptance
bash acceptance/check-wheel.sh "$PWD/backend/.venv/bin/python"
```

The table records phase-specific source revisions rather than claiming a single
uninterrupted aggregate success. Full raw JUnit/browser/log evidence, including
failed and interrupted runs, remains outside Git; the compact measurements and
selected screenshots below are the published evidence.

## Actual screenshots

The selected unedited captures below come from the complete green production
browser run at `b8565a6`, using the full native checkpoint, CPU runtime and browser environment recorded above. Viewports are
1440×960 and 760×960 at DPR 1. Only the fixed public test prompt is present.

| Real journey state | Capture |
| --- | --- |
| Layer 7 and neuron 427 in the inspector | [Desktop inspector](evidence/neuron-integration/neuron-reference-inspector-1440.png) |
| Native down-projection column 427 | [Narrow weight view](evidence/neuron-integration/neuron-reference-weight-760.png) |
| Signed activations at each actual token position | [Desktop activation table](evidence/neuron-integration/neuron-activation-1440.png) |
| Same-context baseline/zero/scale probabilities and deltas | [Desktop comparison](evidence/neuron-integration/neuron-comparison-1440.png), [narrow comparison](evidence/neuron-integration/neuron-comparison-760.png) |
| Three separate greedy continuations | [Narrow continuation cards](evidence/neuron-integration/neuron-continuations-760.png) |
| Explicit cancellation, with no partial result accepted | [Narrow cancelled state](evidence/neuron-integration/neuron-cancelled-760.png) |
| Unsupported replacement revision after rejecting the old response | [Desktop unavailable state](evidence/neuron-integration/neuron-revision-unavailable-1440.png) |

These are full-checkpoint CPU screenshots of the public test prompt, not the
design mockup or API fixtures. The unsupported capture uses the isolated changed
config described above. No product UI changed in this issue, so no before/after
UI repair is claimed. Numeric values and complete weight navigation take
precedence over illustrative values in the design reference.
