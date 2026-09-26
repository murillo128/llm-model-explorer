# Integrated product acceptance

This harness runs the current production backend, independent API and built UI
as one application. Component tests remain separate gates. Nothing under this
directory is imported by the shipped backend, and the browser probe is injected
by Playwright, not bundled into the UI.

## Reproduce

From the repository root, with uv 0.12.13, Python 3.12 and Node 24.14.x/npm 11.9.x:

```sh
uv sync --locked --project backend --python 3.12
uv venv api/.venv --python 3.12
uv pip install --python api/.venv/bin/python -r api/requirements.txt
npm ci --prefix ui
(cd ui && npx playwright install --with-deps chromium)
acceptance/check.sh
```

Linux needs `xvfb-run` for both browser suites. Product tests run headed Chromium
under Xvfb with native scrollbars and real WebGL2 at DPR 1 and 2,
using SwiftShader by default. Set `LMEX_WEBGL_BACKEND=vulkan` to request the host
Vulkan path; the actual renderer and limits are recorded, so this setting alone
is not a hardware claim. This tests browser GL behavior, not physical GPU
performance. The backend computes fixture results on CPU. Dependency installation
may use the network; the acceptance application runs with `HF_HUB_OFFLINE=1`.

The headed product suite uses one worker in both local runs and CI to isolate
native window/pointer interactions. Parallel runs on the shared X display showed
intermittent loss of transient inspection/drag state, while the same scenarios
passed serially. Separate ports do not isolate native input. Component browser
parallelism is unchanged.

The runner writes JUnit, JSON counts, measurements and version information to a
new `/tmp/lmex-evidence-*` directory. Set `LMEX_EVIDENCE_DIR` for a stable output
location. Browser screenshots and failure-only traces go to ignored
`ui/test-results/`; CI uploads these for 14 days. Only a concise report and a few
small selected screenshots belong in Git. `LMEX_CONTRACT_PYTHON` can select an
existing API-tools interpreter; API tooling stays separate from the backend lock.

Focused commands:

```sh
backend/.venv/bin/python -m pytest acceptance -ra
(cd ui && npm run build && xvfb-run -a npm run test:acceptance)
```

The product browser harness reserves loopback ports 4175 (static UI) and 8765
(backend). Each test starts a fresh backend subprocess with a temporary model and
cache root and stops it at teardown. The HTTP suite uses dynamically allocated
loopback ports. Do not run two product browser suites simultaneously. A static
server serves `ui/dist` and a runtime URL file; it does not proxy API traffic or
serve development modules. CORS allows only the static origin and exposes
`X-Operation-Id` through the production middleware.

## CI ownership

On `main`, Application acceptance runs `bash acceptance/check-integration.sh
--main-ci`: acceptance lint/format checks, the real HTTP suite, a production UI
build from the same checkout, and the complete product browser suite at DPR 1
and 2. It does not rerun backend unit/type/lint checks, UI unit/type/lint or
component-browser checks, or API fixture/binding validation. Those remain owned
by `backend-ci.yml`, `ui-ci.yml`, and `api-contract.yml`, respectively, subject to
their existing path filters. Their Python-version and browser-project coverage
is unchanged. The historical application job/check name is retained.

PRs and epic integration branches still use `check-integration.sh` without
arguments, including their contract/binding checks and DPR 1 product coverage.
`check.sh` remains the standalone full local command: it runs every layer and
produces the aggregate report. The lean main CI gate uploads HTTP JUnit and
product browser evidence, not a duplicated component report. API-tool environment
setup is unnecessary in that main gate and is skipped; backend, Node and browser
setup remain available for real integration tests. No test results or backend
responses are cached or mocked by this change.

The entrypoint regression tests use command stubs solely to verify orchestration,
including failure propagation; they do not replace application tests. Run them
without application dependencies using `python -m unittest discover -s acceptance
-p test_ci_entrypoints.py`.

## Fixture and coverage

`fixtures.py` generates HF-compatible safetensors and tokenizer assets outside
Git. Seed 17 defines `((index * 17) % 257 - 128) / 128`, exactly representable in
FP16 and float32. The fixture has `[576,1536]` and `[1536,576]` MLP orientations,
a `[1025,576]` embedding, a supported Llama embedding configuration, a 576-value
vector, an empty tensor, and a small F32 scientific tensor with NaN/infinity/signed
zero. The browser additionally generates four overflow matrices (`[32,32]`,
`[1200,32]`, `[32,1600]`, `[1200,1600]`), 80 small inventory leaves, four 20×20
distribution-scale edge cases (near-zero concentration, outliers, constant, and
all-nonfinite), and an
unsupported-embedding model with the same usable tokenizer. The byte BPE uses an
explicitly ordered alphabet with no stochastic training.

| Gate | Real application evidence |
| --- | --- |
| Core exploration routes | Model/session/inventory/three tensor streams/tokenize/embeddings/cancel over TCP; CORS preflight and structured pre-stream errors |
| Scientific correctness | Exact fixture float32 bytes; independent NumPy float64 statistics/percentiles and independently accumulated uint32 row/column counts |
| Progressive delivery | Hold the real producer after its first flushed block; observe DATA and rendered pixels while publication is still impossible; distributions have their own barrier |
| Sharing and cache | Two sessions, distinct public operation IDs, late join from byte zero, parked slow reader, one producer, warm reuse with identical key/digest/inode/mtime |
| Lifecycle | One/all-consumer cancellation, session deletion, socket disconnection, pre-META and midstream failure; no partial valid artifacts; live operations/readers/tasks return to zero |
| Restart/invalidation | New process loses sessions and reuses artifacts; mutated source rejects old pinned sessions and generates new keys; stopped-cache deletion permits rebuild |
| UI scientific view | Native vector and both matrix orientations, embedding, device-pixel extents, linked scrolling, exact hover value, green scalar ordering/contrast and linked amber guides, all 81 magnifier pixels |
| UI ownership | Native allocation/upload observers, one scalar texture representation, bounded uploads, no hover reupload, reader/texture baseline after repeated navigation/cancel; weak references plus explicit GC check retained CPU owners |
| Embeddings | Exact ordered IDs including duplicates, independent float32 upload oracle, first-row barrier, requested-row-only source reads, A→B→A and model/session fences, unsupported region isolation |
| Compact layout | 1000×700 and 390×640 at DPR 1/2 with real scrollbar gutters; all four overflow modes and four scroll corners, fixed document, independent inventory, aligned profiles |
| Prompt regression | Pre-embedding snapshots retained from issue #44; real prompt pixels unchanged across embedding completion, selection/history and composition lifecycle |
| Live tokenizer | Real Unicode echo/IDs, overlapping emoji code-point spans, one editable source, delayed actual old response cannot overwrite current annotations; refresh reconnects |

`server.py` wraps only the test process. Its `/__test/*` routes control delays and
faults and expose internal counters. They are absent from the production CLI.
Production algorithms, artifact storage, session ownership, LMEX framing and the
real tokenizer remain active. Fault injection raises in real production seams;
stale-token and embedding tests delay actual backend responses. A cancellation-aware
barrier in the real embedding consumer pauses after the first block; row-iterator
observers record materialized elements and full-tensor accesses without changing
source values. Embedding responses are ephemeral and publish no artifacts. Python
wire parsing and numeric oracles are independent of the service's encoder and calculations.

Timing reports distinguish first received DATA (TCP), first upload and populated
render (browser), the deliberate release barrier, and completion. These are
observations of an instrumented test, with no millisecond performance thresholds.
Framebuffer readback is off unless a test explicitly declares `capturePixels: true`.
The product suite's progressive geometry/inspection/cleanup case opts in through
its Playwright fixture; its matrix, profile and magnifier pixel oracles all remain
active. Architecture cases, reference smoke, DOM/geometry/lifecycle cases and
screenshot-only cases leave capture off. `installProbe` receives the option in the
same initialization script that installs its observers, before the bundle starts.
Scalar/count upload capture remains a separate, independently toggled observation.
Calling `pixel` while capture is off, before a valid draw, after a canvas reset,
with stale dimensions/DPR, or outside the snapshot throws a diagnostic error.
A live canvas retains one reusable readback buffer; weak keys and context teardown
release that test-owned evidence. Screenshots and failure traces are unchanged. GPU byte counts are allocated scalar texture bytes; browser/driver-internal copies
and total process/VRAM usage are not measured. Readback snapshots are test-only
display evidence. They are not application buffers. Correctness and explicit owner
cleanup are the gates.

## Optional local reference and CUDA

Supply a directory containing **HuggingFaceTB/SmolLM2-135M Base**:

```sh
export LMEX_REFERENCE_MODEL_DIR=/absolute/path/to/SmolLM2-135M
# Optional: export LMEX_REFERENCE_DEVICE=cuda:0
# Optional: export LMEX_WEBGL_BACKEND=vulkan
acceptance/check.sh
```

The directory must be an immediate model child of its parent; supported assets
must resolve inside that root. The operator is responsible for supplying the Base
checkpoint. Smoke validation rejects an explicitly Instruct identity and checks
the reference architecture. It enumerates actual descriptors and reads bounded
local samples using safetensors, streams the normalization vector, both MLP
orientations and embedding through real HTTP, compares those samples, and checks
real tokenizer IDs. The production UI opens those same tensors and verifies matrix
hover samples; the actual inventory/sample manifest is attached to browser evidence.
No weights are copied into Git or downloaded. Missing configuration produces SKIP;
a supplied invalid/unreadable directory fails.

For the accepted SmolLM2 Base plus SmolTalk LoRA pair, place both complete
checkpoints as immediate children of one local model root and run the focused
catalogue/session/architecture and production-browser gates:

```sh
export LMEX_LORA_REFERENCE_MODEL_ROOT=/absolute/path/to/models
backend/.venv/bin/python -m pytest -q acceptance/test_lora_reference.py
(cd ui && npm run test:acceptance -- --grep "SmolLM2 LoRA reference")
```

The Python gate checks the bare and composed sessions, all 60 `q_proj`/`v_proj`
branches, A/B inventory bindings, `alpha/r` and residual edges, and exact
streamed values for representative first-layer A/B factors against the local
Safetensors source. The browser gate navigates the generic graph and opens the
first-layer q/v A/B factors through the existing weight modal. Both commands
skip when `LMEX_LORA_REFERENCE_MODEL_ROOT` is unset; keep the local model root
outside the repository.

The CUDA test runs only when the installed PyTorch build and hardware expose CUDA;
otherwise it reports an explicit reason for SKIP. Use a compatible operator-managed
CUDA environment to run it. The repository's default Linux lock is CPU-only.
Neither a CPU pass nor a software WebGL pass is CUDA evidence.


For the validated GTX 1650 / driver 535 host, an explicit local CUDA override is:

```sh
uv pip install --python backend/.venv/bin/python 'torch==2.14.0+cu126' \
  --index-url https://download.pytorch.org/whl/cu126
backend/.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

This keeps the application version at 2.14.0 and selects its CUDA 12.6 build.
Use `backend/.venv/bin/python -m llm_model_explorer ... --device cuda:0` or
`uv run --no-sync` for that local installation: `uv sync`/`uv run --locked` will
restore the repository's CPU build. The acceptance runner uses the environment
directly. CUDA checks need host access to NVIDIA device nodes; a sandbox that hides
them cannot provide hardware evidence. The installed PyTorch wheel provides its
CUDA libraries; no driver replacement was needed for the recorded run. See
[PyTorch's installer](https://pytorch.org/get-started/locally/) and
[NVIDIA's CUDA 12.x compatibility requirements](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

The operator-requested reference installation can be reproduced separately from
the offline harness (using the Hugging Face CLI):

```sh
hf download HuggingFaceTB/SmolLM2-135M \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --local-dir /absolute/path/to/models/SmolLM2-135M
```

Navigable matrix and stale-tokenizer coverage is documented in [usability.md](usability.md).
Integrated explorer polish and expanded numeric coverage are documented in [polish.md](polish.md).
The earlier compact explorer and embedding evidence is in [improvements.md](improvements.md).
The earlier [evidence.md](evidence.md) remains the historical PoC report.

## Static Architecture Explorer

The acceptance stack also exercises all four static descriptions through real
HTTP and the built browser UI. See [architecture acceptance](architecture.md) for
the complete-local-reference manifest, strict no-skip gate, coverage boundaries,
and [current evidence](architecture-evidence.md). Ordinary fixture success does
not complete the actual Qwen/V-JEPA/SmolLM2 reference gate.

## Test harness timing

Backend and HTTP entrypoints print `pytest --durations=25`; JUnit and exit-code
ownership are unchanged. Each non-skipped product/architecture browser case adds
one `harness-timing` JSON attachment to the existing acceptance report, containing
fixture generation, backend spawn-to-ready, browser setup, test body and teardown
milliseconds, plus probe counters. Fixture generation for product fixtures is
measured inside the fresh Python subprocess (and is included in spawn-to-ready);
architecture fixtures measure the separate generation subprocess. A local
reference has no generated fixture and reports null. These phases must not be
added as if all were disjoint. Application first-DATA/upload/render measurements
remain separate and retain their original meanings.

`framebufferReadbacks`, `framebufferBytesRead`, `snapshotAllocations` and
`pixelQueries` explain probe cost. Disabled cases assert zero full-framebuffer
readbacks. Counters belong to the current document; navigation/reload installs a
fresh probe. No timings are CI pass/fail thresholds, and no server/session/cache
state is shared between tests. Existing Playwright JSON/HTML reports retain test
and hook durations for comparing the common workload. See the [optimization
measurements](test-harness-performance.md) for commands, coverage reconciliation,
and the baseline limitations.
