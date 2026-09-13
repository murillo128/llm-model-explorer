# PoC integration acceptance — 2026-09-13

The real application passed CPU-fixture acceptance, host CUDA checks, and
operator-supplied SmolLM2-135M Base HTTP/UI smoke checks. This is implementation
acceptance evidence for issue #17, not a final audit or a default-branch merge.
Reproduction commands and coverage are in [README.md](README.md); the compact
machine-readable observations are in [evidence/host.json](evidence/host.json).

## Observed gates

| Gate | Result |
| --- | --- |
| Independent OpenAPI/schema/fixture validation | PASS: 121 references, 60 instance cases, 50 reproducible wire fixtures |
| Contract fixture and UI binding regeneration | PASS: no diff |
| Backend Ruff lint/format and mypy | PASS; 40 typed source/test files |
| Backend unit/integration | 462 passed, 0 failed, 0 skipped with the host CUDA installation |
| UI TypeScript/ESLint/unit/production build | PASS; 181 unit tests passed |
| Existing real WebGL2 component browser suite | 118 passed, 0 failed, 0 skipped; host Xvfb and SwiftShader |
| Real TCP integrated acceptance | 13 passed, 0 failed, 0 skipped, including CUDA and local Base smoke |
| Production UI + real backend | 10 passed, 0 failed, 0 skipped at DPR 1/2; NVIDIA Vulkan WebGL2, CPU fixtures and CUDA reference backend |

The initial CPU-only TCP run explicitly skipped CUDA. Subsequent hardware runs
used the user-requested CUDA installation and model; the table describes those
completed final runs. CI keeps the CPU lock and explicitly skips capabilities it
has not been supplied. Neither absent hardware nor an absent reference model is
counted as PASS.

The backend emitted two upstream Starlette TestClient deprecation warnings; these
did not affect results. Product network acceptance uses actual Uvicorn/TCP, not
TestClient. No application production behavior or normative API was changed.

## Environment and measurements

Python 3.12.14; Node 24.14.0; Playwright 1.63.0 / Chromium
153.0.8010.12; PyTorch 2.14.0+cu126 (CUDA 12.6); FastAPI 0.141.1,
Starlette 1.6.0, Uvicorn 0.52.4, Transformers 4.57.6, tokenizers 0.22.2.
Linux 6.8.0-139, x86_64. NVIDIA GeForce GTX 1650, 4 GiB, compute capability
7.5, driver 535.288.01. A real CUDA matrix multiplication also matched CPU output.
Fixture seed: **17**.

The product browser reported ANGLE / NVIDIA Vulkan 1.3.242. Actual maximum
texture size and renderbuffer size were **32,768**; maximum viewport dimensions
were **32,768 × 32,768**, at DPR **1** and **2**. The component/software runs
reported limits of 8,192. The reference embedding spans multiple native texture
bands and remains one complete `[49152,576]` logical tensor.

| Instrumented fixture measurement | DPR 1 | DPR 2 |
| --- | ---: | ---: |
| First scalar upload after opening | 137.8 ms | 110.1 ms |
| First populated render | 138.3 ms | 110.5 ms |
| Explicit producer barrier released | 229.2 ms | 145.6 ms |
| Complete UI result | 470.7 ms | 610.4 ms |

The TCP test separately observed first DATA at **10.6 ms**, before completion at
**100.0 ms**, using its explicit first-append barrier. These numbers include test
instrumentation and deliberate waiting; they are not latency promises or speedup
claims. Passing requires strict before-completion ordering, not a fixed duration.

For the fixture MLP view, allocation probes observed exactly **3 scalar textures**
and **4,383,744 scalar GPU bytes** (one matrix plus two uint32 profiles). Retained
CPU owners were 3,538,944, 230,400 and 614,400 bytes. Maximum observed upload input
was 65,536 bytes; maximum concurrent HTTP readers was 3. Hover made no scalar
upload/allocation. All 81 magnifier pixels matched the corresponding main pixels
within one display code; decoded luminance remained within the accepted 0.0045
tolerance. Repeated matrix/vector/embedding navigation and cancellation returned
live textures and readers to zero; explicit GC found no retained old scalar arrays.
Backend operations, consumers, readers, flights, producer tasks and temporary
writers returned to zero. A paused client may already have its remaining bytes
in kernel buffers; the test permits the backend to close its disk cursor early.

## Reference evidence and limits

The operator explicitly requested installation of `HuggingFaceTB/SmolLM2-135M`
Base. Download revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
Hugging Face checksum verification checked the nine downloaded model/tokenizer/doc
files. The unused `.gitattributes` was not required locally. Weights and HF cache
metadata remain outside Git.

The smoke enumerated the actual safetensors inventory and compared real HTTP
float32 samples against bounded local reads for the normalization vector,
`[576,1536]` down projection, `[1536,576]` up projection, and `[49152,576]`
embedding. The real production UI opened all four at both DPRs and checked matrix
hover values. Real Unicode tokenizer IDs were compared with local AutoTokenizer.
The reference backend used `cuda:0`; the mandatory generated-fixture backend
continued to use CPU.

**Observed browser limitation:** this host's Chromium 153 SwiftShader GPU process
segfaulted (exit 139) while rendering the full reference embedding, losing all three
WebGL contexts. It reproduced with test-only framebuffer readbacks disabled. The
UI correctly showed failed/incomplete rendering. Host NVIDIA Vulkan completed the
same smoke at both DPRs. This report does not claim successful reference-model
rendering on that SwiftShader build. Select `LMEX_WEBGL_BACKEND=vulkan` on this
host to reproduce the passing hardware run. Fixing Chromium/SwiftShader is outside
this application's scope; context loss remains an explicit supported failure state.

A sandboxed Xvfb also could not create WebGL2 contexts here; the identical probe
and the full 118-test suite passed outside that sandbox. No checks were waived.
Physical display/colorimeter measurements and cross-machine performance were not
performed. Separate-origin TCP/CORS acceptance establishes the deployment boundary;
operator hostnames/firewalls still need to be configured as documented.

## Compact visual evidence

[Matrix inspection](evidence/matrix.png) shows the real fixture stream's exact
native surface and 9×9 magnifier. [Live Unicode tokenization](evidence/tokenizer.png)
shows one editable source with IDs from the real backend. Full traces are retained
only for failed runs outside Git; CI uploads native reports and failure artifacts.
