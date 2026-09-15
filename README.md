# LLM Model Explorer

LLM Model Explorer is a browser-based interactive application for understanding transformer models by exposing their tensors and, in later phases, the individual computation steps involved in inference.

The project is intentionally split into three independent system boundaries: a Python backend that owns local model access and mathematical computation, a contract-first API that defines communication, and a browser UI that owns visualization and interaction.

## Current proof of concept

The initial reference model is `HuggingFaceTB/SmolLM2-135M` Base.

The implemented application retains the two proof-of-concept capabilities and adds the accepted static architecture view:

- **Tensor Explorer**: discover model tensors through a hierarchical list and open complete 1D or 2D tensors for progressive visualization.
- **Architecture Explorer**: inspect a prepared static graph, expand concrete layers, show dimensions, and inspect available native weights in the existing Matrix Explorer modal. Descriptions cover selected Qwen3, Qwen3.5, V-JEPA 2 encoder/predictor, and SmolLM2 variants; [local checkpoint acceptance and evidence](acceptance/architecture.md) are tracked separately from implementation.
- **Tokenizer Explorer**: run the real Hugging Face tokenizer associated with the selected model, inspect its inline annotations, and progressively view the exact input-embedding rows for the current token sequence.

The detailed behavior of these explorers lives in their dedicated specifications. The general architecture must already support the later move to step-by-step inference without replacing the backend model, API boundary, session model, streaming mechanism, artifact cache, or renderer boundary.

Future work will extend the same architecture to execution of transformer layers, attention and matrix/vector operations, plus activations, token generation, KV cache, sampling and other intermediate inference state. Execution remains UI-driven: even a continuous Play mode is conceptually a sequence of explicit steps requested by the UI rather than an autonomous backend process.

## System architecture

The **backend** is Python with FastAPI/Starlette for HTTP and PyTorch for tensor computation. CUDA is the optimized path when configured and available; CPU remains supported with the same logical API behavior. Local Hugging Face models are discovered below a configured model root, opened lazily, and treated as read-only.

The **API** is a first-class independent contract. Ordinary control and metadata operations use typed HTTP/JSON contracts, while large numeric results use binary HTTP streaming. Backend and UI evolve together against the current contract; public backwards-compatible API versioning is not a proof-of-concept requirement.

The **UI** is React + TypeScript + Vite. WebGL2 rendering lives behind a reusable renderer boundary independent from React. Backend and UI are separately deployable and may run on different computers.

## Data, streaming, and rendering invariants

Large numeric payloads are binary and progressive. The UI must be able to consume and display data before the complete result has arrived. Potentially expensive work uses explicit cancelable long operations with an `operation_id`; the same HTTP response that starts an operation carries its progressive result.

The backend owns physical model-format knowledge. The main visualization path exposes logical tensor values in canonical `float32`, so the UI does not need to implement NF4, INT8, or other quantization decoders. Model values remain authoritative and visualization must not mutate them.

The tensor renderer preserves **one authoritative scalar per square matrix cell**. Matrix Explorer defaults to fit width, supports focal-point wheel/pinch zoom with a native 1:1 minimum, and uses scrollbars for navigation. Exact indexed sampling never interpolates or aggregates weights. Weight value is encoded through luminosity using a configurable nonlinear sigmoid-like transfer based on robust tensor statistics. Scientific surfaces use sequential green brightness. Amber inspection guides are composited over that display without changing the scalar transfer or stored values.

## Sessions and artifact cache

Multiple UI sessions may be active concurrently. Each session is bound to one model and its logical state is held in backend memory for the proof of concept. Expensive GPU work is serialized through one execution queue per GPU device, while disk reads, cache hits, tokenization, and HTTP streams may proceed concurrently.

Derived reusable results are stored in a shared filesystem artifact cache. Cached artifacts are complete, immutable, reconstructible, and keyed by all inputs that determine their content, including the model content fingerprint. Chunks are transport units, not cache units. The cache is disposable: with the backend stopped, the entire cache directory may be deleted and rebuilt on demand.

## Specification

The accepted product and architecture specification lives under [`docs/spec/`](docs/spec/README.md):

- [`docs/spec/product.md`](docs/spec/product.md) — product scope, proof-of-concept boundaries, and future direction.
- [`docs/spec/backend/`](docs/spec/backend/) — backend runtime, models, sessions/execution, and artifact cache.
- [`docs/spec/api/`](docs/spec/api/) — API contract and binary streaming protocol.
- [`docs/spec/ui/`](docs/spec/ui/) — UI architecture, rendering rules, visual language, and dedicated explorer specifications.
- [`docs/spec/ui/visual-language.md`](docs/spec/ui/visual-language.md) — the shared light editorial UI language, composition rules, typography, spacing, and interaction styling used across the application.

Tensor Explorer and Tokenizer Explorer behavior is owned by `docs/spec/ui/tensor-explorer.md` and `docs/spec/ui/tokenizer-explorer.md`. Cross-cutting UI styling is owned by `docs/spec/ui/visual-language.md`.

## Repository workflow

This repository uses the Skillforge issue-driven development workflow. Durable accepted design belongs in repository documentation, bounded implementation work belongs in GitHub issues, and non-trivial implementation follows the repository agent and review workflow defined in [`AGENTS.md`](AGENTS.md).

## Run on separate computers

Use Python 3.12 (3.14 is also checked), uv 0.12.13, Node 24.14.x and npm 11.9.x.
The lockfiles pin application dependencies. Models are supplied locally; the
application never downloads weights. Place each HF checkpoint in an immediate
child directory of the model root, including its configuration, all selected weight shards,
and assets required for advertised capabilities. Native numeric views support F32,
F16 and BF16 safetensors. The admitted GPTQ Int4 and ModelOpt NVFP4 layouts expose
decoded logical float32 weights through the same numeric views. Inventory coverage
is complete only when all records are accounted for; unresolved storage remains
explicitly partial. Input embeddings support the accepted Llama/SmolLM2, Qwen3 and
Qwen3.5 text layouts when their tables are actionable. A tokenizer is optional for
V-JEPA 2. See [integrated polish acceptance](acceptance/polish.md) for evidence and limits.
Startup prepares architectures before accepting connections, including content hashing
on warm starts; graph retrieval never generates a model or runs inference.

On the backend computer:

```sh
cd backend
uv sync --locked --python 3.12
uv run --locked llm-model-explorer-backend \
  --model-root /srv/models \
  --cache-dir /srv/lmex-cache \
  --device cpu --host 0.0.0.0 --port 8000 \
  --cors-origin http://ui-computer:8080
```

`--model-root` and `--cache-dir` are required and must not overlap. `--device`
accepts `cpu`, `cuda`, or `cuda:N`; CUDA needs a compatible operator-installed
PyTorch/CUDA runtime (the default lock installs CPU PyTorch on Linux).
`--host` defaults to `127.0.0.1`, `--port` to `8000`. Repeat `--cors-origin`
for each exact allowed UI origin, including its port; it defaults to no allowed
cross-origin UI. Do not put a path or trailing slash in an origin.

On the UI computer, from the repository root:

```sh
cd ui
npm ci
npm run build
printf '%s\n' '{"backend_base_url":"http://backend-computer:8000"}' > dist/runtime-config.json
python3 -m http.server 8080 --bind 0.0.0.0 --directory dist
```

Open `http://ui-computer:8080`. The runtime URL must be reachable **from the
browser**, and can be changed without rebuilding. Serve `runtime-config.json`
alongside `index.html`; avoid caching that deployment file. This uses different
origins directly, with CORS and exposed operation IDs, without an API proxy.
For local UI development use `npm run dev` and edit `ui/public/runtime-config.json`;
allow the resulting Vite origin explicitly on the backend.

The PoC has **no authentication** and is intended for a trusted network or otherwise
trusted environment. Model files are read-only. Sessions survive refresh while
the backend stays alive, but backend restart discards them. Complete artifacts
survive restart. To rebuild the disposable cache, stop the backend, manually delete
the configured cache directory (for the example: `rm -rf /srv/lmex-cache`), and
restart it. There is no automatic cache eviction or cache-management UI.

## Check the application

The combined command is `acceptance/check.sh` after the dependency setup in
[acceptance/README.md](acceptance/README.md). It runs independent OpenAPI validation
and reproducible fixture/binding generation, backend lint/type/unit/integration
checks, UI lint/type/unit/build checks, and real-network/WebGL2 acceptance.
Individual entry points:

```sh
# With the separate API tool environment installed:
api/.venv/bin/python api/validate_contract.py
(cd backend && uv run --locked ruff check . && uv run --locked ruff format --check . && uv run --locked mypy && uv run --locked pytest)
(cd ui && npm run check && xvfb-run -a npm run test:browser)
backend/.venv/bin/python -m pytest acceptance -ra
(cd ui && npm run build && xvfb-run -a npm run test:acceptance)
```

[The reproducibility report](acceptance/evidence.md) distinguishes fixture evidence
from optional operator-supplied `HuggingFaceTB/SmolLM2-135M` Base and CUDA evidence.
A missing reference directory or CUDA capability is reported as **SKIP**. No
reference weights are downloaded by the harness.

Local discovery, all eleven API operations, sessions, complete artifact caching,
progressive tensor/statistics/distribution delivery, both explorers, input-embedding
row lookup, and matrix inspection are implemented. The compact shell keeps inventory and scientific
scrolling inside the workspace. See the [integrated improvement evidence](acceptance/improvements.md)
for native scrollbar, prompt-regression, embedding-value, and lifecycle checks.
Inference, attention, KV cache, matrix multiplication,
FFT/SVD, clustering, and quantization inspection remain future work. Final epic
acceptance and default-branch integration are separate workflow decisions.
See [backend](backend/README.md), [UI](ui/README.md), and [API](api/README.md)
for component-specific commands and implementation details.
