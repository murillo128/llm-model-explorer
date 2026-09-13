# LLM Model Explorer

LLM Model Explorer is a browser-based interactive application for understanding transformer models by exposing their tensors and, in later phases, the individual computation steps involved in inference.

The project is intentionally split into three independent system boundaries: a Python backend that owns local model access and mathematical computation, a contract-first API that defines communication, and a browser UI that owns visualization and interaction.

## Current proof of concept

The initial reference model is `HuggingFaceTB/SmolLM2-135M` Base.

The first proof of concept has two user-facing capabilities:

- **Tensor Explorer**: discover model tensors through a hierarchical list and open complete 1D or 2D tensors for progressive visualization.
- **Tokenizer Explorer**: run the real Hugging Face tokenizer associated with the selected model and inspect its result.

The detailed behavior of both explorers lives in their dedicated specifications. The general architecture must already support the later move to step-by-step inference without replacing the backend model, API boundary, session model, streaming mechanism, artifact cache, or renderer boundary.

Future work will extend the same architecture to embeddings, transformer layers, attention, matrix/vector operations, activations, token generation, KV cache, sampling, and other intermediate inference state. Execution remains UI-driven: even a continuous Play mode is conceptually a sequence of explicit steps requested by the UI rather than an autonomous backend process.

## System architecture

The **backend** is Python with FastAPI/Starlette for HTTP and PyTorch for tensor computation. CUDA is the optimized path when configured and available; CPU remains supported with the same logical API behavior. Local Hugging Face models are discovered below a configured model root, opened lazily, and treated as read-only.

The **API** is a first-class independent contract. Ordinary control and metadata operations use typed HTTP/JSON contracts, while large numeric results use binary HTTP streaming. Backend and UI evolve together against the current contract; public backwards-compatible API versioning is not a proof-of-concept requirement.

The **UI** is React + TypeScript + Vite. WebGL2 rendering lives behind a reusable renderer boundary independent from React. Backend and UI are separately deployable and may run on different computers.

## Data, streaming, and rendering invariants

Large numeric payloads are binary and progressive. The UI must be able to consume and display data before the complete result has arrived. Potentially expensive work uses explicit cancelable long operations with an `operation_id`; the same HTTP response that starts an operation carries its progressive result.

The backend owns physical model-format knowledge. The main visualization path exposes logical tensor values in canonical `float32`, so the UI does not need to implement NF4, INT8, or other quantization decoders. Model values remain authoritative and visualization must not mutate them.

The proof-of-concept tensor renderer follows an exact spatial rule: **one weight equals one rendered pixel**. Matrices are not fit to the viewport, resampled, or aggregated; oversized content uses normal scrolling. Weight value is encoded through luminosity using a configurable nonlinear sigmoid-like transfer based on robust tensor statistics. Color remains an independent semantic channel for later uses such as selection, activations, clusters, and highlighting.

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

The product architecture specification is established; the application implementation has not yet been scaffolded.
