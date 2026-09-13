# LLM Model Explorer specification

This directory contains the current product and architecture specification for LLM Model Explorer.

The specification is split along the three system boundaries: backend, API contract, and UI. `product.md` owns the cross-cutting product scope. Documents under `backend/`, `api/`, and `ui/` own decisions inside those boundaries and must not duplicate or override each other.

## Documents

- [`product.md`](product.md): product intent, proof-of-concept scope, future direction, and non-goals.
- [`backend/architecture.md`](backend/architecture.md): backend runtime, technology choices, device execution, and deployment assumptions.
- [`backend/models.md`](backend/models.md): local Hugging Face discovery, model identity, lazy loading, and logical tensor materialization.
- [`backend/sessions-and-execution.md`](backend/sessions-and-execution.md): sessions, UI-driven execution, long operations, cancellation, concurrency, and GPU scheduling.
- [`backend/artifact-cache.md`](backend/artifact-cache.md): persistent disk artifacts, deduplication, publication, and invalidation.
- [`api/contract.md`](api/contract.md): API ownership, capabilities, typed operations, errors, and contract rules.
- [`api/binary-streaming.md`](api/binary-streaming.md): binary streaming semantics and common framing requirements.
- [`ui/architecture.md`](ui/architecture.md): React application architecture and separation from the WebGL2 renderer.
- [`ui/rendering.md`](ui/rendering.md): common tensor rendering rules and visual encoding.
- [`ui/tensor-explorer.md`](ui/tensor-explorer.md): reserved for the dedicated Tensor Explorer specification.
- [`ui/tokenizer-explorer.md`](ui/tokenizer-explorer.md): reserved for the dedicated Tokenizer Explorer specification.

## Authority

These documents describe accepted design decisions. Detailed Tensor Explorer and Tokenizer Explorer behavior is intentionally not specified here and must be completed from their dedicated design work rather than reconstructed or duplicated in this general specification.
