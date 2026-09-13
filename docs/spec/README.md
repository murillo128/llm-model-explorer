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
- [`ui/visual-language.md`](ui/visual-language.md): cross-cutting UI visual language, shared composition rules, typography, spacing, neutral surfaces, and interaction styling.
- [`ui/tensor-explorer.md`](ui/tensor-explorer.md): accepted Tensor Explorer behavior, Matrix Inspector geometry, hover/selection, magnifier, and scientific-view guardrails.
- [`ui/tokenizer-explorer.md`](ui/tokenizer-explorer.md): accepted Tokenizer Explorer behavior and the reusable live prompt/tokenization surface.

## Authority

These documents describe accepted design decisions. Cross-cutting UI styling belongs in `ui/visual-language.md`; component-specific behavior and visual semantics belong in the dedicated component specification. When a generic visual-language rule and a component-specific accepted rule disagree, the more specific component document controls that component.
