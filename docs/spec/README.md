# LLM Model Explorer specification

This directory contains the current accepted product and architecture specification for LLM Model Explorer. Accepted requirements are not a claim that their implementation or checkpoint validation has shipped. Execution scope and activation belong to the controlling issues.

The specification is split along the three system boundaries: backend, API contract, and UI. `product.md` owns cross-cutting product scope. Documents under `backend/`, `api/`, and `ui/` own decisions inside those boundaries and must not duplicate or override each other.

## Documents

- [`product.md`](product.md): product intent, accepted capabilities, reference checkpoints, future direction, and non-goals.
- [`backend/architecture.md`](backend/architecture.md): runtime, deployment, and blocking architecture preparation at startup.
- [`backend/models.md`](backend/models.md): local checkpoint admission, identity, lazy access, and physical/logical representations.
- [`backend/architecture-analysis.md`](backend/architecture-analysis.md): static semantic analysis, family coverage, parameter binding, and verification requirements.
- [`backend/sessions-and-execution.md`](backend/sessions-and-execution.md): sessions, prepared architecture lookup, explicit computation, cancellation, and scheduling.
- [`backend/artifact-cache.md`](backend/artifact-cache.md): numeric and structured artifacts, identity, publication, and invalidation.
- [`api/openapi.yaml`](api/openapi.yaml): published machine-readable HTTP endpoint, schema, identifier, and status-code contract.
- [`api/contract.md`](api/contract.md): API ownership and shared HTTP, tensor, tokenizer, embedding, and architecture semantics.
- [`api/architecture-explorer.md`](api/architecture-explorer.md): accepted architecture endpoint, graph records, capability separation, bounds, and conformance requirements.
- [`api/binary-streaming.md`](api/binary-streaming.md): binary framing, payload layouts, and terminal behavior.
- [`ui/architecture.md`](ui/architecture.md): React composition, explorer integration, and renderer boundaries.
- [`ui/rendering.md`](ui/rendering.md): tensor rendering rules and visual encoding.
- [`ui/visual-language.md`](ui/visual-language.md): shared shell, typography, spacing, surfaces, and interaction styling.
- [`ui/tensor-explorer.md`](ui/tensor-explorer.md): tensor navigation, matrix geometry, camera, inspection, and scientific guardrails.
- [`ui/tokenizer-explorer.md`](ui/tokenizer-explorer.md): live prompt/tokenization and input-embedding inspection.
- [`ui/architecture-explorer.md`](ui/architecture-explorer.md): global graph navigation, grouping, dimensions, and modal inspection.

## Authority and delivery status

These documents describe accepted design. `api/openapi.yaml` remains authoritative for the published HTTP schemas; `api/binary-streaming.md` owns streaming bytes. `api/contract.md` owns their shared semantics. The Architecture Explorer extension is accepted but pending implementation. Its dedicated API document defines the target contract; the contract-publication child must incorporate it into OpenAPI together with conformance fixtures and generated bindings before backend/UI consumers ship. This documentation-only adoption does not claim that the new endpoint already exists or silently invalidate the current generated bindings.

Architecture Explorer is a separate increment, not an expansion of an already executing issue. Existing matrix-camera and tokenizer-continuity requirements remain in force. Its graph camera does not replace or constrain the Matrix Explorer camera.

Cross-cutting styling belongs in `ui/visual-language.md`; component behavior belongs in its dedicated specification. A specific accepted component rule controls over a generic visual-language rule for that component.
