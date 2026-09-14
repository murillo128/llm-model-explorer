# Product specification

## Purpose

LLM Model Explorer is a browser-based interactive application for understanding transformer models by exposing their tensors and, in later phases, the individual computation steps involved in inference.

The product has three independent system boundaries: a backend that owns local model access and mathematical computation, an API contract that defines all communication between backend and UI, and a browser UI that owns visualization and interaction.

The initial reference model is `HuggingFaceTB/SmolLM2-135M` Base.

## Proof-of-concept scope

The first proof of concept contains two user-facing capabilities:

1. Tensor exploration: discover the tensors in a local Hugging Face model through a hierarchical list and open complete 1D or 2D tensors for visualization.
2. Tokenizer exploration: run the real Hugging Face tokenizer associated with the selected model and inspect its result.

The detailed behavior and visual design of both explorers are intentionally delegated to their dedicated specifications.

The proof of concept must establish the architecture that later inference exploration will use. It must not be a disposable implementation that requires replacing the backend, API model, session model, streaming mechanism, artifact cache, or renderer boundary when inference is added.

## Accepted post-PoC capability

Tokenizer Explorer also exposes the model input embedding matrix for its latest successful token sequence. For a sequence of `N` token IDs, the result has shape `[N, hidden_size]`; row `i` is the model input embedding for token sequence position `i`. Token order, repeated token IDs, and inserted special tokens are preserved.

This capability is limited to input embedding lookup. It does not include positional encoding, transformer forward execution, logits, generation, or any later inference stage.

## Future direction

A later phase will extend the same architecture to step-by-step inference: embeddings, transformer layers, attention, matrix/vector operations, activations, generation, KV cache, sampling, and other intermediate state. Execution remains driven by the UI: even a continuous Play mode is conceptually a sequence of steps requested by the UI rather than an autonomous backend process.

Reusable matrix and vector visualization primitives are expected to become building blocks for those later views.

## Product invariants

Tensor and weight values remain exact unless an explicit operation produces a different logical representation. Visualization must not mutate model values.

The backend owns mathematical truth and model-format knowledge. The frontend owns visual representation and interaction. The API is an independent contract between them.

Large numeric payloads are binary and stream progressively. The UI must be able to start consuming and displaying a result before the complete payload has arrived.

Multiple UI sessions may be active concurrently and independently.

## Out of scope for the proof of concept

The initial proof of concept does not implement transformer inference, attention visualization, KV cache inspection, model execution timelines, interactive matrix multiplication, zoom or pan, authentication, remote model downloads, non-Hugging-Face model formats, automatic cache garbage collection, or backwards-compatible API versioning.
