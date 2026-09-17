# Product specification

## Purpose

LLM Model Explorer is a browser-based interactive application for understanding models through their tensors, navigable architecture, and, in later phases, individual inference computations. The initial focus is transformer language models; the explicitly selected V-JEPA 2 checkpoint and project-local BDB-2025 LeWorldModel export are bounded non-language architecture references, not generic world-model support.

The product has three independent system boundaries: a backend that owns local model access and mathematical computation, an API contract that defines communication, and a browser UI that owns visualization and interaction.

The initial reference model is `HuggingFaceTB/SmolLM2-135M` Base.

## Proof-of-concept scope

The first proof of concept contains two user-facing capabilities:

1. Tensor exploration: discover tensors in a local Hugging Face model and open complete 1D or 2D tensors for visualization.
2. Tokenizer exploration: run the real Hugging Face tokenizer associated with the selected model and inspect its result.

Their detailed behavior belongs in their dedicated specifications. The proof of concept must remain evolvable without replacing the backend, API boundary, session model, streaming mechanism, artifact cache, or renderer boundary.

## Accepted post-PoC capabilities

Tokenizer Explorer exposes the input embedding matrix for its latest successful token sequence. For `N` token IDs, the result has shape `[N, hidden_size]`; row `i` is the input embedding for sequence position `i`. Preserve order, duplicate IDs, and inserted special tokens. This is embedding lookup only, not positional encoding, a transformer forward pass, logits, or generation.

Matrix exploration supports viewport zoom and navigation while preserving exact logical values, row/column identity, native order, and square cells. Navigation changes the view, never the tensor.

Architecture Explorer adds a static, exhaustive, navigable description of supported local checkpoints. It explains components, recognizable mathematical operations, their data dependencies, known/symbolic dimensions, and references to weights. It reuses the existing explorers rather than replacing them. Its domain rules belong in [architecture analysis](backend/architecture-analysis.md), its transport in the [architecture API contract](api/architecture-explorer.md), and its interaction in the [Architecture Explorer specification](ui/architecture-explorer.md).

Implementation and reference-checkpoint validation status are recorded in [integrated acceptance evidence](../../acceptance/architecture-evidence.md). It performs no inference, tracing, activation capture, token generation, or interactive execution. Future links to other explorers are represented by semantic resource references, not fictional execution results or unimplemented endpoints.

## Architecture reference checkpoints and bounded coverage

| Local HF checkpoint variant | Required role |
| --- | --- |
| `JunHowie/Qwen3-0.6B-GPTQ-Int4` | Compact Qwen3 language architecture; selected GPTQ Int4 storage. |
| `AxionML/Qwen3.5-0.8B-NVFP4` | Compact hybrid Qwen3.5 language architecture; selected NVFP4 storage. |
| `facebook/vjepa2-vitl-fpc64-256` | V-JEPA 2 encoder and predictor architecture; Transformers Safetensors variant. |
| `HuggingFaceTB/SmolLM2-135M` Base | Existing-capability regression and a small native-weight architecture/inspection reference. |
| `nfl_world_model` / `BDB2025LeWorldModel` export from `murillo128/nfl-world-model` | Bounded structured NFL encoder, causal temporal state and physical-horizon latent predictor reference. Synthetic weights are acceptable for structural inspection and are not scientific evidence. |

The two compact Qwen checkpoints replace the exploratory requirement to use the latest Qwen and DeepSeek releases. They do not establish MoE or DeepSeek coverage. Minimize installed checkpoint storage; do not require an extra unquantized copy. Repository size estimates are not acceptance evidence: record the exact revisions and local bytes actually used.

A complete local checkpoint means its configuration, all weight shards of the chosen variant, and the local assets required for advertised capabilities. It does not mean downloading all alternative formats, quantizations, or the V-JEPA 2 `original` weight copy. Metadata-only model directories, GGUF, remote-ID onboarding, automatic downloads, and execution of checkpoint Python code remain excluded.

For Qwen, detail only the language component. Other modalities may appear as identified context blocks with verified connections. V-JEPA 2 and the BDB-2025 LeWorldModel are explicit bounded exceptions. V-JEPA 2 must expose its visual encoder and predictor. The BDB-2025 export must expose its structured entity/frame projections, same-time spatial Transformer, causal temporal Transformer, current play latent and physical-horizon latent predictor. A text tokenizer, vocabulary embedding, decoder stack, or language head is not required for either non-language family.

The V-JEPA exception does not introduce video/image upload, playback or processing, visual inference, representation prediction, training losses, actions, planning, simulation, V-JEPA 2-AC, or generic JEPA/LeJEPA support. The BDB-2025 exception likewise does not introduce NFL ingestion, training, live inference, probes, actions, planning or generic world-model discovery. Its reviewed training-only target/loss/SIGReg structure may be described as static context when present in the selected model, but it must be visibly separated from the reusable live inference path and must never imply future-observation access at inference time.

The existing API scope value `visual_encoder_predictor` remains the historical transport identifier for these bounded non-language encoder/predictor graphs. For `nfl_world_model` it does not assert visual input; the graph's input nodes and attributes remain authoritative and describe structured NFL tracking tensors.

## Product invariants

Tensor and weight values remain exact unless an explicit operation produces a different logical representation. Visualization never mutates model values. The backend owns mathematical truth and storage-format knowledge; the frontend owns presentation; the API remains an independent contract.

Large numeric payloads are binary and progressive. Structural graphs contain metadata, not numeric tensor values. Multiple UI sessions may be active independently. Architecture availability, semantic completeness, tokenizer availability, and numeric-weight inspection are separate capabilities; unsupported architecture or quantization must not disable an otherwise supported existing capability.

## Future direction and exclusions

Later work may extend the same system to step-by-step inference, attention, matrix/vector computations, activations, generation, KV cache, sampling, and intermediate state. Inference remains UI-driven, including a future Play mode requesting successive steps. Static startup analysis is not such an inference step.

Inference, runtime attention maps, execution timelines, interactive matrix multiplication, authentication, remote model downloads, automatic cache garbage collection, public backwards-compatible API versioning, and non-HF formats are not added by the architecture increment. A diagram of attention is not a calculated attention map. No new quantization decoder or arbitrary-rank numeric slicing is required merely to describe architecture.
