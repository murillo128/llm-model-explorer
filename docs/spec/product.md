# Product specification

## Purpose

LLM Model Explorer is a browser-based interactive application for understanding models through their tensors, navigable architecture, and, in later phases, individual inference computations. The initial focus is transformer language models; the explicitly selected V-JEPA 2 checkpoint is a bounded non-language architecture reference, not generic world-model support.

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

## Local adapter compositions and quantized checkpoints

A selectable logical model may consist of one complete local base checkpoint and one local PEFT adapter directory. Initially accept only a standard causal-LM LoRA adapter with `adapter_config.json` and Safetensors weights, under the restrictions in [local model admission](backend/models.md#local-peft-adapter-composition). Both asset sets remain local, read-only inputs. The runtime does not download bases or adapters and never executes checkpoint Python or custom code.

The selected PEFT references use the ordinary LoRA branch during evaluation: the base linear result plus the adapter's `A` and `B` projections scaled by `alpha/r`. The adapter weights are separate inspectable tensors when their representations are admitted. Do not materialize a merged checkpoint or expose a virtual delta-weight tensor. QLoRA describes this same accepted LoRA composition with a numerically admitted quantized base; it does not add another adapter format.

The initial quantized additions are saved bitsandbytes NF4 with double quantization and the exact compressed-tensors `pack-quantized` W4A16 INT4 layouts present in the GLM/Kimi references. The backend owns their validated physical layouts and logical float32 decoding. These selections do not establish generic bitsandbytes, AWQ, compressed-tensors, or arbitrary four-bit support.

## Architecture reference checkpoints and bounded coverage

| Local HF checkpoint variant (pinned repository revision) | Required role |
| --- | --- |
| `HuggingFaceTB/SmolLM2-135M` Base (`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`) | Existing-capability regression and small native-weight architecture/inspection reference. |
| `hfm8tr/smollm2-135m-smoltalk-lora` (`fb39c7012c3b125f94d2bb1a093254994053f4ca`) | Standard PEFT LoRA adapter over SmolLM2 Base. |
| `unsloth/SmolLM2-135M-bnb-4bit` (`738fd459cdfaf82f75846cf075738987ed6eb2b8`) | Saved bitsandbytes NF4 with double quantization and a compatible SmolLM2 Base identity. |
| `JunHowie/Qwen3-0.6B-GPTQ-Int4` | Compact Qwen3 language architecture; selected GPTQ Int4 storage. |
| `AxionML/Qwen3.5-0.8B-NVFP4` | Compact hybrid Qwen3.5 language architecture; selected NVFP4 storage. |
| `facebook/vjepa2-vitl-fpc64-256` | V-JEPA 2 encoder and predictor architecture; Transformers Safetensors variant. |
| `slowfastai/DeepSeek-V2-Lite-bnb-4bit` (`9fc357346aeba67950a34a86f3520fc276ca2daf`) | DeepSeek-V2-Lite multi-head latent attention and routed/shared MoE; saved NF4 double-quantized weights. |
| `cyankiwi/GLM-4.7-Flash-AWQ-4bit` (`25624b53414e585bcf7dcb9584667c3106c6089b`) | GLM-4.7-Flash MoE with config-ordered dense/sparse layers; selected compressed-tensors packed W4A16 INT4. |
| `cyankiwi/Kimi-Linear-48B-A3B-Instruct-AWQ-4bit` (`5d029d1844aa64ec302e14466be7d0353c6e697f`) | Kimi Linear's configured KDA/full-attention and MoE reference; selected compressed-tensors packed W4A16 INT4. |

The base and adapter revisions above are reproducible source references. Complete checkpoint weights may be downloaded for acceptance into a configured local model root, but are never repository artifacts. The product runtime has no automatic download feature. Record the actual local content fingerprints and byte inventories separately; a repository revision is not evidence that a complete local checkpoint was tested.

The supported PEFT composition is selected as a logical model in addition to each base checkpoint. Resolve its adapter's `base_model_name_or_path` against locally discovered compatible upstream identities. If more than one physical base variant is compatible, expose a distinct composition for each; native and quantized variants must not collide. A composition pins both contents, and replacing either source creates a different content identity.

The two compact Qwen checkpoints replace the exploratory requirement to use the latest Qwen and DeepSeek releases. They do not establish MoE or DeepSeek coverage; the separate DeepSeek-V2-Lite row above bounds that family coverage. Minimize installed checkpoint storage; do not require an extra unquantized copy. Repository size estimates are not acceptance evidence: record the exact revisions and local bytes actually used.

A complete local checkpoint means its configuration, all weight shards of the chosen variant, and the local assets required for advertised capabilities. It does not mean downloading all alternative formats, quantizations, or the V-JEPA 2 `original` weight copy. Metadata-only model directories, GGUF, remote-ID onboarding, automatic downloads, and execution of checkpoint Python code remain excluded.

For Qwen, detail only the language component. Other modalities may appear as identified context blocks with verified connections. V-JEPA 2 remains the sole non-language exception: both its visual encoder and predictor must be expandable to the agreed mathematical level. DeepSeek-V2, GLM-4.7-Flash, and Kimi Linear add bounded language-model family descriptions covering their verified attention, routing, expert, and configured layer-pattern behavior. Do not imply executed routing, token inference, or generation. An encoder-only V-JEPA graph does not meet that reference. A text tokenizer, vocabulary embedding, decoder stack, or language head is not required for every model.

The JEPA exception does not introduce video/image upload, playback or processing, visual inference, representation prediction, training losses, actions, planning, simulation, V-JEPA 2-AC, or generic JEPA/LeJEPA support. Do not add training or planning components absent from the selected model.

## Product invariants

Tensor and weight values remain exact unless an explicit operation produces a different logical representation. Visualization never mutates model values. The backend owns mathematical truth and storage-format knowledge; the frontend owns presentation; the API remains an independent contract.

Large numeric payloads are binary and progressive. Structural graphs contain metadata, not numeric tensor values. Multiple UI sessions may be active independently. Architecture availability, semantic completeness, tokenizer availability, and numeric-weight inspection are separate capabilities; unsupported architecture or quantization must not disable an otherwise supported existing capability.

## Future direction and exclusions

Later work may extend the same system to step-by-step inference, attention, matrix/vector computations, activations, generation, KV cache, sampling, and intermediate state. Inference remains UI-driven, including a future Play mode requesting successive steps. Static startup analysis is not such an inference step.

Inference, runtime attention maps, execution timelines, interactive matrix multiplication, authentication, remote model downloads, automatic cache garbage collection, public backwards-compatible API versioning, and non-HF formats are not added by the architecture increment. A diagram of attention is not a calculated attention map. No new quantization decoder or arbitrary-rank numeric slicing is required merely to describe architecture.
