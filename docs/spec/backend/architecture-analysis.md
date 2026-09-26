# Static architecture analysis

## Responsibility and scope

This document owns the accepted Architecture Explorer analysis capability. [Product scope](../product.md#architecture-reference-checkpoints-and-bounded-coverage) owns the reference checkpoints and exclusions; [models.md](models.md) owns checkpoint admission and representations; [the API document](../api/architecture-explorer.md) owns graph records; [startup](architecture.md#blocking-architecture-preparation) and [cache](artifact-cache.md#structured-architecture-artifacts) own lifecycle.

Construct a semantic graph from guarded local configuration, indexes, Safetensors headers, and packaged, versioned architecture descriptions. Descriptions are verified against identified revisions of original model implementations during development. They encode mathematical relationships, not pixel layouts, hand-drawn per-checkpoint diagrams, or a second implementation of inference.

Production analysis must not call `forward()` or `generate()`, trace a real or dummy execution, require a prompt/video, load all weights into RAM/GPU, run preprocessors, execute Python in a checkpoint directory, or access the network. It must not instantiate an entire numerical model merely to enumerate its modules. Reading file bytes for the existing content fingerprint is permitted and bounded.

TorchLens, TransformerBridge, and an upgrade of Transformers are not prerequisites for this static increment. Existing compatible code/knowledge may be reused with provenance and license attribution. Inference/instrumentation dependencies can be selected by a later design; do not impose their numerical loading path on this analysis.

## Model-supplied definitions

The optional data-only `architecture.json` path is defined by
[model-owned architecture](model-owned-architecture.md). It takes precedence over
packaged selection and uses the same graph/binding validator. It describes what
the author declares, with explicit model-supplied provenance, rather than claiming
a packaged source review. Arbitrary operation names remain static explanatory
records. No checkpoint Python is imported and no numerical model is constructed.

## Description selection and coverage

Select descriptions by supported configuration discriminators, architecture classes, relevant options, per-layer types, and checked storage bindings, not a marketing name or one matching tensor suffix. In addition to the existing families, the bounded discriminators include `deepseek_v2` / `DeepseekV2ForCausalLM`, `glm4_moe_lite` / `Glm4MoeLiteForCausalLM`, and `kimi_linear` / `KimiLinearForCausalLM`, for the pinned reference variants in [product scope](../product.md#architecture-reference-checkpoints-and-bounded-coverage). Normalize known configuration defaults only from the reviewed reference implementation. Unrecognized fields that can change structure, contradictory metadata, or missing required parameters must prevent a complete-coverage claim. Unrelated training metadata need not invalidate an otherwise recognized architecture.

A result is complete only within its declared scope and mathematical abstraction. Partial results identify unknown regions and their reasons while preserving verified structure. Do not guess edges from module registration order, silently substitute a Llama template, or treat an opaque required block as full support. Unavailable means no valid useful graph could be constructed; never return an empty graph labelled successful.

A valid partial graph is a complete serialized artifact with explicitly incomplete semantic coverage. Interrupted generation is not a partial graph. These distinctions drive the API states and cache publication rules.

## Mathematical abstraction

Represent linear and patch projections, normalizations, positional encoding, attention products, softmax, activations, multiplication between branches, residual addition, selection, and meaningful reshape/split/concat/transpose operations. Preserve biases and gates where relevant. Treat recognizable normalization/activation/state-update algorithms as operations with explanatory attributes/formulas rather than expanding every arithmetic instruction, cast, or allocation.

Edges mean verified data dependencies. Constants are checked against configuration and storage. Batch, sequence, spatial, temporal, context, and target dimensions may be symbolic; unresolved dimensions stay explicitly unknown. No displayed dimension or state tensor implies that an input was executed. Recurrent dependencies use symbolic prior/next-state ports, not an invented captured KV cache or an unrolled inference timeline.

Groups describe model hierarchy and actual instances. Every repeated layer retains identity, sequence index, parameters, and any structural exceptions. The backend validates the full semantic instance graph. Repetition metadata enables compact presentation; verified routed-expert families may use the API's compact wire definition and explicit instance maps. A description of a representative layer never substitutes for bindings of all instances.

For compact routed experts, construct and validate the complete source graph within a separate bounded working budget. Compare each instance with a reviewed prototype after substituting only explicit node/edge/parameter IDs, source prefix, shape-symbol names, root label and configured expert index. Publish a family only when every reconstructed node and edge equals the source record and every parameter role resolves to its exact source-scoped name. Unknown or exceptional structure remains explicit. The decoded response still has the existing 32 MiB limit; the larger bounded working graph does not raise that limit or permit incomplete cache publication.

Reviewed dense Qwen3/Llama, hybrid Qwen3.5 and V-JEPA encoder/predictor layers
emit explicit Attention and MLP groups. Membership is authored in the packaged
mathematical description: head preparation, Q/K normalization, rotary treatment,
gating, state updates, activations and projections stay in their actual component.
Layer norms and residual additions outside that component remain in the layer.
Reuse an existing component group rather than wrapping it again. Every crossing
uses an exact group port; preserve auxiliary/unused interfaces, fan-out and each
symbolic state independently. No operation or tensor binding changes merely to
introduce a boundary.

Use the API's optional `semantic_role` annotation for component/operation roles
and concise source-backed labels. Retain semantic source keys in description
provenance (`rule: Semantic source key in the reviewed packaged description`)
so inspection and cross-revision comparisons do not depend on display labels.
The bounded role mapping in a packaged description names its authored operations;
it does not infer membership from checkpoint tensor prefixes. A composite that
does not correspond to a framework module must have description-derived
provenance and no invented module reference. Semantic document changes bump the
affected description revision and use ordinary startup/cache validation; all
graph-local hashes may change while logical tensor identities remain unchanged.

## Optional component equivalence annotations

Packaged descriptions may opt into the API's verified shared structures. Open
candidate collection explicitly while constructing a known Attention or MLP
component, and assign relative roles from that description's authored construction
keys. Do not discover candidates by scanning checkpoint prefixes, display labels,
or frontend patterns. Dense Attention/MLP, hybrid full/linear Attention and MLP,
and each V-JEPA stack are separately declared families within one loaded model
and reviewed description. The complete semantic comparison required by the API
must still pass; a common name never proves correspondence.

Keep this metadata path removable. A singleton or unverified candidate remains an
ordinary component, and optional budget exhaustion cannot consume source graph
records or downgrade mathematical coverage. Construction and validation use one
bounded graph index and component-local mappings, not another full checkpoint or
graph. Never read weights, execute operations or infer weight equality to decide
a family. Parameter bindings and logical tensor identities stay attached to the
real instance, including verified aliases and unavailable inspections.

The current annotation definition is `exact-component-roles-1`. The analyzer core
revision and generated graph-schema revision invalidate structured startup cache
entries; numeric artifact keys and logical tensor identities are unchanged.

## Dense language reference coverage

Qwen3 must preserve its actual embedding/output relationships, decoder order, Q/K normalization, grouped-query attention, positional encoding, attention projection, gated MLP branches, and residual paths. SmolLM2 Base provides the corresponding small Llama-family regression path; do not apply Qwen-specific normalization or gate assumptions to it. Tied parameters must be represented as aliases where supported by configuration and inventory, not reported missing or duplicated as independent weights.

The tokenizer is a context node outside the neural model when one exists. Its presence is a capability reference, not a tokenization operation performed by analysis. No general requirement for text tokenization or vocabulary logits is imposed on the graph core.

## Hybrid Qwen3.5 coverage

Preserve the exact configured order of linear-attention and full-attention layers. Both required interiors must be inspectable at the mathematical level; merely identifying a `GatedDeltaNet` class does not meet coverage.

For full attention, preserve the variant's query/gate split, Q/K normalization, rotary treatment, attention calculation, output gating, and projection. For linear attention, represent verified input projections, local mixing/convolution, branch/gate transformations, normalization where applicable, recurrent/DeltaNet state update, and output projection with their actual dependencies. Attribute variant-specific details to the reviewed source rather than assuming one universal formula. Show symbolic state inputs/outputs without running the recurrence.

The language MLP and residual paths remain explicit. Qwen visual/audio/video components are context only; their internal operations are outside the required detailed scope. A context-only modality does not make complete *language-scope* coverage partial.

## V-JEPA 2 coverage

The selected Transformers checkpoint must expose both encoder and predictor. Verify the actual configuration and inventory; the reviewed candidate declares 24 encoder layers and 12 predictor layers, which are reference expectations to check, not constants to impose on other variants.

Represent symbolic visual input, spatial/temporal patch projection, positional information, encoder blocks, context selection at the verified point, target position/mask inputs, predictor projection/blocks, and output representations. Preserve attention, normalization, MLP, residual and selection paths inside both stacks. Review the actual implementation rather than pasting a training diagram: context masking must not be placed at an assumed location.

No text tokenizer, vocabulary head, target/EMA teacher, training loss, classifier, action conditioner, or planner may be invented. An encoder-only checkpoint or a predictor whose parameters do not match the declared configuration does not satisfy this reference. No image/video execution or preprocessing is needed for any acceptance assertion.

Patch-projection parameters may have rank above two. Preserve their full shapes and bindings; mark numeric modal inspection unsupported at that rank. Do not flatten, slice, or hide them to make a matrix fit. Native rank-1/rank-2 weights use the existing inspection path.

## DeepSeek-V2-Lite coverage

Require `model_type="deepseek_v2"`, `architectures` containing `DeepseekV2ForCausalLM`, and a compatible configuration and inventory. For the pinned reference, verify 27 ordered decoder layers, one initial dense MLP layer (`first_k_dense_replace=1`), then routed MoE layers at the declared `moe_layer_freq=1`; verify 64 routed experts, top six, two shared experts, and their complete gate/up/down weights. The source stores the two shared experts as one combined shared MLP whose intermediate width is `n_shared_experts * moe_intermediate_size`; preserve that actual branch and binding. Do not assume these counts for another variant. The pinned router uses softmax scores, greedy top-six selection, `norm_topk_prob=false`, and routed scaling from configuration; describe its configured behavior rather than a generic softmax router. Preserve the selected expert SwiGLU computations and weighted reduction, plus the separate shared-MLP output added to the routed result. Routing is symbolic: the analyzer chooses no experts and executes no tokens.

Describe the reference's Multi-head Latent Attention from its actual ranks and projection inventory: direct query projection when `q_lora_rank` is null (or the supported low-rank query path only when the config and weights verify it); split non-positional and rotary query dimensions; the `kv_a_proj_with_mqa` latent and rotary-key branches; latent normalization; `kv_b_proj` expansion into non-positional key and value heads; rotary application; causal attention; output projection. Preserve the shared rotary-key path and latent key/value state as separate symbolic shapes. Verify `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, head counts, RoPE scaling and routing options instead of copying reference constants into other variants. A `q_lora_rank` or inventory contradiction is unsupported coverage, not permission to select a nearby template.

## GLM-4.7-Flash coverage

Require `model_type="glm4_moe_lite"`, `architectures` containing `Glm4MoeLiteForCausalLM`, and a compatible configuration and inventory. The pinned reference has 47 ordered layers and supplies the `mlp_layer_types` sequence explicitly; retain that sequence, including the initial dense layer and every subsequent sparse layer, rather than reconstructing it from layer count. Verify 64 routed experts, top four, one shared expert, and the complete expert and shared-MLP parameters.

Describe its MLA query A projection, normalization and B projection using the configured rank; split query into non-positional and rotary components. Describe the shared compressed key/value A projection, latent normalization, B expansion into keys/values, configured RoPE/interleave behavior, causal attention and output projection. For sparse layers preserve the source-verified sigmoid router, expert-group selection, top-k weights and configured normalization/scaling; each selected expert applies its gate/up, configured activation, elementwise product, down projection and routing weight, while the shared expert branch is added separately. The selected reference's `num_nextn_predict_layers=1` is not an extra executed token-generation event or an inference result; include an auxiliary prediction block only if the reviewed causal-LM graph declares it as part of the selected static model scope. Never infer a training or generation step from that metadata alone.

## Kimi Linear coverage

Require `model_type="kimi_linear"`, `architectures` containing `KimiLinearForCausalLM`, and a compatible configuration and inventory. Expand `linear_attn_config.full_attn_layers` and `kda_layers` as one-indexed positions, as the pinned configuration specifies; require the two lists to be disjoint, in range, and together to cover all 27 layers. The reference full-attention positions are `[4, 8, 12, 16, 20, 24, 27]`; preserve those seven positions and the complementary twenty KDA positions in order. The bundled reference implementation requires `q_lora_rank=null` and `mla_use_nope=true`; fail closed if the checked reference config or inventory contradicts them. Also honor its configured dense/sparse MLP pattern, including the initial dense layer and later MoE layers; verify the 256 routed experts, top eight, one shared expert and full inventory. Do not assume these counts or a periodic layer pattern for another variant.

For full-attention layers, describe the verified direct query projection, latent key/value projections, configured query/key dimension split, causal attention and output projection. The pinned implementation concatenates its configured query/key slices directly; do not add an unverified rotary operation merely because a field contains `rope` in its name. For KDA layers, preserve the separate query/key/value projections, grouped causal short convolution (reference kernel size 4) and SiLU activation, query/key normalization, the forget-gate projections plus `dt_bias`/`A_log` decay, sigmoid input gate, symbolic delta-rule recurrent update, gated normalization, and output projection. Declare prior/next recurrent state symbolically and do not run it. The mixed-attention decoder retains both residual additions and its two normalizations. Its MoE router's score function, group selection, top-k normalization/scaling, routed expert branch, and shared expert branch follow the checked config and reviewed implementation; routing is not performed by static analysis.

## PEFT LoRA composition semantics

The model-admission contract owns adapter matching, identity, integrity, physical weights and numeric availability. When a validated LoRA composition is selected, extend each targeted base linear operation with its verified low-rank branch: `A(x)`, then `B`, scale by `lora_alpha/r`, and add the result to the ordinary base linear result in evaluation mode. The graph binds `A` and `B` to the real adapter tensors exposed in that composition's inventory, preserving their shapes and provenance. Do not create a merged base weight, virtual delta tensor, executable adapter reference, or inference timeline. With no adapter selected, the existing base graph remains unchanged.

## Parameter binding and provenance

A mathematical parameter can bind to one complete native tensor, a documented tied alias, a region of a fused tensor, or several packed/scaling tensors. Preserve both logical geometry and physical storage geometry without conflating them. Quantization scales are storage metadata, not extra neural layers; an integer cast is not dequantization.

The graph binds only to inventory identities that actually exist and to documented storage relationships. Admitted packed parameters with complete logical numeric descriptors expose those identities while retaining quantized bindings and physical provenance; decoder semantics belong to [models.md](models.md#admitted-packed-weight-decoding). Unresolved or unavailable numeric views retain their parameter descriptions and reasons, but never fabricate a usable tensor endpoint. The original checkpoint stays authoritative. Arbitrary region extraction, rank slicing, and raw-storage exploration are not required.

Record provenance categories for configuration declarations, observed storage metadata, and reviewed adapter rules. Keep source implementation revision and packaged description revision in compact provenance/evidence. Do not embed checkpoint code or local filesystem paths in public graphs. Resource references describe models/modules/parameters/tokenizers; frontend navigation and future runtime identifiers do not belong in the analysis.

## Validation and evidence

Use independent expected structures derived from reviewed source and checkpoint metadata, not expected graphs generated by the code under test. Tests must check graph closure, port directions, hierarchy/repetition identity, layer order and exceptions, residual and gated branches, symbolic shapes, parameter aliases/packing, and incomplete/unknown coverage.

Maintain small, distributable deterministic local fixtures for positive and negative cases: missing shards, invalid offsets, unsafe paths, contradictory configuration, unsupported variants, missing predictor, broken alias, swapped weight geometry, unresolved binding, unknown dimension, and malformed graph references. Intercept network, model-construction/forward/tracing, and full-weight materialization entry points so tests fail if analysis uses them. Check bounded fingerprint reads separately from tensor reads. Generation must work with read-only model directories and no GPU.

For actual reference acceptance, operators must supply the complete approved local variants. Record checkpoint repository/revision (or explain absent upstream revision), content fingerprint, selected file inventory and bytes, quantization metadata, reference-code revisions, and analyzer/description revisions. Keep weights and large generated graphs out of Git. Missing local models may be reported as SKIP by ordinary fixture CI, but do not count as actual-model validation; full reference acceptance remains pending until the required evidence exists. Source review, fixtures, and actual-checkpoint observations must remain distinguishable.

Integrated acceptance must validate DeepSeek-V2-Lite, GLM-4.7-Flash, and Kimi Linear alongside the existing Qwen references and preserve SmolLM2 workflows. Validate the SmolLM2 LoRA composition against both compatible base variants, and inspect NF4 and compressed-tensors logical weights through the same progressive path. Record cold/warm startup, hashing, graph generation/cache lookup, graph node/edge counts, layout time, and peak memory on an identified environment. Do not claim numerical equivalence, inference correctness, universal model support, or hardware throughput from static tests. Browser inspection of native and decoded 1D/2D weights must still verify exact source values.

## Maintenance and reference sources

Maintain bounded family/variant descriptions, selected storage mappings, graph validation, and their tests. New checkpoints within a checked variant may need only validation; changed operations or storage layout may need description changes. Unknown variants return partial/unavailable rather than executing arbitrary code. A third-party conversion's name alone is never proof of compatibility.

For the new families, the selected model/config revisions in [product scope](../product.md#architecture-reference-checkpoints-and-bounded-coverage) are acceptance discriminators. Review DeepSeek and Kimi's checkpoint-bundled Python files as inert source at those exact revisions; never import or execute them. Review GLM against the corresponding Transformers implementation below. A matching `auto_map`, architecture string, or repository name does not prove compatibility by itself.

| Reviewed source | Pinned source |
| --- | --- |
| SmolLM2 adapter config, weights and base metadata | [`hfm8tr/smollm2-135m-smoltalk-lora` at `fb39c7012c3b125f94d2bb1a093254994053f4ca`](https://huggingface.co/hfm8tr/smollm2-135m-smoltalk-lora/tree/fb39c7012c3b125f94d2bb1a093254994053f4ca) |
| DeepSeek-V2-Lite configuration and bundled reference implementation | [`slowfastai/DeepSeek-V2-Lite-bnb-4bit` at `9fc357346aeba67950a34a86f3520fc276ca2daf`](https://huggingface.co/slowfastai/DeepSeek-V2-Lite-bnb-4bit/tree/9fc357346aeba67950a34a86f3520fc276ca2daf) |
| GLM-4.7-Flash configuration and checkpoint | [`cyankiwi/GLM-4.7-Flash-AWQ-4bit` at `25624b53414e585bcf7dcb9584667c3106c6089b`](https://huggingface.co/cyankiwi/GLM-4.7-Flash-AWQ-4bit/tree/25624b53414e585bcf7dcb9584667c3106c6089b); reviewed model source at [Transformers `c8b81b63232be35ab1774dd3cabbf499d8b9808f`](https://github.com/huggingface/transformers/tree/c8b81b63232be35ab1774dd3cabbf499d8b9808f/src/transformers/models/glm4_moe_lite) |
| Kimi Linear configuration and bundled reference implementation | [`cyankiwi/Kimi-Linear-48B-A3B-Instruct-AWQ-4bit` at `5d029d1844aa64ec302e14466be7d0353c6e697f`](https://huggingface.co/cyankiwi/Kimi-Linear-48B-A3B-Instruct-AWQ-4bit/tree/5d029d1844aa64ec302e14466be7d0353c6e697f) |
| bitsandbytes NF4 quant-state serialization and dequantization semantics | [`bitsandbytes` `functional.py` at `833649043474794b8fe7a4136e0c40faf077b2e0`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py) |
| compressed-tensors packed int weights and bit order | [`compressed-tensors` `base.py` and `helpers.py` at `4a696625b7ada2cb857c75f67ce02dc56b381323`](https://github.com/vllm-project/compressed-tensors/tree/4a696625b7ada2cb857c75f67ce02dc56b381323/src/compressed_tensors/compressors/pack_quantized) |

The relevant HF Transformers sources for `llama`, `qwen3`, `qwen3_5`, `vjepa2`, `deepseek_v2`, and `glm4_moe_lite` are the `configuration_*` and `modeling_*` files at the reviewed Transformers revision above where applicable; use their pinned file paths, not a moving `main` branch. These sources and selected checkpoint files are evidence for bounded descriptions, not a guarantee of installed-library support. Implementation evidence must retain exact reviewed revisions and applicable licenses when reusing code.
