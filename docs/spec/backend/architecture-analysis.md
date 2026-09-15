# Static architecture analysis

## Responsibility and scope

This document owns the accepted Architecture Explorer analysis capability. [Product scope](../product.md#architecture-reference-checkpoints-and-bounded-coverage) owns the reference checkpoints and exclusions; [models.md](models.md) owns checkpoint admission and representations; [the API document](../api/architecture-explorer.md) owns graph records; [startup](architecture.md#blocking-architecture-preparation) and [cache](artifact-cache.md#structured-architecture-artifacts) own lifecycle.

Construct a semantic graph from guarded local configuration, indexes, Safetensors headers, and packaged, versioned architecture descriptions. Descriptions are verified against identified revisions of original model implementations during development. They encode mathematical relationships, not pixel layouts, hand-drawn per-checkpoint diagrams, or a second implementation of inference.

Production analysis must not call `forward()` or `generate()`, trace a real or dummy execution, require a prompt/video, load all weights into RAM/GPU, run preprocessors, execute Python in a checkpoint directory, or access the network. It must not instantiate an entire numerical model merely to enumerate its modules. Reading file bytes for the existing content fingerprint is permitted and bounded.

TorchLens, TransformerBridge, and an upgrade of Transformers are not prerequisites for this static increment. Existing compatible code/knowledge may be reused with provenance and license attribution. Inference/instrumentation dependencies can be selected by a later design; do not impose their numerical loading path on this analysis.

## Description selection and coverage

Select descriptions by supported configuration discriminators, architecture classes, relevant options, per-layer types, and checked storage bindings, not a marketing name or one matching tensor suffix. Normalize known configuration defaults only from the reviewed reference implementation. Unrecognized fields that can change structure, contradictory metadata, or missing required parameters must prevent a complete-coverage claim. Unrelated training metadata need not invalidate an otherwise recognized architecture.

A result is complete only within its declared scope and mathematical abstraction. Partial results identify unknown regions and their reasons while preserving verified structure. Do not guess edges from module registration order, silently substitute a Llama template, or treat an opaque required block as full support. Unavailable means no valid useful graph could be constructed; never return an empty graph labelled successful.

A valid partial graph is a complete serialized artifact with explicitly incomplete semantic coverage. Interrupted generation is not a partial graph. These distinctions drive the API states and cache publication rules.

## Mathematical abstraction

Represent linear and patch projections, normalizations, positional encoding, attention products, softmax, activations, multiplication between branches, residual addition, selection, and meaningful reshape/split/concat/transpose operations. Preserve biases and gates where relevant. Treat recognizable normalization/activation/state-update algorithms as operations with explanatory attributes/formulas rather than expanding every arithmetic instruction, cast, or allocation.

Edges mean verified data dependencies. Constants are checked against configuration and storage. Batch, sequence, spatial, temporal, context, and target dimensions may be symbolic; unresolved dimensions stay explicitly unknown. No displayed dimension or state tensor implies that an input was executed. Recurrent dependencies use symbolic prior/next-state ports, not an invented captured KV cache or an unrolled inference timeline.

Groups describe model hierarchy and actual instances. Every repeated layer retains identity, sequence index, parameters, and any structural exceptions. The backend emits the full semantic instance graph; repetition metadata enables compact presentation without asking the browser to reconstruct unknown computation. A description of a representative layer never substitutes for bindings of all instances.

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

## Parameter binding and provenance

A mathematical parameter can bind to one complete native tensor, a documented tied alias, a region of a fused tensor, or several packed/scaling tensors. Preserve both logical geometry and physical storage geometry without conflating them. Quantization scales are storage metadata, not extra neural layers; an integer cast is not dequantization.

The graph binds only to inventory identities that actually exist and to documented storage relationships. Admitted packed parameters with complete logical numeric descriptors expose those identities while retaining quantized bindings and physical provenance; decoder semantics belong to [models.md](models.md#admitted-packed-weight-decoding). Unresolved or unavailable numeric views retain their parameter descriptions and reasons, but never fabricate a usable tensor endpoint. The original checkpoint stays authoritative. Arbitrary region extraction, rank slicing, and raw-storage exploration are not required.

Record provenance categories for configuration declarations, observed storage metadata, and reviewed adapter rules. Keep source implementation revision and packaged description revision in compact provenance/evidence. Do not embed checkpoint code or local filesystem paths in public graphs. Resource references describe models/modules/parameters/tokenizers; frontend navigation and future runtime identifiers do not belong in the analysis.

## Validation and evidence

Use independent expected structures derived from reviewed source and checkpoint metadata, not expected graphs generated by the code under test. Tests must check graph closure, port directions, hierarchy/repetition identity, layer order and exceptions, residual and gated branches, symbolic shapes, parameter aliases/packing, and incomplete/unknown coverage.

Maintain small, distributable deterministic local fixtures for positive and negative cases: missing shards, invalid offsets, unsafe paths, contradictory configuration, unsupported variants, missing predictor, broken alias, swapped weight geometry, unresolved binding, unknown dimension, and malformed graph references. Intercept network, model-construction/forward/tracing, and full-weight materialization entry points so tests fail if analysis uses them. Check bounded fingerprint reads separately from tensor reads. Generation must work with read-only model directories and no GPU.

For actual reference acceptance, operators must supply the complete approved local variants. Record checkpoint repository/revision (or explain absent upstream revision), content fingerprint, selected file inventory and bytes, quantization metadata, reference-code revisions, and analyzer/description revisions. Keep weights and large generated graphs out of Git. Missing local models may be reported as SKIP by ordinary fixture CI, but do not count as actual-model validation; full reference acceptance remains pending until the required evidence exists. Source review, fixtures, and actual-checkpoint observations must remain distinguishable.

Integrated acceptance must validate all three new references and preserve existing SmolLM2 workflows. Record cold/warm startup, hashing, graph generation/cache lookup, graph node/edge counts, layout time, and peak memory on an identified environment. Do not claim numerical equivalence, inference correctness, universal model support, or hardware throughput from static tests. Browser inspection of native 1D/2D weights must still verify exact source values.

## Maintenance and reference sources

Maintain bounded family/variant descriptions, selected storage mappings, graph validation, and their tests. New checkpoints within a checked variant may need only validation; changed operations or storage layout may need description changes. Unknown variants return partial/unavailable rather than executing arbitrary code. A third-party conversion's name alone is never proof of compatibility.

Reference implementations are the relevant `configuration_*` and `modeling_*` files in [Hugging Face Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models), specifically `llama`, `qwen3`, `qwen3_5`, and `vjepa2`. Model/configuration sources are the [selected Qwen3 variant](https://huggingface.co/JunHowie/Qwen3-0.6B-GPTQ-Int4), [selected Qwen3.5 variant](https://huggingface.co/AxionML/Qwen3.5-0.8B-NVFP4), and [selected V-JEPA 2 variant](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256). These are research entry points, not pinned acceptance evidence or a guarantee of installed-library support. Implementation evidence must identify exact reviewed revisions and retain applicable licenses when reusing code.
