---
name: architecture-json-authoring
description: Design, create, and review Architecture Explorer semantic graphs from model-owned architecture.json files or built-in/static graph producers so views are mathematically faithful, visually legible, parameter-inspectable, and consistent across producer paths.
---

# Author Architecture Explorer Semantic Views

## Responsibility

Use this skill when creating or materially revising **any producer of an Architecture Explorer semantic graph**: a model-owned `architecture.json`, a built-in/static architecture analyzer, a packaged reviewed architecture description, or another repository-owned producer that emits the same graph contract. Also use it when reviewing an existing graph whose records are technically valid but semantically or visually poor.

The goal is not merely to satisfy a schema, graph validator, or backend type. Every producer must yield a faithful, inspectable **semantic view of the computation**: meaningful hierarchy, explicit important operations, truthful data dependencies, exact checkpoint parameter bindings, concise formulas or operation signatures, visible computational constants, and names that let a reader follow values from ports through equations without translation.

This skill owns authoring procedure and presentation-quality rules **independently of where the graph comes from**. A built-in producer does not get a weaker quality bar than a model-owned `architecture.json`. The normative exchange contract remains:

- `docs/spec/backend/model-owned-architecture.md`
- `docs/spec/backend/architecture-definition.schema.json`
- `docs/spec/backend/architecture-analysis.md`
- `docs/spec/ui/architecture-explorer.md`

Examples under `examples/model-owned-architecture/` illustrate one producer path but do not override those specifications or this authoring discipline. Built-in/static producers must follow the same semantic and presentation rules even when they construct records programmatically rather than serializing JSON.

## Load the minimum required context

Before authoring:

1. read the current Architecture Explorer graph/API and presentation specifications;
2. for a model-owned file, also read the model-owned architecture specification and JSON Schema; for a built-in/static producer, read the owning architecture-analysis specification and the producer's current construction helpers;
3. inspect the model's actual implementation or authoritative architecture description;
4. inspect the exact checkpoint tensor inventory and shapes that the producer will bind;
5. inspect the existing producer output only as evidence of current intent, not as a structure that must be preserved.

Do not infer an architecture from tensor-name prefixes alone. Do not copy the Python/PyTorch module tree mechanically. Do not preserve obsolete visual structure merely because an earlier JSON or built-in generator used it.

## Author the semantic graph first

Design the computation before writing JSON. Identify the model's meaningful stages, the values crossing those stages, the trainable parameters used by each operation, and the real branching, residual, state, recurrent, or conditioning paths.

Choose hierarchy from **semantic computational units**, not from source-code nesting. A group should exist because its children form a useful architectural concept that a reader may collapse, expand, isolate, or compare. Avoid wrapper groups that add no information.

Examples of useful groups include an encoder block, attention block, MLP, temporal predictor, gating block, state-update block, projection head, or objective-specific head when those are real units of the model. These are examples, not a fixed vocabulary.

Do not put the whole model under a redundant group whose only purpose is to repeat the model name. Do not create a group solely to make the diagram look boxed. Do not introduce a group around an already meaningful component unless the additional boundary has a real interface or architectural purpose.

## Apply the same quality bar to every producer

Do not treat repository-owned analyzers as an implementation shortcut around authoring rules. If a built-in producer constructs nodes through helpers such as `node(...)`, `linear(...)`, `group(...)`, templates, or semantic-role maps, review the **materialized graph** exactly as you would review an authored JSON file.

Prefer reusable producer helpers for invariants that should hold across families: canonical operation formulas/signatures, port naming, computational scalar attributes, group interfaces, parameter locality, and graph validation. Model-specific code may specialize real semantics, but it must not silently weaken these invariants.

When the same semantic defect can recur across producers, fix or validate the shared construction path rather than patching only one screenshot or one checkpoint. Add regression coverage at the producer boundary and at least one rendered/consumer path when the defect is visible only after projection into the UI. Follow `skills/test-quality/SKILL.md`: reuse existing proof and extend its owning cases before adding tests; this rule does not require duplicate suites, a checkpoint Cartesian product, or a new browser scenario when existing coverage already proves the distinct risk.

## Decompose compound computations instead of hiding them in one card

A compound component must be expanded into internal operations when a single operation card would hide useful structure.

Use the following test: if understanding the component requires seeing independent branches, distinct trainable projections, a meaningful normalization, a gate, a softmax, a state transition, a residual merge, a multi-stage transform, or another semantically important dependency, represent those as child operations rather than compressing all of them into one formula.

**Attention is the canonical example, not a special case.** An Attention group should normally contain the actual Q, K, and V projections and the meaningful computation that follows them, rather than a single `Attention` card containing the entire equation. Q/K/V are naturally parallel branches; score computation, scaling/masking when applicable, softmax, value aggregation, and output projection should remain explicit at the level needed to understand the real model.

Apply the same rule everywhere else. For example, an MLP with gate/up branches and a multiply should expose that branch structure; a predictor with conditioning and state fusion should expose those operations; a gated state update should expose the transforms and merge that define the update. The criterion is semantic value, not the name of the component.

An adapted projection is also a compound component. When a targeted linear operation has a real additive adapter branch such as LoRA, keep that projection as the semantic unit: the collapsed component should expose the same external input/output contract, while expansion reveals the base linear path and the verified adapter path (for standard LoRA, `A -> B -> scale -> add`). Do not dump adapter operations as unrelated siblings of the surrounding Attention/MLP group merely because they are easy to emit there. Do not create one global adapter group spanning unrelated projections; group each adapter with the operation whose computation it modifies.

Do not over-decompose implementation noise. Pure casts, allocations, framework bookkeeping, harmless view/reshape steps, or arithmetic that contributes no architectural understanding should stay implicit unless they change tensor meaning or are needed to explain a dependency.

A group card is a boundary and navigation unit. It must not become a substitute for the computations that belong inside it.

## Keep ports and formulas in one mathematical vocabulary

Visible input/output names and formula variables must agree exactly.

If a card formula uses `x`, its visible input port must be named `x`. If the formula produces `q`, the output port should be `q`. Do not show an input called `hidden_states` while the equation refers to `x`, or label a connection `features` while the receiving formula calls that value `z`.

The stable internal JSON `id` may remain technical, but the visible port `label` must use the same mathematical symbol or canonical variable name used by the operation signature/formula.

Every meaningful symbol in a formula should have an obvious source:

- an input port;
- an output port;
- a referenced parameter;
- a declared scalar computational attribute;
- or a standard mathematical operator/function.

Avoid unexplained aliases. Avoid changing variable names as a value crosses a group boundary unless the model semantics actually change. Group boundary ports should preserve the mathematical identity of the value they forward.

Prefer concise formulas that describe the card's own computation. Do not paste an entire block's equations into a parent group header.

A visible leaf operation that performs meaningful computation should normally carry a concise formula or operation signature. Common arithmetic/transform cards such as linear, add, multiply, scale, matmul, softmax, reshape and transpose should not render as semantically blank boxes when their operation is known. A signature such as `out = reshape(x, ...)` is sufficient when a fuller equation would add noise.

Treat the rendered card as a closed mathematical vocabulary. Every non-operator symbol used by the formula/signature must be represented by a visible port, referenced parameter, or computational attribute that the card actually exposes. Conversely, a displayed computational scalar should use the same name the formula uses when it participates in that formula. Do not write `alpha / r` in a formula while the card only exposes a derived `factor`; either expose `alpha` and `r` truthfully, or write the formula in terms of the visible `factor`. The output symbol must match the visible output port (for example `out`, not an unexplained `y`).

## Put parameters on the operation that uses them

Every inspectable checkpoint tensor must be bound through the exact tensor name and truthful logical shape required by the model-owned contract.

Attach a parameter reference to the operation that actually consumes it. Do not aggregate descendant weights on a group card. Architecture Explorer intentionally renders a card's own referenced tensors and does not use a group as a bag of all child parameters.

Keep matrices and vectors semantically local. A query projection should reference its own query weight/bias; an output projection should reference its own output weight/bias; an MLP gate projection should not be shown on the parent MLP card merely to reduce the number of rows.

Reuse a parameter ID only for genuine shared/tied use of the same logical parameter. Structural similarity between repeated components does **not** imply weight sharing.

Parameter names must match the admitted checkpoint inventory exactly. Never invent a prettier tensor path in the binding. Human-friendly semantics belong in the node label/formula; the parameter binding remains exact.

## Make computational constants visible on the card

Scalar values that materially participate in a computation should be represented as **node attributes** so Architecture Explorer can render them beneath the parameter rows alongside the operation's matrices/vectors.

Examples include an actual attention scale, epsilon, temperature, head count, head dimension, kernel size, dropout probability when it is genuinely part of the described computation, or another scalar/hyperparameter needed to understand the card.

Use the original meaningful name and actual value. Do not hide such values only inside `description` or bury them in a large formula when the card can expose them directly.

Do not turn `attributes` into a configuration dump. Include computational values relevant to that operation. Provenance, semantic roles, diagnostics, historical notes, file paths, and unrelated configuration are not computational constants.

Do not invent defaults. Missing or unknown values remain missing or unknown.

Literal mathematical constants such as the fixed exponent in a standard formula need not become attributes unless exposing them improves understanding; model/configuration values that vary between architectures normally should.

## Name for the architecture that exists now

Write labels as if this were the first published architecture of the current model.

Use concise canonical English names. Do not encode design history in cards or groups. Avoid labels such as `v3`, `v4`, `unchanged`, `legacy`, `new`, `organized`, `raw residual output`, or comments about a previous architecture unless that version distinction is itself part of the model's runtime semantics.

Put experiment/version identity in `architecture_revision` and checkpoint metadata, not repeatedly in visual component labels.

Prefer `Attention`, `MLP`, `Temporal Predictor`, `Stop Head`, `Player Projection`, or similarly canonical names when those terms faithfully describe the model. Do not force these names onto a model whose concepts differ.

Descriptions may explain non-obvious semantics, but labels should remain compact.

## Preserve real topology

Edges must describe verified data dependencies, not a visually convenient sequence.

Ordinary edges must connect `output → input`. The only exceptions are the group-forwarding cases explicitly defined by the model-owned contract. Every dependency must respect group boundaries and use the declared boundary forwarding mechanism; no edge may skip across a group boundary.

An input port that receives a value is a consumer, not a new producer. If the same value feeds multiple consumers, fan out from the original output rather than forwarding from an input port.

Each exact `source node/source port/target node/target port` tuple must appear only once. A dependency must also have one semantic role: do not represent the same dependency simultaneously as both `data` and `state`.

Keep residual bypasses, parallel branches, masks, conditioning signals, context, prior/next state, and other semantically distinct flows separate. Never merge signals because they have the same shape or a similar label.

When a group is expanded, its internal routes must explain how its boundary inputs become boundary outputs. Do not create a parent formula that contradicts or duplicates the child graph.

Use `data`, `state`, and `context` edge kinds according to their real role. Do not represent a recurrent or state dependency as ordinary serial data merely to simplify layout.

Reserve `input` and `output` for passive data interfaces. Recurrent K/V caches, optimizer moments, counters, and previous/next state belong on `state` interfaces rather than being modeled as ordinary inputs or outputs.

Shapes should be as truthful as the contract permits: constants where known, declared symbols where symbolic, display-only expressions where appropriate, and explicit unknowns when unresolved. Known dimensions at the two ends of an edge must be compatible. Broadcasting must be declared by the operation itself or represented with an explicit broadcast/expand node; never simulate broadcasting by changing the receiving port to an incompatible shape. Never fabricate a dimension for presentation.

## Repetitions describe real repeated instances

Use `repetitions` to group **existing explicit instances** into a repetition window. Repetition metadata is presentation/navigation metadata, not a loop interpreter and not permission to omit concrete layers.

Preserve real source order, instance identity, indices, and structural variants. A repeated stack with exceptional layers must declare those exceptions truthfully.

Do not use a repetition to imply tied weights or structural equivalence beyond what is actually declared.

## Shared structures require explicit correspondence

Use `templates` only when repeated Attention/MLP-like components or other supported component families satisfy the model-owned Shared contract and their correspondence can be declared exhaustively.

A Shared family represents verified structural correspondence across concrete source groups. It does not replace those groups and does not mean their weights are shared.

Map nodes, ports, internal edges, and referenced parameters by stable semantic roles for every declared instance. Omit the family when exact correspondence cannot be asserted.

Repetitions and Shared structures are independent: repetitions organize ordered repeated instances; Shared templates enable structure comparison/navigation.

## Compose indexed stack views from repetitions and Shared structures

For a compact serial stack, reuse a verified Shared structure as the symbolic `Layer[i]` inside the existing repetition container. Do not introduce another abstract computational node kind or replace the concrete source layers with a self-loop. Shared supplies structural correspondence; the repetition supplies ordered instances; the real inter-instance edges establish whether those instances form a serial chain. Neither a family label nor repetition membership alone proves that chain.

Author complete whole-layer correspondences, including nested Attention/MLP groups, operations, ports, internal edges and parameter roles, when those layers are structurally equivalent. Preserve eligible nested Shared families as well. Extend the owning graph contract, producer and both consumers together if the current supported component roles do not yet admit whole layers; do not label a layer as Attention/MLP, invent unsupported JSON fields, or create a competing template system to bypass that limitation.

The source graph retains its exact initial input, every `layer[k].out -> layer[k+1].x` dependency, and its final output, with contract-defined boundary forwarding wherever a real group boundary exists. A presentation-only repetition container does not justify adding a redundant source wrapper. Keep all concrete identities, mathematical operations, shapes, parameter bindings and edge kinds recoverable, including when an already-supported compact wire form is decoded.

The indexed view derives three relationships from that source graph. Its entry displays `x[0] -> x[i]` and applies only to the first instance. Its return displays `out[i] -> x[i]` but means `x[i+1] = out[i]` for `0 <= i < N-1`, not feedback into the same concrete layer. Its exit displays `out[i] -> out[N-1]` and applies only to the last instance. For `Decoder ×30`, the final label is `out[29]`: thirty layer applications have twenty-nine inter-layer transitions. Derive actual first/last indices from the repetition rather than hard-coding thirty or renumbering source instances.

These are **presentation relationships with source provenance**, not new `ArchitectureEdge.kind` values. Retain exact source edge IDs and ordered forwarding paths for entry/exit, and the separate ordered inter-instance paths represented by the return. Keep endpoint and instance correspondence available for inspection; do not concatenate unrelated paths into a fictitious source path or treat boundary entry plus return as a simultaneous arithmetic merge. Preserve the ordinary `data`, `state` and `context` meanings.

The compact group keeps its existing title, multiplicity, appearance and controls. Its first expansion shows one symbolic Shared layer, which can itself expand into the common operations; explicit concrete-instance/window and exhaustive views remain available. Route the return outside the inner layer card but inside the repetition container, with space for labels and headers. Connect each invariant input, such as `cos`, `sin` or `mask`, separately through its corresponding group and layer ports. Only an identical verified source value may be shared across instances; never merge these distinct signals into a bus or create a shared-input computation box.

Use indexed display aliases consistently in the symbolic layer's ports and local formulas/signatures, including its expanded operations: `x[i]`, `out[i]`, and layer-dependent parameter roles such as `weight[i]`. A concise group signature may read `out[i] = layer[i](x[i], cos, sin, mask)` when it matches the real interface; it does not replace the inner computation. Invariant inputs/constants remain unindexed. The layer index is not a tensor dimension or a rewrite of checkpoint names, source IDs or binding paths. Preserve genuine tied parameters and any existing mathematical indices; do not apply blind string replacement to formulas.
Parameters remain on their consuming operations, not collected into a group-level weights bag.

A symbolic parameter role is not an actionable tensor binding. Numeric inspection requires an explicitly selected concrete instance and resolves its exact existing parameter/tensor identity. Never silently use layer zero, expose anchor-instance weights in structure-only mode, or materialize tensor values to establish correspondence.

Only offer a single homogeneous-layer return when complete correspondence and actual ordered connections prove it without hiding extra dependencies. A mixed stack must preserve its variants through compact compatible contiguous ranges or a verified repeated pattern, not falsely equate all layers or reorder nonconsecutive members. Keep per-instance side inputs and outputs, including `state[i]`, distinct from the serial activation carry. A singleton has no return. Unknown or malformed external graphs retain the established truthful fallback or validation behavior, but a blanket fallback for a supported family's ordinary hybrid/stateful stack is not completion of an all-family migration.

This pattern abbreviates finite structural depth, not autoregressive token generation, prefill, sampling, a runtime execution engine or temporal KV-cache recurrence. Those require their own accepted semantic contract and must not be inferred from the visual return.

When adopting this pattern, migrate eligible existing built-in/static producers and repository-owned architecture definitions through their shared construction/annotation paths, not just a screenshot fixture. Preserve exceptional variants and record genuine eligibility limits. Update affected producer/definition revisions through normal startup cache invalidation; never hand-edit disposable caches or silently rewrite user-owned model files. Validate source conservation, correspondence and projection at their owning boundaries, reusing existing cases and one representative real-rendering path rather than duplicating the same matrix across backend, API and browser tests.

## Describe no-cache autoregressive generation outside the model

When a task adopts a generation view, distinguish the neural model's single forward pass from its external generation procedure. Keep one existing `model` boundary and its real internals inside a meaningful `Generation` container. Add three ordinary operation cards outside `model`: `Prepare inputs` before it, then `Next token` and `Append token` after its vocabulary logits. These operations and their dependencies must be declared by the producer; the frontend must not invent them from a label named `logits`, a tokenizer capability, or a `language_model` scope alone. A declared explanatory greedy policy is not a claim about the checkpoint's runtime sampling default.

Without KV caching, `token_ids[t]` denotes the entire sequence at generation step `t`, not the token at position `t`. Initialize it from `prompt_ids`. `Prepare inputs` forwards that sequence and derives the positions, masks and invocation inputs actually required by the reviewed family. Do not require a fictitious position input on a model without one, collapse two distinct masks into one, or impose SmolLM2's exact port list and ranks on every model. Forward the full sequence on every iteration; never feed only the last generated ID in a no-cache diagram. Use truthful symbolic lengths, rank and mask conventions. A simple `arange` and triangular mask describe the bounded unpadded case, not arbitrary padded, packed, multimodal or encoder-decoder inputs.

`Next token` consumes vocabulary logits, selects the last valid sequence position, and outputs one token ID. For the initial greedy view its signature can be `next_token_id[t] = argmax_vocab(last_valid(logits[t]))`. The selected vocabulary index is the ID; neither detokenization nor re-tokenization belongs on this return path. Preserve the selected dimension and output shape. Do not reinterpret candidate/option scores as vocabulary logits or introduce sampling, temperature and top-k controls into a greedy-only increment.

`Append token` has two independent inputs: the full `token_ids[t]` and `next_token_id[t]`. Derive the former from the same `Prepare inputs` output that feeds the model; this visible bypass is essential. Its formula is `token_ids[t+1] = concat(token_ids[t], next_token_id[t])`, along the sequence axis, with compatible ranks (for example `[B, S]` plus `[B, 1]` yields `[B, S+1]`). Do not obtain both inputs from `Next token`, route IDs through logits, or drop the original prefix.

Describe the sequence carried between steps explicitly, using the owning contract's current/next `state` interfaces and bounded generation-phase correspondence. This sequence state is not a KV cache. Entry initializes the current sequence; return advances the completed sequence to the next step; exit exposes the terminal sequence. They are different phases, not a simultaneous arithmetic merge or three unconditional same-step data edges. Keep source endpoints, actual edge IDs, state ownership and phase semantics inspectable. Reuse existing presentation relationship/routing machinery, but do not invent a Shared family, thirty copies of the model or a list of runtime iterations to satisfy structural repetition metadata.

An ordinary edge must not falsely equate current length `S` with next length `S+1`. Keep the change at the declared state transition and keep connected ordinary port shapes compatible; do not overwrite shapes, add an implicit broadcast or loosen general validation for this drawing. Define any newly needed generation semantics in the owning API/backend/UI specifications and validators together, preferring existing group, operation, state and attribute records over a new loop language. Formulas and symbolic step/length expressions remain data-only descriptions, never code to execute.

The visual return runs from `Append token` back to `Prepare inputs`, outside `model` and inside `Generation`, with both ends attached to their real projected ports. The initial `prompt_ids` route enters that same projected input; the completed sequence also reaches `Generation`'s output as `token_ids[T]`, explicitly including the prompt. Preserve each route's distinct initial, next-step or terminal meaning. Reserve return/bypass gutters without crossing model interiors, headers or labels, and retain the existing line/port/label highlighting and inspection. Reuse ordinary card styles and boundary ports; a source-level sequence state need not become an extra visible card when its complete meaning is projected into these relationships.

Do **not** add a Stop condition card, decision diamond, Yes/No branches, boolean control port, separate legend, loop-controller card or stopping subtitle on `Generation`. At most, show a compact terminal predicate as a second formula line inside `Append token`, such as `(next_token_id[t] in eos_token_ids) or (t+1 >= max_new_tokens)`. Only include terms whose symbols and policy values are actually declared; omit this optional line rather than inventing an EOS ID or token limit. This annotation is not an extra output or an executed decision node. Zero generated steps, when a declared policy permits them, return the prompt without claiming a model pass or token append.

Use `t` for generation and retain `i` for layer depth. The same model and exact weights are reused between generation steps: do not add `weight[t]`, duplicate parameter resources or require choosing a generation step to inspect a weight. Existing layer-indexed Shared views and genuine weight tying retain their own rules. Keep temporal aliases at generation interfaces/formulas where useful; do not mechanically rewrite every operation inside the model.

Apply the pattern only to a verified autoregressive vocabulary-generation path. Migrate eligible existing native/static producers and repository-owned definitions through shared helpers and normal producer/cache revisions; a one-off fixture is not delivery. Preserve ordinary views for genuinely ineligible models, with explicit reasons. Decision-oriented CLM/Kev graphs and V-JEPA encoder/predictor graphs do not acquire a text-generation loop merely because they contain language backbones or repeated layers. Preserve their real interfaces, independent states and output meanings. User-owned model files remain read-only.

Keep this a static, no-cache explanatory view. Do not add inference endpoints, a generation runner, polling, live token values, model downloads, numerical execution or cross-step KV-cache wiring. Reuse tests for source bindings, indexed returns, layout and lifecycle. Add only missing generation topology/phase/shape assertions at their owner and one representative producer-to-built-browser extension for distinct rendering risk; do not replay a model/quantization matrix or execute real generation to validate static records.

## Complete migrations across supported families

When the controlling issue requests all supported families, enumerate the shipped producer registry and its structurally distinct paths before choosing implementation or acceptance cases. Record the required result for each feature in the existing issue/PR evidence. A small mandatory reference model is a regression anchor, not the only required positive outcome. A row saying that a family was audited and left unchanged does not satisfy a positive migration requirement.

Distinguish semantic non-applicability from an implementation gap. A decision/visual model without an autoregressive vocabulary head must not receive a fabricated generation loop, but its real repeated encoder/decoder/predictor layers still receive applicable indexed-depth support. In contrast, multiple masks, a different position convention, layer-local state, MoE routing, mixed variants, or a generator that exposes cache arguments are adaptation work for an already-supported generative family, not automatic permission to exclude it.

Derive the family's no-cache full-sequence invocation from its pinned reviewed implementation. Declare empty prior KV state, convolution/recurrent initialization and position/mask preparation only where justified by that implementation; reset invocation-local state for every full-sequence generation step without erasing state updates inside the forward computation. No-cache across calls does not mean stateless computation inside a call. Keep this preparation inside the existing semantic owners, with expandable detail when useful, rather than adding caller obligations or a second top-level flowchart. A current cached-only graph may need a source-backed specialization or corrected interface; preserving an earlier representation is not more important than faithfully describing the requested invocation.

For mixed or stateful depth stacks, preserve a designated activation carry separately from invariant inputs and indexed per-instance side inputs/outputs. Retain K/V, convolution and recurrent state identities; never feed one layer's next cache to another layer as though it were the hidden-state carry. Compact verified homogeneous segments, keep genuine exceptions explicit, and use an exact repeated pattern only when its source order and instance mapping are proven. Interleaved selection/stacking helpers or a single-output-only visual implementation are not semantic reasons to reject an otherwise representable stack. Reuse and extend Shared correspondence rather than suppressing genuine differences, equating unknown shapes or copying one variant's internals into another.

A missing reviewed fact is a concrete evidence blocker to resolve, not permission to invent an initializer or declare the migration complete. If the authorized family outcome still cannot be established, keep that requirement visibly unresolved under the existing issue workflow and obtain a scope decision before handoff. Do not silently redefine supported families, delete an acceptance row, waive a family because its metadata is expensive, or treat a permissive fallback as the requested positive result. Unknown external author-owned graphs retain safe fallback and validation; that safety mechanism does not waive the controlled native migration.

Prove breadth cheaply at the producer/consumer boundary: extend existing small parameterized family cases with independent expected feature presence, actual bindings and representative shape/state/variant assertions. A positive result means the produced graph is consumed into the intended view, not merely that a helper was called or a template role was attached. Reuse existing quantization, cache, provenance and lifecycle proof; do not multiply every family by every storage format and browser scenario. Real-browser evidence should target only the distinct rendering risks, including a non-dense or hybrid path when that is the newly affected behavior. Shared helper correctness alone is insufficient evidence that every required producer adopts it.

## Prefer semantic fidelity over visual compactness

The default view should be readable, but never obtain compactness by deleting meaningful computation or inventing opaque mega-cards.

A good architecture view has a clear left-to-right computational story, uses vertical separation for genuine parallelism, and reveals detail progressively through meaningful groups. It should be possible to collapse the model to major stages and expand a stage to understand its actual computation.

Do not optimize the JSON for a screenshot. Architecture Explorer controls layout; the JSON controls semantics.

## Authoring workflow

### 1. Inventory the model

Write down the real model inputs, outputs, major stages, repeated stacks, state/context dependencies, and exact checkpoint parameters. Resolve ambiguity from implementation/configuration before drawing the graph.

### 2. Draft groups and interfaces

Choose only semantic groups. Define their boundary values with canonical mathematical names. Confirm each group has a reason to exist when collapsed and useful detail when expanded.

### 3. Expand compound components

Within each group, create operations for meaningful projections, branches, normalizations, gates, attention products, activations, residuals, state updates, selections, concatenations, or other real stages. Use Attention/Q/K/V as a reference pattern for the principle, not a hard-coded template.

### 4. Normalize operation vocabulary

For every card, align port labels, formula/signature variables, parameter references, and scalar attribute names. A reader should not need to translate between UI labels and equations. Ensure meaningful common operation cards are not blank when their semantics are known, and prefer shared producer helpers for canonical signatures when several built-in analyzers emit the same operation.

### 5. Bind real parameters

Map each logical parameter to the exact checkpoint tensor name and shape. Keep each reference on the operation that consumes it. Verify tied/shared parameters deliberately.

### 6. Add repetitions and Shared metadata

Only after the concrete graph is correct, add repetition windows and eligible Shared families. Presentation metadata must annotate the real graph, never compensate for a missing graph. For an eligible indexed stack, include whole-layer correspondence and verify the exact initial, inter-instance and final paths plus invariant inputs. Keep symbolic display aliases separate from concrete parameter/resource identity.

For an adopted no-cache generation view, separately declare the external input preparation, vocabulary selection, append operation and sequence-state phases. Do not reuse structural repetition membership as evidence of temporal generation. Verify the full-sequence bypass and initial/return/final relationships before checking their compact presentation.

### 7. Validate contract and presentation

For a model-owned package, validate the materialized package with `llm-model-explorer-validate-architecture <model-directory> [--json]` from the installed Explorer backend. This is the canonical graph and binding validator for that producer path; a successful exit proves the package passes the same static import path used by cold startup. Use `--json` in model-repository CI. JSON Schema checks remain useful while editing, but cannot prove checkpoint bindings.

For a built-in/static producer, run its owning backend graph/binding tests plus the repository API/consumer validation that materializes the resulting graph. Add focused regression assertions for hierarchy, formulas/signatures, port names, visible computational attributes, and parameter bindings affected by the change. When the defect is visual, exercise the actual Architecture Explorer projection/rendering path rather than proving only backend serialization.

In either path, load a representative actual model in Architecture Explorer when practical and inspect both collapsed and expanded views. No static validator proves equivalence to `forward()`.

## Review checklist

Before accepting an Architecture Explorer graph producer, verify all of the following:

1. The hierarchy reflects semantic computational units rather than the implementation class tree.
2. No redundant wrapper group exists only for presentation.
3. Compound blocks are decomposed wherever a monolithic card would hide important internal structure.
4. Attention, when present, uses the same general decomposition rule as every other compound component; Q/K/V and other meaningful suboperations are not hidden merely for compactness.
5. Port labels and formula/signature variables use the same names, including the output symbol.
6. Formula/signature symbols have obvious sources in visible ports, parameters, displayed computational attributes, or standard math; no hidden alias is required to understand the card.
7. Meaningful common operation cards are not blank when a concise formula or signature can truthfully describe them.
8. Parameter bindings use exact checkpoint tensor names and truthful shapes.
9. Parameters appear on the operations that consume them, not aggregated on parent groups.
10. Computational scalar constants/hyperparameters that matter to understanding are exposed as operation attributes with real names and values, and formulas use the same visible names.
11. Attributes are not a configuration/provenance dump.
12. Labels are concise, canonical, English, and free of design-history wording.
13. Real residual, branch, mask, context, state, and conditioning paths are preserved.
14. Compound adapted operations are grouped with the base operation they modify rather than flattened into unrelated siblings of a larger parent component.
15. Ordinary edges are `output → input`; only contract-defined group forwarding is used, group boundaries are never skipped, and fan-out originates from the real producer output.
16. Every exact source-port/target-port connection appears once, and no dependency is represented simultaneously as both `data` and `state`.
17. Known dimensions at connected endpoints are compatible, and broadcasting is explicit in an operation or node rather than encoded as a false receiver shape.
18. Passive data interfaces use `input`/`output`; recurrent caches, optimizer moments/counters, and previous/next state use `state`.
19. Repetitions annotate explicit concrete instances and preserve order/variants.
20. Shared templates declare exact correspondence and do not imply tied weights.
21. Unknown information remains explicitly unknown rather than guessed.
22. The applicable source and materialized graph pass the real producer/import/consumer validation path; schema or backend type validity alone is not treated as proof of full compatibility.
23. The graph is understandable when viewed collapsed and expanded in Architecture Explorer.
24. Matrix/vector buttons resolve to the intended real tensors for representative operations.
25. The expanded graph does not rely on a giant parent formula to explain computation that should exist as child nodes.
26. Indexed stack views compose verified Shared layers with real repetition wiring; entry, return and exit retain exact source provenance without adding a source self-loop.
27. Indexed formulas/ports/parameter roles agree, invariant inputs stay distinct, and numeric inspection requires a concrete instance.
28. Existing eligible producers adopt the pattern through normal generation/cache revisions; fallback preserves exceptions and tests reuse the minimum independent owning proof.
29. No-cache generation declares preparation, vocabulary selection and two-input append outside the unchanged model; the full sequence, phase-aware return and truthful evolving shapes are preserved without duplicating weights.
30. Generation adds no separate stopping element or boolean control port; an optional supported terminal predicate appears only inside Append token, and ineligible decision/visual models remain unchanged.
31. An all-family migration has an observed required outcome for every applicable supported producer; interface, state and variant adaptation gaps are not reported as successful exclusions.

## Common failure modes

**Valid JSON, bad architecture:** passing the schema proves the exchange format, not the quality of the semantic view. Review the actual canvas.

**One giant operation per subsystem:** split meaningful computation into child operations and retain the subsystem as a group.

**Framework-tree transcription:** redesign around mathematical units and real interfaces instead of preserving every Python module boundary.

**Formula/port/attribute drift:** rename the visible ports, formula/signature, or displayed computational attributes so the card uses one closed vocabulary. A formula must not depend on hidden aliases or hidden constants.

**Blank semantic cards:** if a meaningful visible operation is known but its card has no formula/signature or other local explanation, add a concise source-backed formula/signature rather than relying on the reader to infer behavior from the title alone.

**Flattened adapter internals:** keep a real adapter branch inside the projection/operation it modifies. Use that compound component's ordinary boundary ports and progressive disclosure rather than placing A/B/scale/add as unrelated siblings of the surrounding Attention/MLP group.

**Constants hidden in prose:** move real computational scalar values into operation attributes when the UI should show them on the card.

**Parameters on group cards:** attach them to the child operations that consume them.

**Historical labels:** remove version/change-history wording from visual names and use `architecture_revision` for revision identity.

**Fake compactness:** use groups/repetitions for progressive detail instead of deleting real branches or dependencies.

**Shared confused with tied weights:** structural correspondence and parameter identity are separate claims; represent each truth independently.

**Symbolic return mistaken for source recurrence:** keep concrete inter-layer edges and derive the indexed return from their correspondence. Shared membership alone does not prove serial flow, and the return is not a temporal generation loop.

**Indexed labels mistaken for tensor bindings:** display `weight[i]` as a role, preserve exact checkpoint names, and require an explicit instance before numeric inspection. Do not silently reuse the anchor layer's resources.

**Logits wired directly back to token IDs:** insert explicit last-position vocabulary selection and a two-input append operation. Return the full enlarged sequence without caching; positions and masks alone do not supply newly generated token identities.

**Generation drawn as a decision flowchart:** reuse ordinary cards and initial/return/final relationships. Do not add a separate stopping element; any supported terminal predicate stays inside Append token.

**One-family delivery called a migration:** do not turn the first reference model's exact ports, shapes or homogeneous topology into a universal eligibility gate. Adapt the remaining supported producers and verify their actual consumer outcome before declaring completion.

**Wrong-direction or boundary-skipping edges:** ordinary dependencies are `output → input`; use only contract-defined group forwarding, route through group boundaries, and fan out from the original producer instead of from an input port.

**Duplicate or dual-role dependencies:** emit each exact source/target port tuple once and assign the dependency one real role; do not duplicate it or model it as both `data` and `state`.

**Implicit broadcasting through port shapes:** connected known dimensions must be compatible. Represent broadcasting in the operation or with an explicit node instead of changing the receiver shape to make the edge appear valid.

**State exposed as passive I/O:** recurrent K/V caches, optimizer moments, counters, and previous/next state use `state`, not ordinary `input`/`output` interfaces.

**Source-only validation:** validate the authored source and the materialized package through the real import path when available; passing JSON Schema or a local validator alone does not prove compatibility.

## Final principle

Treat every Architecture Explorer producer as an authored mathematical interface to the model, not as serialization of the implementation and not as a hand-drawn diagram. A high-quality graph lets a reader move from model-level structure to local equations and real weights without encountering hidden computation, unexplained names, invented topology, blank operations, or producer-specific quality drift.
