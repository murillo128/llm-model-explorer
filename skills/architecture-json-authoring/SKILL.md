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

When the same semantic defect can recur across producers, fix or validate the shared construction path rather than patching only one screenshot or one checkpoint. Add regression coverage at the producer boundary and at least one rendered/consumer path when the defect is visible only after projection into the UI.

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

Only after the concrete graph is correct, add repetition windows and eligible Shared families. Presentation metadata must annotate the real graph, never compensate for a missing graph.

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

**Wrong-direction or boundary-skipping edges:** ordinary dependencies are `output → input`; use only contract-defined group forwarding, route through group boundaries, and fan out from the original producer instead of from an input port.

**Duplicate or dual-role dependencies:** emit each exact source/target port tuple once and assign the dependency one real role; do not duplicate it or model it as both `data` and `state`.

**Implicit broadcasting through port shapes:** connected known dimensions must be compatible. Represent broadcasting in the operation or with an explicit node instead of changing the receiver shape to make the edge appear valid.

**State exposed as passive I/O:** recurrent K/V caches, optimizer moments, counters, and previous/next state use `state`, not ordinary `input`/`output` interfaces.

**Source-only validation:** validate the authored source and the materialized package through the real import path when available; passing JSON Schema or a local validator alone does not prove compatibility.

## Final principle

Treat every Architecture Explorer producer as an authored mathematical interface to the model, not as serialization of the implementation and not as a hand-drawn diagram. A high-quality graph lets a reader move from model-level structure to local equations and real weights without encountering hidden computation, unexplained names, invented topology, blank operations, or producer-specific quality drift.
