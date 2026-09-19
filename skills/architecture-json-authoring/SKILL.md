---
name: architecture-json-authoring
description: Design, create, and review model-owned architecture.json files so Architecture Explorer views are mathematically faithful, visually legible, parameter-inspectable, and consistent with the model-owned architecture contract.
---

# Author Model-Owned Architecture Views

## Responsibility

Use this skill when creating or materially revising a model-owned `architecture.json` for Architecture Explorer, or when reviewing an existing file whose graph is technically valid but semantically or visually poor.

The goal is not merely to satisfy the JSON Schema. The file must provide a faithful, inspectable **semantic view of the computation**: meaningful hierarchy, explicit important operations, truthful data dependencies, exact checkpoint parameter bindings, concise formulas, visible computational constants, and names that let a reader follow values from ports through equations without translation.

This skill owns authoring procedure and presentation-quality rules. The normative exchange contract remains:

- `docs/spec/backend/model-owned-architecture.md`
- `docs/spec/backend/architecture-definition.schema.json`
- `docs/spec/backend/architecture-analysis.md`
- `docs/spec/ui/architecture-explorer.md`

Examples under `examples/model-owned-architecture/` illustrate the format but do not override those specifications or this authoring discipline.

## Load the minimum required context

Before authoring:

1. read the current model-owned architecture specification and JSON Schema;
2. read the Architecture Explorer presentation rules relevant to cards, groups, parameters, constants, repetitions, and Shared structures;
3. inspect the model's actual implementation or authoritative architecture description;
4. inspect the exact checkpoint tensor inventory and shapes that the JSON will bind;
5. inspect the existing `architecture.json` only as evidence of current intent, not as a structure that must be preserved.

Do not infer an architecture from tensor-name prefixes alone. Do not copy the Python/PyTorch module tree mechanically. Do not preserve obsolete visual structure merely because an earlier JSON used it.

## Author the semantic graph first

Design the computation before writing JSON. Identify the model's meaningful stages, the values crossing those stages, the trainable parameters used by each operation, and the real branching, residual, state, recurrent, or conditioning paths.

Choose hierarchy from **semantic computational units**, not from source-code nesting. A group should exist because its children form a useful architectural concept that a reader may collapse, expand, isolate, or compare. Avoid wrapper groups that add no information.

Examples of useful groups include an encoder block, attention block, MLP, temporal predictor, gating block, state-update block, projection head, or objective-specific head when those are real units of the model. These are examples, not a fixed vocabulary.

Do not put the whole model under a redundant group whose only purpose is to repeat the model name. Do not create a group solely to make the diagram look boxed. Do not introduce a group around an already meaningful component unless the additional boundary has a real interface or architectural purpose.

## Decompose compound computations instead of hiding them in one card

A compound component must be expanded into internal operations when a single operation card would hide useful structure.

Use the following test: if understanding the component requires seeing independent branches, distinct trainable projections, a meaningful normalization, a gate, a softmax, a state transition, a residual merge, a multi-stage transform, or another semantically important dependency, represent those as child operations rather than compressing all of them into one formula.

**Attention is the canonical example, not a special case.** An Attention group should normally contain the actual Q, K, and V projections and the meaningful computation that follows them, rather than a single `Attention` card containing the entire equation. Q/K/V are naturally parallel branches; score computation, scaling/masking when applicable, softmax, value aggregation, and output projection should remain explicit at the level needed to understand the real model.

Apply the same rule everywhere else. For example, an MLP with gate/up branches and a multiply should expose that branch structure; a predictor with conditioning and state fusion should expose those operations; a gated state update should expose the transforms and merge that define the update. The criterion is semantic value, not the name of the component.

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

Keep residual bypasses, parallel branches, masks, conditioning signals, context, prior/next state, and other semantically distinct flows separate. Never merge signals because they have the same shape or a similar label.

When a group is expanded, its internal routes must explain how its boundary inputs become boundary outputs. Do not create a parent formula that contradicts or duplicates the child graph.

Use `data`, `state`, and `context` edge kinds according to their real role. Do not represent a recurrent or state dependency as ordinary serial data merely to simplify layout.

Shapes should be as truthful as the contract permits: constants where known, declared symbols where symbolic, display-only expressions where appropriate, and explicit unknowns when unresolved. Never fabricate a dimension for presentation.

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

For every card, align port labels, formula variables, parameter references, and scalar attribute names. A reader should not need to translate between UI labels and equations.

### 5. Bind real parameters

Map each logical parameter to the exact checkpoint tensor name and shape. Keep each reference on the operation that consumes it. Verify tied/shared parameters deliberately.

### 6. Add repetitions and Shared metadata

Only after the concrete graph is correct, add repetition windows and eligible Shared families. Presentation metadata must annotate the real graph, never compensate for a missing graph.

### 7. Validate contract and presentation

Validate against the checked-in JSON Schema and repository-native model-defined architecture tests. Then load the actual model in Architecture Explorer and inspect the visual result. Schema validity alone is insufficient.

## Review checklist

Before accepting an `architecture.json`, verify all of the following:

1. The hierarchy reflects semantic computational units rather than the implementation class tree.
2. No redundant wrapper group exists only for presentation.
3. Compound blocks are decomposed wherever a monolithic card would hide important internal structure.
4. Attention, when present, uses the same general decomposition rule as every other compound component; Q/K/V and other meaningful suboperations are not hidden merely for compactness.
5. Port labels and formula variables use the same names.
6. Formula symbols have obvious sources in ports, parameters, attributes, or standard math.
7. Parameter bindings use exact checkpoint tensor names and truthful shapes.
8. Parameters appear on the operations that consume them, not aggregated on parent groups.
9. Computational scalar constants/hyperparameters that matter to understanding are exposed as operation attributes with real names and values.
10. Attributes are not a configuration/provenance dump.
11. Labels are concise, canonical, English, and free of design-history wording.
12. Real residual, branch, mask, context, state, and conditioning paths are preserved.
13. Repetitions annotate explicit concrete instances and preserve order/variants.
14. Shared templates declare exact correspondence and do not imply tied weights.
15. Unknown information remains explicitly unknown rather than guessed.
16. The graph passes schema/backend validation and is also understandable when viewed collapsed and expanded in Architecture Explorer.
17. Matrix/vector buttons resolve to the intended real tensors for representative operations.
18. The expanded graph does not rely on a giant parent formula to explain computation that should exist as child nodes.

## Common failure modes

**Valid JSON, bad architecture:** passing the schema proves the exchange format, not the quality of the semantic view. Review the actual canvas.

**One giant operation per subsystem:** split meaningful computation into child operations and retain the subsystem as a group.

**Framework-tree transcription:** redesign around mathematical units and real interfaces instead of preserving every Python module boundary.

**Formula/port drift:** rename the visible ports or the formula so both use one vocabulary.

**Constants hidden in prose:** move real computational scalar values into operation attributes when the UI should show them on the card.

**Parameters on group cards:** attach them to the child operations that consume them.

**Historical labels:** remove version/change-history wording from visual names and use `architecture_revision` for revision identity.

**Fake compactness:** use groups/repetitions for progressive detail instead of deleting real branches or dependencies.

**Shared confused with tied weights:** structural correspondence and parameter identity are separate claims; represent each truth independently.

## Final principle

Treat `architecture.json` as an authored mathematical interface to the model, not as serialization of the implementation and not as a hand-drawn diagram. A high-quality file lets a reader move from model-level structure to local equations and real weights without encountering hidden computation, unexplained names, invented topology, or historical clutter.
