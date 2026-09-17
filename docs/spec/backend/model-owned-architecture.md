# Model-owned architecture definitions

## Scope and trust boundary

A local Hugging Face-style checkpoint may contain an optional `architecture.json`
next to `config.json` and its Safetensors weights. This is a static description
supplied by the model author, not executable model code and not automatic
architecture inference. Any model family can use it without an Explorer adapter.
The existing catalogue still requires a valid configuration and weight inventory;
a description by itself is not an admitted checkpoint.

The loader checks the JSON, graph relationships and parameter bindings. It does
**not** prove that the graph implements the checkpoint's `forward()`, or that the
model has been trained. Every resulting graph includes a visible model-supplied
origin notice and description provenance without a source-verification claim.
Authors own correspondence tests between their implementation and exporter.

The transport scope is `model_defined`. The author's free-text `scope`, for
example `feature_encoder`, `world_model` or `training_objective`, is retained as
declared metadata. New domains or operation names do not require new enum values,
family registrations or sport-specific frontend code.

## Version 1 file contract

The published [JSON Schema](architecture-definition.schema.json) defines the
closed record format. An [example](../../../examples/model-owned-architecture/architecture.json)
describes an encoder with a linear projection and GELU inside a group. The
[example instructions](../../../examples/model-owned-architecture/README.md)
explain how to generate inspectable weights without committing model payloads.

Required top-level fields are `schema_version: 1`, `architecture_revision`,
`name`, `scope`, and a nonempty `nodes` array. Optional `symbols`, `edges`,
`parameters`, `repetitions`, and `templates` default to empty arrays. `coverage` defaults to
`complete`; `partial` requires `incomplete_reason`. Complete is a declaration of
coverage within the author's scope, not an independent semantic certification.
Missing required storage always downgrades the resulting graph to partial.

Nodes have file-local IDs, kinds, labels, ports, optional parent/group children,
parameter references, explanatory descriptions/formulas and scalar metadata.
Operations require a name but names are open-ended: an unfamiliar operation is
still displayable. Ports and dimensions reuse the architecture API's records:
constant, declared symbol, display-only expression, or explicitly unknown.
Expressions and formulas are **never evaluated**. Group crossings are explicit
ports; all repeated instances are written into the file. Repetition metadata
only groups existing instances; it is not a loop or template interpreter.

Parameters declare only an ID, the exact checkpoint tensor `name`, and `shape`.
Version 1 binds complete native F32/F16/BF16 tensors; it does not introduce new
packed decoders, arbitrary slices, or user-supplied storage descriptions. Reuse
one parameter ID across nodes for shared weights. A fused QKV matrix can be
referenced as a complete native parameter and followed by declared split
operations. Native rank above two retains its descriptor with unavailable matrix
inspection; absent storage is unresolved, not fabricated. Existing packaged
quantized descriptions retain their existing behavior when no JSON is supplied.

The backend generates graph-scoped IDs from the pinned content and producer
revision. It resolves numeric tensor IDs and storage metadata from the admitted
inventory; the file cannot supply those identities, verified provenance or cache
keys. References to modules remain author declarations. A tokenizer capability
reference is valid only when the selected model actually has that capability.

`schema_version` is the Explorer exchange format. `architecture_revision` is
chosen by the author and can change with every experiment without changing the
Explorer. Python may generate the JSON in the model's development/export
environment, but Explorer does not import or execute checkpoint Python.

## Declared Shared structures

`templates` is an additive, optional version-1 extension. Existing definitions
without it remain valid. Readers predating this extension reject the new field;
update Explorer before using it. The model-defined producer revision is bumped
so old prepared artifacts cannot conceal the new import behavior. The HTTP graph
contract is unchanged.

Each family declares `id`, `label`, `component_role` (`attention` or `mlp`), and
at least two ordered `instances`. Each instance declares its source group
`node_id` plus four exhaustive role-mapping arrays:

- `nodes`: `{role, node_id}` for the group and every descendant;
- `ports`: `{role, node_id, port_id}` for all their ports, including unused ones;
- `edges`: `{role, edge_id}` for exactly the internal edges and group forwarding;
- `parameters`: `{role, parameter_id}` for all referenced logical parameters,
  including parameter resource references.

These are file-local references. The importer maps each reference to its own
runtime record, never to the first instance's weights. Family `revision` and
`provenance` are backend-assigned and cannot be supplied by the author. Root
groups declare `attributes: [{"name": "semantic_role", "value": "attention"}]`
or the equivalent `mlp` role. Grouping never implies weight tying.

The same conservative correspondence validator used for packaged descriptions
checks full closure, unique roles/targets, source scope/order, ordered containment,
operations, formulas, descriptions, attributes, port shapes and edge directions.
Separate Q/K/V branches and prior/next state ports must correspond exactly. It
checks declared structural equivalence, **not** source-code equivalence. The
model-supplied origin notice remains visible. Unknown shapes or unequal
components cannot establish a Shared family. Omit the declaration for components
whose equivalence cannot be asserted; they remain ordinary inspectable groups.

Repetitions and Shared families are independent annotations. Repetitions enable
layer windows; templates enable shared-component navigation. Instances follow
their enclosing repetition order, or sibling order within a common parent when
no repetition applies. No filename heuristic or automatic family discovery is
performed. See the complete [Shared example](../../../examples/model-owned-architecture/shared-architecture.json)
and its [loading instructions](../../../examples/model-owned-architecture/SHARED.md).

Invalid declarations make the supplied architecture unavailable without falling
back. Valid optional families that do not fit the output metadata budget are
omitted with `templates_omitted` when diagnostic space permits, otherwise logged;
the ordinary graph remains intact. Every declaration is still validated even
when output space is exhausted, so size omission never conceals invalid metadata.

## Selection, errors and lifecycle

Read the exact conventional filename `architecture.json`; no custom paths,
includes, remote references or plugin entry points are accepted. If absent, keep
the packaged-description selection path unchanged. If present, it takes
precedence, even for an otherwise supported packaged family. Invalid JSON,
unsupported versions, invalid graph relationships or contradictory bindings
produce a localized unavailable architecture result with safe diagnostics and
**no silent fallback**. Otherwise valid tensor inspection remains independent.

Read through the pinned source snapshot with the existing confinement and
regular-file rules. The definition is limited to 8 MiB before JSON decoding;
duplicate keys, non-finite constants and structurally excessive documents are
rejected. Output construction and serialization retain the existing 32 MiB graph
budget. Validate closure, unique identities, hierarchy, group ports, repetitions,
symbols and connected dimensions with the common graph validator. Unknown
configuration or payload fields are not passed through to runtime records.

The sidecar is already covered by the checkpoint's JSON file snapshot/content
fingerprint. Adding, removing or modifying it changes content identity and
invalidates pinned sessions normally. Cold startup builds a new artifact; warm
startup may reuse only the matching content/producer/schema artifact. GET reads
prepared state only and never reruns analysis. A malformed sidecar cannot reuse
an old graph or fall back to a packaged interpretation. Definitions are read-only.

## Validation and maintenance

The runtime schema records and checked-in JSON Schema must stay synchronized;
tests regenerate and compare the schema and validate the worked example. Changes
to the exchange contract require an explicit schema-version decision. Changes to
loader semantics bump its producer revision; architecture API changes keep their
normal generated-record and cache-invalidation discipline.

Test valid and invalid graphs, explicit instance identity, native weight values,
missing/wrong bindings, precedence and no fallback, schema/version/resource
bounds, read-only operation, safe diagnostics, mutation/cache behavior, and the
absence of model construction, tracing, code execution and payload reads during
analysis. Existing packaged descriptions and tensor/tokenizer workflows remain
regression gates. Successful static checks are not inference-correctness claims.
