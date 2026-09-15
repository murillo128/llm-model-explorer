# Architecture Explorer API contract

## Status and ownership

The static Architecture Explorer contract is published in `openapi.yaml` with shared conformance fixtures and generated UI types/runtime schemas. [contract.md](contract.md) retains shared API/error/session semantics. Architecture analysis and the runtime retrieval route remain pending their implementation children; contract publication does not mean that the endpoint exists.

The graph describes architecture, not execution. [Backend analysis](../backend/architecture-analysis.md) owns semantic truth and [UI behavior](../ui/architecture-explorer.md) owns layout and interaction. JSON contains bounded structural metadata only. Weights continue to use the existing progressive binary endpoints.

## Prepared architecture endpoint

Add `GET /sessions/{session_id}/architecture`, operation ID `getArchitecture`, using the existing session parameter, error schema, and `Cache-Control: no-store` convention. It reads the result prepared for the exact pinned model snapshot; it never starts analysis, retries analysis, executes a model, or downloads anything. It has no request body, operation ID, polling protocol, or WebSocket counterpart.

A successful HTTP 200 uses a discriminated `ArchitectureResponse`:

| Field | Contract |
| --- | --- |
| `status` | `available` or `unavailable`. No running/pending background-job state. |
| `model_id` | The session's public logical model ID, never a path. |
| `diagnostics` | Array of safe `ArchitectureDiagnostic` records; empty is allowed. |
| `graph` | Required only for `available`; absent for `unavailable`. |
| `reason` | Required only for `unavailable`: `unsupported_architecture`, `analysis_failed`, `restart_required`, `unsupported_size`, or `cache_unavailable`. |
| `requires_restart` | Required boolean on `unavailable`; true when another startup preparation is required. |

An available graph has `coverage: complete | partial`. Missing quantization decoding is not by itself incomplete architecture coverage. Complete means complete within the declared scope and abstraction, not every instruction of an implementation or every modality.

Use existing HTTP errors for malformed session identifiers (422), unknown sessions (404), changed pinned content (409 `model_content_changed`), runtime resource exhaustion (503), and unexpected request failures (500). A known analysis failure, unsupported architecture, or missing prepared result is the typed capability result above, not a fabricated empty graph or a new HTTP error-code family. Permission/I/O failures during an actual request are not silently changed into successful empty results.

For a newly discovered model/snapshot that was not prepared at startup, a new valid session may obtain `unavailable/restart_required`. Existing pinned-session mutation still returns 409. Missing/corrupt graph artifacts discovered after readiness return `cache_unavailable` with `requires_restart: true`; do not regenerate on a GET. Unsupported descriptions also need a compatible analyzer installation before restart can help; diagnostics must not promise that restart alone adds support.

## Graph document and identity

`ArchitectureGraph` is an object with required `graph_id`, `scope`, `coverage`, `symbols`, `nodes`, `edges`, `repetitions`, `parameters`, and `diagnostics` fields. Arrays may be empty except `nodes`, which must contain the supported scope's structure. `scope` is `language_model` or `visual_encoder_predictor`. There is no universal text-tokenizer or language-head requirement.

`graph_id` is an opaque URL-safe analysis identity, stable for the same checkpoint content and semantic producer/schema revisions. It is not the raw internal checkpoint fingerprint. Model/session IDs, local paths, viewport settings, and timestamps are not graph-identity inputs. The public `model_id` lives in the response envelope, not in the cached graph, so relocating a checkpoint with a directory-name fallback does not corrupt reusable content.

All record IDs are nonempty URL-safe strings, at most 256 characters, scoped by `graph_id`; clients treat them as opaque. Display labels, module names, and layer numbers are not interchangeable with IDs. Every cross-reference must resolve in the same graph. Identity/repetition expansion is supplied by the backend; the browser must not synthesize tensor IDs or infer computational edges from names.

The first implementation sends the full instance graph. Repetition records group existing instances for display; no template interpreter or lazy graph-fetch protocol is required. Server responses may use ordinary HTTP compression, but the decoded JSON size is bounded and all required instances remain present. No camera positions, renderer classes, modal commands, local filesystem paths, executable code, tensor values, or runtime execution IDs belong in graph records.

## Nodes, ports, edges, and repetition

`ArchitectureNode` requires `id`, `kind`, `label`, `ports`, `parameter_ids`, `references`, `attributes`, and `provenance`. `kind` is `group`, `operation`, `input`, `output`, `context`, or `state`. Optional `parent_id` names a group; absent means a root. Optional `operation` is a nonempty semantic string identifier, not a callable or endpoint name. All `provenance` fields are arrays of `ArchitectureProvenance` records; all `references` fields are arrays of `ArchitectureReference` records. Optional `description` and `formula` are non-executable explanatory text. A group has ordered `children` IDs; non-groups do not. Unknown components carry a diagnostic and may not be disguised as understood operations.

`ArchitecturePort` requires `id`, `direction` (`input` or `output`), `label`, and `shape`. Port IDs are unique within their node. A shape is an ordered array of dimensions, or null when rank itself is unknown. Dimension records use a `kind` discriminator: `constant` with a nonnegative safe-integer `value`; `symbol` with a declared `name`; `expression` with display-only `text` and a list of declared `symbols`; or `unknown` with a safe `reason`. Expressions are never executed by the client. `symbols` at graph level is an array of records with required `name` and `meaning` strings; it declares each symbol, distinguishing batch, token sequence, frame, spatial, context, and target dimensions. Scalar shape is `[]`, not null.

`ArchitectureEdge` requires `id`, `source` and `target` endpoints (`node_id`, `port_id`), `kind` (`data`, `state`, or `context`), and `provenance`. Optional `label` explains a role. Shapes are obtained from endpoint ports rather than stored as a conflicting second shape. Endpoints must exist and have compatible directions; group boundary ports explicitly preserve crossings. Prior/next-state ports represent recurrence symbolically, not an executed timeline or an invented cache value.

`ArchitectureRepetition` requires `id`, `parent_id`, `label`, and ordered `instances`. Each instance names its existing group `node_id`, integer `index`, and `variant` identifier. All instances retain their actual children, edges, and parameter bindings. Different variants are not asserted equivalent. Repetition ordering must agree with parent ordering; it does not authorize reordering hybrid layer patterns.

`attributes` is an array of records with `name`, a scalar or scalar-array `value` (string, finite number, boolean, or null), and `provenance`. It contains architectural facts, not arbitrary nested library objects or numeric tensor payloads. Explanatory text fields are limited to 16,384 characters, and labels/names to 1,024 characters; never interpret checkpoint strings as HTML or script.

## Parameters and explorer resource references

`ArchitectureParameter` requires `id`, `name`, `logical_shape`, `binding`, `storage`, `inspection`, and `provenance`. `logical_shape` uses the dimension representation above. `binding` is `native`, `alias`, `fused_region`, `quantized`, or `unresolved`. An alias supplies `alias_of`; a fused region supplies a declarative `region` object with `storage_name` and plain-text `description` of its axes/ranges when known, never an executable slice expression. Storage records give physical tensor `name`, `dtype`, and integer `shape`; optional `role` distinguishes packed data, scales, or other verified encoding data. Storage names are checkpoint tensor names, not shard paths.

`inspection` is a discriminated object. With `status: available`, it contains a real existing inventory `tensor_id` for one complete logical rank-1/rank-2 tensor, through a native or admitted packed decoder path. With `status: unavailable`, it contains `reason` (`unsupported_representation`, `unsupported_rank`, `requires_view`, or `unresolved_binding`) and a safe `message`, and no fabricated actionable `tensor_id`. A known native tensor of higher rank retains its storage/logical descriptor even though this modal cannot render it. Aliases may resolve to an existing actionable tensor only when identity and full geometry are verified. Quantized bindings preserve their physical storage records and must agree with the inventory's logical name, shape, and admitted packed relationship. Packed integers, scales, arbitrary regions, and flattened higher-rank weights are not offered as substitute logical matrices.

`ArchitectureReference` is tagged as `module` (original module `name`), `parameter` (`parameter_id`), or `tokenizer` (the current model's tokenizer capability, without a URL). Nodes may use zero or more references. Model scope is implicit in the session/response; references must not cross models. Edges have stable identities for future resource association but no activation references in this increment. Visual context does not advertise a nonexistent upload or video-processing endpoint. The UI decides which supported action to expose; the backend never directs it to open a modal or route.

`ArchitectureProvenance` records use `kind: configuration | storage | description`, with a safe `source` identifier and optional `revision`/`rule`. Configuration sources may be JSON pointers; storage sources are tensor names; description sources identify packaged descriptions/reviewed rules. Provenance is evidence of origin, not proof that inference ran. `ArchitectureDiagnostic` requires `code` and `message`, with optional `node_id` or `parameter_id` targeting a record in this graph. It must not expose traceback or filesystem content.

## Quantized checkpoint capability separation

The existing tensor endpoints remain logical-value endpoints, not a raw quantization inspector. Retain a complete validated physical inventory internally. Publish through `TensorInventory.tensors` complete logical tensors whose native or admitted packed decoder path and role/geometry are verified. The admitted GPTQ Int4 and ModelOpt NVFP4 groups each expose one logical `…weight` matrix in `[output, input]` order. Encoding auxiliaries remain internal. The backend owns decoding; data, statistics, distributions and architecture inspection use the same logical tensor identity and float32 values.

`TensorInventory` includes `coverage: complete | partial` and `diagnostics` (safe code/message records without graph-local targets). Unquantized baseline inventories retain every prior tensor and report complete. Quantized inventories report complete when every logical parameter is actionable and all physical records are accounted for by logical tensors or validated encoding companions. Auxiliary count differences do not imply partial coverage. Unknown/orphan records remain non-actionable with precise partial-coverage diagnostics. Preserve existing native tensor IDs and bytes. Unknown/non-exposed IDs retain `tensor_not_found`; known but unsupported operations retain their existing typed errors.

Tokenizer availability continues to be independent of tensor and architecture coverage. A V-JEPA 2 session is valid without text-tokenizer assets; clients must not issue a text-tokenization request merely to retrieve its graph. Input-embedding lookup still requires a verified input table and supported model semantics; it must not guess a packed tensor or use a V-JEPA visual representation. Backend row access supports admitted decoded logical matrices without independently extending embedding model coverage.

## Validation, limits, and publication

Reject unknown discriminator values and malformed records rather than silently rendering approximate structure. Check ID uniqueness, reference closure, parent/child agreement, acyclic containment, repetition membership/order, alias termination, port directions, parameter inspection membership in the session inventory, and resolvable shape symbols. Check all integer products/ranges with the existing safe-integer rules. Disagreement of known connected dimensions must be diagnosed, not overwritten by the frontend.

The decoded architecture response is limited to 32 MiB (33,554,432 bytes), with no numeric payload hidden in attributes. Enforce byte limits during construction/serialization and bounded reads, not only after an unbounded allocation. Exceeding the bound yields a localized unavailable `unsupported_size` result; it never silently truncates nodes. Text/ID/collection validation must also guard nesting and counts within this budget. Every selected reference must fit and retain all required instances; hitting a limit is not successful reference acceptance.

The publication child owns the OpenAPI schema transcription, closed union schemas, positive/negative fixtures, validator integration, and regenerated TypeScript/runtime bindings as one coherent contract change. Include complete/partial/unavailable graphs, every reason, no-tokenizer V-JEPA, symbolic/unknown dimensions, fused/quantized/native/rank-limited bindings, invalid references, oversize payloads, changed sessions, and current tensor-inventory compatibility. No backend or UI may ship against a divergent handwritten interface. This contract adds no generic execute endpoint, generation polling, public `/v1`, tensor JSON transport, or second cache API.
