# API contract

## Ownership and authority

The API is a first-class, contract-first project boundary. It is not generated conceptually from Python implementation classes and is not owned by either backend or UI.

`openapi.yaml` is the machine-readable authority for HTTP endpoint paths, request/response schemas, identifiers, and status codes. `binary-streaming.md` is the authority for the progressive binary body used by long operations. This document owns the cross-cutting semantics that connect those two contracts.

The proof of concept does not expose public API versions such as `/v1` and does not require backwards compatibility with older contract revisions. Backend and UI evolve together against the current accepted contract. The `version: current` field required by OpenAPI is document metadata, not a public protocol version.

The Architecture Explorer endpoint/records and inventory capability extension are published in OpenAPI, conformance fixtures, and generated UI bindings together. [architecture-explorer.md](architecture-explorer.md) owns the graph semantics. The runtime architecture route remains pending implementation.

## Design rules

Capabilities use explicit, strongly typed endpoints rather than a universal `execute` endpoint carrying an operation-name string.

JSON is used for commands, descriptors, metadata, tokenizer results, and other small responses. Bounded architecture graphs are structural metadata, not numerical execution results. Tensor values and large numeric derived results are never serialized as JSON or Base64.

The UI never sends or receives backend filesystem paths. Public model identity, tensor identity, session identity, and operation identity are logical identifiers only.

Long operations use ordinary HTTP streaming rather than WebSockets. The request that starts the operation carries its progressive response. Every long-operation response exposes a unique `X-Operation-Id` header, and the backend CORS configuration must expose that header to the browser.

## Endpoint set

The currently published API includes the proof-of-concept operations and the accepted post-PoC input-embedding lookup:

| Operation | Method and path | Response |
| --- | --- | --- |
| List models | `GET /models` | JSON |
| Create session | `POST /sessions` | JSON |
| Get session | `GET /sessions/{session_id}` | JSON |
| Delete session | `DELETE /sessions/{session_id}` | Empty |
| List tensors | `GET /sessions/{session_id}/tensors` | JSON |
| Stream tensor | `GET /sessions/{session_id}/tensors/{tensor_id}/data` | Binary stream |
| Stream tensor statistics | `GET /sessions/{session_id}/tensors/{tensor_id}/statistics` | Binary stream |
| Stream row/column distributions | `GET /sessions/{session_id}/tensors/{tensor_id}/distributions` | Binary stream |
| Tokenize text | `POST /sessions/{session_id}/tokenize` | JSON |
| Look up input embeddings | `POST /sessions/{session_id}/embeddings` | Binary stream |
| Cancel long operation | `DELETE /operations/{operation_id}` | Empty |

The published architecture addition is `GET /sessions/{session_id}/architecture` (`getArchitecture`), returning the prepared structural result under the [architecture contract](architecture-explorer.md#prepared-architecture-endpoint). It is read-only retrieval, not an analysis trigger or a numerical long operation. No generic operation endpoint is introduced.

## Models and sessions

`GET /models` exposes the models discovered below the backend-configured local model root. A model descriptor may include architecture, model type, parameter count, total local size, and tokenizer availability when those values can be derived reliably. Local paths and the internal content fingerprint are not exposed.

`POST /sessions` binds a new session permanently to one public `model_id`. A session does not change model during its lifetime. Creation pins the model's current content fingerprint internally; subsequent model-dependent requests must reject changed content instead of silently reading a new revision. `GET /sessions/{session_id}` exists so a refreshed UI can reconnect while the in-memory backend session still exists. `DELETE /sessions/{session_id}` releases the session's ephemeral runtime state; it does not delete shared persistent artifacts. Deletion cancels that session's consumers, but must not cancel shared production still needed by another session.

Session IDs and long-operation IDs are UUIDs. Tensor IDs are opaque URL-safe identifiers returned by the tensor inventory and must not be reconstructed by the UI from tensor names.

A valid model/session need not have a text tokenizer. Architecture retrieval uses the same pinned snapshot independently of tokenizer and numeric-weight availability; its resource references never authorize cross-model access or rebind an existing session.

## Tensor inventory

`GET /sessions/{session_id}/tensors` returns descriptors for the session model's tensors. Each descriptor includes the full logical tensor name, logical path segments, shape, rank, element count, physical storage dtype/format metadata, and the canonical logical visualization dtype.

The UI builds its hierarchical inventory from the returned logical path segments. Filesystem structure is not part of the hierarchy contract.

The current canonical logical visualization dtype is always `float32`. Physical representations such as FP16, BF16, INT8, NF4, or another quantized format remain backend concerns.

For the accepted quantized-architecture extension, [capability separation](architecture-explorer.md#quantized-checkpoint-capability-separation) defines explicit inventory coverage and exclusion of packed/unresolved storage from the existing logical data path. Existing native inventories/IDs/bytes remain unchanged. Parameter/storage references that cannot be served numerically remain descriptive graph records, never falsely materializable tensor identities.

## Logical tensor data

`GET /sessions/{session_id}/tensors/{tensor_id}/data` starts a long operation whose binary-stream result kind is `tensor`.

The request is for one complete logical tensor. Transport frames are only delivery chunks; they are not addressable tensor tiles and are not cache units.

The logical payload is IEEE-754 `float32`, little-endian, in deterministic C-contiguous tensor order. Shape remains generic for arbitrary rank even though the proof-of-concept UI directly visualizes rank-1 and rank-2 tensors.

The backend may stream already-compatible physical values with minimal transformation or materialize/dequantize another physical representation into this logical form. That distinction is invisible to the UI. Describing a quantized parameter in a graph does not implement its decoder or authorize integer-to-float reinterpretation.

## Tensor statistics

`GET /sessions/{session_id}/tensors/{tensor_id}/statistics` starts an independent long operation whose result kind is `tensor_statistics`. Statistics are reusable derived artifacts and do not block the tensor-data stream.

Statistics are computed over the finite values of the logical `float32` tensor and expose:

- total element count;
- finite-value count and non-finite-value count;
- minimum and maximum finite value;
- arithmetic mean;
- population standard deviation (`correction = 0`);
- percentiles `p01`, `p05`, `p50`, `p95`, and `p99`.

For sorted finite logical values `x[0..n-1]`, percentile fraction `q`, `h = (n - 1) * q`, `i = floor(h)`, and `j = ceil(h)`, the exact percentile definition is `(1 - (h - i)) * x[i] + (h - i) * x[j]`, for `q = 0.01, 0.05, 0.50, 0.95, 0.99`. A single finite value yields that value for every percentile. This specifies a mathematical result, not a requirement to call a particular PyTorch primitive or accept its size limits.

Mean and population standard deviation use sufficiently wide accumulation (at least float64, with stable variance evaluation) to avoid float32 sum, squared-difference, or range overflow. Derived floating reductions (mean, stddev, and percentiles) are compared using `abs(actual - expected) <= 1e-7 + 1e-6 * abs(expected)`. CPU and CUDA need not produce bitwise-identical reductions. Minimum/maximum, counts, raw materialized float32 bytes, and histogram integer counts remain exact; these tolerances do not authorize changes to logical values.

The default rendering transfer may use `p01` and `p99` as robust anchors, but that is a UI/rendering decision. The statistics contract itself does not apply a visual normalization.

For empty tensors and tensors containing only NaN/infinities, finite-derived scalar fields are returned as JSON `null` in the stream metadata and `finite_count` is zero.

## Row and column distributions

`GET /sessions/{session_id}/tensors/{tensor_id}/distributions` is defined for rank-2 tensors only. A valid request for another rank returns HTTP `422` before streaming begins.

The distribution artifact uses exactly **100 shared linear bins** over the tensor's complete finite-value range. All row profiles and all column profiles use the same bin domain so hover never changes the histogram coordinate system.

For finite `minimum < maximum`, zero-based bin index is `clamp(floor((float64(value) - float64(minimum)) / (float64(maximum) - float64(minimum)) * 100), 0, 99)`. Perform this arithmetic in float64, including subtraction, so opposite-sign finite float32 extremes do not overflow. The last bin includes the maximum. Non-finite values are excluded.

When all finite values are equal (`minimum == maximum`), every finite value is placed in bin index `50`.

The binary logical payload contains two contiguous `uint32` little-endian C-order sections:

- `row_counts` with shape `[rows, 100]`, mapping to the right-hand row-distribution surface;
- `column_counts` with shape `[100, columns]`, mapping to the bottom column-distribution surface.

With no finite values (including shapes `[0, columns]`, `[rows, 0]`, and `[0, 0]`), both domain endpoints are `null` and every count is zero. Section shapes remain exactly as declared; an empty source can still have a nonempty all-zero section.

Counts are raw histogram counts. Each bin must fit `uint32` (`0..4294967295`); accumulate/check without wrapping and reject an unrepresentable result with `unsupported_size`. Before streaming use HTTP `422`; after streaming starts use `ERROR_JSON`. Visual intensity normalization belongs to the UI and must remain stable while the tensor is open.

Metadata declares the domain, bin count, shapes, byte offsets, and byte lengths of both sections.

## Tokenization

`POST /sessions/{session_id}/tokenize` runs the real Hugging Face tokenizer associated with the session model. The browser never reimplements tokenizer logic.

The request contains current text and optionally `add_special_tokens`, which defaults to `true`. The response echoes both and returns the tokenizer sequence in order.

Each token exposes its numeric ID, tokenizer-native representation, decoded representation, special-token flag, and source offsets when provided by the tokenizer. Token `index` values are consecutive from zero. `start` is inclusive and `end` exclusive, measured in **Unicode code points into the exact echoed input**, without normalization. They are not UTF-8 byte offsets, UTF-16 code-unit offsets, or grapheme-cluster indices. Convert native offsets into these units where necessary. Both fields appear together or both are omitted, with `0 <= start <= end <= code_point_length(text)`.

Preserve overlapping source spans, including multiple byte-level tokens covering one code point. Do not fabricate disjoint boundaries. Per-token `decoded` strings need not concatenate to the input (a token may decode only part of a multibyte character). An absent valid span remains absent.

`special` identifies membership in the tokenizer's special-ID set, not whether the backend inserted that token. A special ID typed in source may have a valid span; an inserted special has none. The UI must not infer source presence from `special` alone.

For example, `A😀é<special>` has 13 code points: `A` at 0, `😀` at 1, `e` at 2, combining acute accent at 3, and `<special>` at `[4,13)`. Two byte-level tokens may both report `[1,2)`. The emoji's UTF-16 span `[1,3)` is not its API span. `api/fixtures/conformance.json` supplies a synthetic tokenizer result with these spans, a source-typed special, and an inserted special; it is an offset example, not a claimed model tokenization.

The UI prevents an older asynchronous result replacing newer editor state. Echoing input text provides the necessary check without introducing a separate request-sequence protocol. Static tokenizer context in an architecture graph does not itself tokenize anything.

## Input embedding lookup

`POST /sessions/{session_id}/embeddings` accepts `InputEmbeddingsRequest` with ordered `token_ids` from the existing tokenizer result for that session model, including special tokens. It starts an LMEX long operation with kind `input_embeddings`. It performs only embedding-table row lookup: no positional encoding, forward execution, logits, or generation.

For IDs `[t0, ..., tN-1]`, the complete logical result has shape `[N, hidden_size]`. Row `i` is the input embedding for `ti` in canonical little-endian float32 C-order. Preserve order and duplicates; equal IDs yield equal row bytes. Do not sort/deduplicate the output, normalize, or substitute output weights. Empty IDs are valid with a known positive hidden size and yield `[0, hidden_size]` with zero payload bytes. Return only requested rows, never the full vocabulary table as a shortcut.

Resolve a trustworthy input-embedding source from supported architecture/configuration and checkpoint metadata, not a tensor-name guess alone. If source, vocabulary range, hidden size, or logical representation cannot be established, use `unsupported_representation` (HTTP 422 before streaming, terminal `ERROR_JSON` afterward), including for empty input with unknown hidden size. Preserve lazy access; do not load/materialize the full table or model. The accepted input-table mappings are owned by [backend model resolution](../backend/models.md#input-embedding-table-resolution). Architecture support for a hybrid or visual model does not independently extend this numeric capability. I/O, source-mutation, protocol, and rendering failures must not be reclassified as unsupported capability.

Validate the entire request before emitting META/DATA: each ID is a nonnegative safe integer below vocabulary size and addresses a row in the resolved table. Invalid fields, strings, booleans, negative/fractional or out-of-range IDs yield `validation_error`. An added token without a corresponding row is invalid. Never deliver a valid prefix of an invalid request. Malformed JSON, unknown sessions/models, and pinned content changes use existing errors.

Metadata uses `InputEmbeddingsMetadata` with exact ordered `token_ids` echo and shape, but no checkpoint `tensor_id` or `name`. Consumers check echoed IDs against the request/session before display; shape/length alone cannot detect stale data. IDs are control metadata; values are binary. If META exceeds the 1 MiB control-frame limit, reject `unsupported_size` before META/DATA; never truncate. Safe shape/product/byte limits apply before allocation.

Reuse `BinaryStream`, `X-Operation-Id`, CORS exposure, `Cache-Control: no-store`, progressive delivery, independent cancellation, and terminal semantics. Share work only for the same model content and ordered IDs; cancellation preserves other consumers. No new public artifact/statistics/distribution endpoint for this derived matrix is introduced.

## Long operations, sharing, and cancellation

Each streaming request creates a unique consumer-visible `operation_id`, returned in `X-Operation-Id`. Equivalent requests may share artifact production internally, not a public operation ID. Each consumer has its own stream and ID.

`DELETE /operations/{operation_id}` cancels only that consumer, is idempotent, and returns `204` for active, terminal, or unretained operations. Shared work may continue while another consumer needs it. Aborting `fetch` also cancels the consumer; use explicit cancellation where needed for prompt release of queued/backend work.

Operations are ephemeral, not listed or retained as history. Persistent successful results are cache artifacts. Closing an architecture weight modal uses this same path; reading an already prepared architecture does not create a numerical operation.

## Errors

Before streaming, use normal HTTP status codes and the OpenAPI `Error` schema. After streaming begins, failures use the `ERROR_JSON` terminal frame; HTTP status has already been committed. Connected cancellations terminate with `CANCELLED`, not an error frame.

The closed machine-readable error code set is:

| HTTP status before streaming | `code` | Meaning |
| --- | --- | --- |
| 400 | `malformed_json` | Request body is not valid JSON |
| 404 | `model_not_found`, `session_not_found`, `tensor_not_found` | Unknown resource |
| 409 | `model_content_changed` | Content differs from the pinned fingerprint |
| 422 | `validation_error` | Invalid fields, types, identifiers, or constraints |
| 422 | `unsupported_representation`, `unsupported_rank`, `unsupported_size` | Unsupported materialization, rank, size, or uint32 count |
| 503 | `resource_exhausted` | Insufficient runtime resources |
| 500 | `internal_error` | Unexpected failure |

`openapi.yaml` enumerates applicable responses per published operation. Cancelling a syntactically valid but unknown/unretained UUID returns `204`, not `404`. Malformed identifiers use `422`. Cancellation is not an error code. Post-header failures use the same `Error` object inside `ERROR_JSON`, including before META.

Error `code` is for logic; `message` is human-readable. Messages/details must not expose paths, tracebacks, or private model locations. All responses require `Cache-Control: no-store`. Allowed cross-origin streams expose `X-Operation-Id`, including streams terminating before META. Architecture capability reasons are separately typed results defined in its document, not additions to this error-code set.

## Cross-field and allocation constraints

Shape dimensions, ranks, lengths, counts, indices, and offsets are nonnegative safe integers at most `2^53 - 1`, including computed products/sums. The LMEX uint32 frame-length bound still applies. Reject unsupported sizes before allocating; rounding/overflow must never validate a size. A safe size may still fail allocation with `resource_exhausted`.

- Tensor descriptors require `rank == len(shape)` and `numel == product(shape)`. The empty product for scalar `[]` is 1; any zero dimension gives 0. Tensor metadata requires `byte_length == 4 * product(shape)`.
- Input embeddings require rank 2, positive hidden size, `shape[0] == len(token_ids)`, and `byte_length == 4 * product(shape)`. IDs match the request in order and multiplicity.
- Statistics require `count == finite_count + non_finite_count`. With zero finite values all finite-derived fields/percentiles are null; otherwise they are finite, ordered within the finite range, with nonnegative stddev.
- Distribution sections occur in row/column order, with `[rows, 100]` and `[100, columns]` shapes and byte lengths four times products. Row offset is zero; column offset is row byte length; total length is their sum. Domain endpoints are both null or finite/ordered. Null domains imply all-zero counts; row-count and column-count sums each equal the finite source count.
- Progress requires `completed <= total` when a positive `total` is supplied.

Schemas validate local structure; implementations also check products, sums, ordering, source-dependent constraints, and token spans. Shared fixtures include valid/invalid cross-field cases. Architecture reference closure, symbolic dimensions, union states, and bounded graph JSON add their own checks without changing these numerical rules.
