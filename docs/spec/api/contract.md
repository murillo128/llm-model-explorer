# API contract

## Ownership and authority

The API is a first-class, contract-first project boundary. It is not generated conceptually from Python implementation classes and is not owned by either backend or UI.

`openapi.yaml` is the machine-readable authority for HTTP endpoint paths, request/response schemas, identifiers, and status codes. `binary-streaming.md` is the authority for the progressive binary body used by long operations. This document owns the cross-cutting semantics that connect those two contracts.

The proof of concept does not expose public API versions such as `/v1` and does not require backwards compatibility with older contract revisions. Backend and UI evolve together against the current accepted contract. The `version: current` field required by OpenAPI is document metadata, not a public protocol version.

## Design rules

Capabilities use explicit, strongly typed endpoints rather than a universal `execute` endpoint carrying an operation-name string.

JSON is used for commands, descriptors, metadata, tokenizer results, and other small responses. Tensor values and large numeric derived results are never serialized as JSON or Base64.

The UI never sends or receives backend filesystem paths. Public model identity, tensor identity, session identity, and operation identity are logical identifiers only.

Long operations use ordinary HTTP streaming rather than WebSockets. The request that starts the operation carries its progressive response. Every long-operation response exposes a unique `X-Operation-Id` header, and the backend CORS configuration must expose that header to the browser.

## Endpoint set

The proof-of-concept API consists of these operations:

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
| Cancel long operation | `DELETE /operations/{operation_id}` | Empty |

No generic operation endpoint is part of the proof of concept.

## Models and sessions

`GET /models` exposes the models discovered below the backend-configured local model root. A model descriptor may include architecture, model type, parameter count, total local size, and tokenizer availability when those values can be derived reliably. Local paths and the internal content fingerprint are not exposed.

`POST /sessions` binds a new session permanently to one public `model_id`. A session does not change model during its lifetime. Creation pins the model's current content fingerprint internally; subsequent model-dependent requests must reject changed content instead of silently reading a new revision. `GET /sessions/{session_id}` exists so a refreshed UI can reconnect while the in-memory backend session still exists. `DELETE /sessions/{session_id}` releases the session's ephemeral runtime state; it does not delete shared persistent artifacts. Deletion cancels that session's consumers, but must not cancel shared production still needed by another session.

Session IDs and long-operation IDs are UUIDs. Tensor IDs are opaque URL-safe identifiers returned by the tensor inventory and must not be reconstructed by the UI from tensor names.

## Tensor inventory

`GET /sessions/{session_id}/tensors` returns descriptors for the session model's tensors. Each descriptor includes the full logical tensor name, logical path segments, shape, rank, element count, physical storage dtype/format metadata, and the canonical logical visualization dtype.

The UI builds its hierarchical inventory from the returned logical path segments. Filesystem structure is not part of the hierarchy contract.

The current canonical logical visualization dtype is always `float32`. Physical representations such as FP16, BF16, INT8, NF4, or another quantized format remain backend concerns.

## Logical tensor data

`GET /sessions/{session_id}/tensors/{tensor_id}/data` starts a long operation whose binary-stream result kind is `tensor`.

The request is for one complete logical tensor. Transport frames are only delivery chunks; they are not addressable tensor tiles and are not cache units.

The logical payload is IEEE-754 `float32`, little-endian, in deterministic C-contiguous tensor order. Shape remains generic for arbitrary rank even though the proof-of-concept UI directly visualizes rank-1 and rank-2 tensors.

The backend may stream already-compatible physical values with minimal transformation or materialize/dequantize another physical representation into this logical form. That distinction is invisible to the UI.

## Tensor statistics

`GET /sessions/{session_id}/tensors/{tensor_id}/statistics` starts an independent long operation whose result kind is `tensor_statistics`. Statistics are reusable derived artifacts and do not block the tensor-data stream.

Statistics are computed over the finite values of the logical `float32` tensor and expose:

- total element count;
- finite-value count and non-finite-value count;
- minimum and maximum finite value;
- arithmetic mean;
- population standard deviation (`correction = 0`);
- percentiles `p01`, `p05`, `p50`, `p95`, and `p99`.

For sorted finite logical values `x[0..n-1]` and percentile fraction `q`, let
`h = (n - 1) * q`, `i = floor(h)`, and `j = ceil(h)`. The exact percentile
definition is `(1 - (h - i)) * x[i] + (h - i) * x[j]`, for
`q = 0.01, 0.05, 0.50, 0.95, 0.99`. A single finite value yields that value
for every percentile. This specifies a mathematical result, not a requirement
to call a particular PyTorch primitive or accept that primitive's size limits.

Mean and population standard deviation use sufficiently wide accumulation
(at least float64, with stable variance evaluation) to avoid float32 sum,
squared-difference, or range overflow. Derived floating reductions (mean,
stddev, and percentiles) are compared to this definition using
`abs(actual - expected) <= 1e-7 + 1e-6 * abs(expected)`. CPU and CUDA need
not produce bitwise-identical reductions. Minimum/maximum, counts, raw
materialized float32 bytes, and histogram integer counts remain exact;
these tolerances do not authorize changes to the logical tensor values.

The default rendering transfer may use `p01` and `p99` as robust anchors, but that is a UI/rendering decision. The statistics contract itself does not apply a visual normalization.

For empty tensors and tensors containing only NaN/infinities, finite-derived scalar fields are returned as JSON `null` in the stream metadata and `finite_count` is zero.

## Row and column distributions

`GET /sessions/{session_id}/tensors/{tensor_id}/distributions` is defined for rank-2 tensors only. A valid request for another rank returns an HTTP `422` error before streaming begins.

The proof-of-concept distribution artifact uses exactly **100 shared linear bins** over the tensor's complete finite-value range. All row profiles and all column profiles use the same bin domain so hover never changes the histogram coordinate system.

For finite `minimum < maximum`, zero-based bin index is:

`clamp(floor((float64(value) - float64(minimum)) / (float64(maximum) - float64(minimum)) * 100), 0, 99)`.

Perform this index arithmetic in float64, including subtraction, so opposite-sign finite float32 extremes cannot overflow the range. The last bin therefore includes the maximum value. Non-finite values are excluded from histogram counts.

When all finite values are equal (`minimum == maximum`), every finite value is placed in bin index `50`.

The binary logical payload contains two contiguous `uint32` little-endian C-order sections:

- `row_counts` with shape `[rows, 100]`, mapping directly to the right-hand row-distribution surface;
- `column_counts` with shape `[100, columns]`, mapping directly to the bottom column-distribution surface.

With no finite values (including shapes `[0, columns]`, `[rows, 0]`, and
`[0, 0]`), both domain endpoints are `null` and every count is zero. The
section shapes remain exactly as declared above; an empty source can still
have a nonempty all-zero section.

Counts are raw histogram counts. Each bin must fit `uint32` (`0..4294967295`);
accumulate/check without wrapping and reject an unrepresentable result with
`unsupported_size`. Before streaming use HTTP `422`; after streaming starts
use `ERROR_JSON`. Visual intensity normalization of those counts belongs to the UI and must remain stable while the tensor is open.

The stream metadata declares the domain, bin count, shapes, byte offsets, and byte lengths of both sections.

## Tokenization

`POST /sessions/{session_id}/tokenize` runs the real Hugging Face tokenizer associated with the session model. The browser never reimplements tokenizer logic.

The request contains the current text and optionally `add_special_tokens`, which defaults to `true`. The response echoes both values and returns the tokenizer sequence in order.

Each token exposes its numeric ID, tokenizer-native representation, decoded
representation, special-token flag, and source offsets when the tokenizer
provides them. Token `index` values are consecutive from zero in sequence order.
`start` is inclusive and `end` is exclusive, measured in **Unicode code points
into the exact echoed input**, without normalization. They are not UTF-8 byte
offsets, JavaScript UTF-16 code-unit offsets, or grapheme-cluster indices.
Convert native tokenizer offsets to these units when necessary. Both fields
must appear together or both be omitted, with
`0 <= start <= end <= code_point_length(text)` when present.

Preserve overlapping source spans, including multiple byte-level tokens
covering the same code point. Do not fabricate disjoint boundaries. Per-token
`decoded` strings need not concatenate to the input (for example, a token may
decode only part of a multi-byte character). An absent valid span stays absent.

`special` identifies membership in the tokenizer's special-ID set, not whether
the backend inserted that token. A special ID typed in source may still have
a valid source span; a backend-inserted special has none. Both cases retain
the existing fields. The UI must not infer source presence from `special` alone.

For example, `A😀é<special>` has 13 code points: `A` at 0, `😀` at 1,
`e` at 2, combining acute accent at 3, and `<special>` at `[4,13)`.
Two byte-level tokens may both report `[1,2)`. The emoji's UTF-16 span
`[1,3)` is not its API span. `api/fixtures/conformance.json` supplies a
synthetic tokenizer result with these spans, a source-typed special, and an
inserted special; it is an offset example, not a claimed model tokenization.

The UI is responsible for preventing an older asynchronous result from replacing a newer editor state. Echoing the input text provides enough information to perform that check without introducing a separate request-sequence protocol.

## Long operations, sharing, and cancellation

Every request to a streaming endpoint creates a unique consumer-visible `operation_id`, returned in the `X-Operation-Id` response header.

Equivalent requests may be deduplicated internally and share one artifact-producing computation, but that sharing is not represented by a shared public operation ID. Each consumer retains its own operation ID and response stream.

`DELETE /operations/{operation_id}` cancels only that consumer-visible operation and is idempotent. The endpoint returns `204` if the operation is active, already terminal, or no longer retained. Shared underlying work may continue while at least one other consumer still requires it.

Aborting the browser `fetch` also cancels that consumer. Implementations should still use the explicit operation-cancellation path where needed to promptly release queued or backend-side work.

Operations are ephemeral. They are not listed or retained as historical records after completion. Persistent successful results are represented by the artifact cache, not by completed operation resources.

## Errors

Errors detected before a streaming response begins use normal HTTP status codes and the OpenAPI `Error` JSON schema.

After streaming has begun, execution failures use an `ERROR_JSON` terminal frame defined by `binary-streaming.md`; the HTTP status has already been committed.

Cancellation is distinct from failure. When the stream remains connected long enough to observe backend cancellation, it terminates with a `CANCELLED` frame rather than an error frame.

The closed machine-readable error code set is:

| HTTP status before streaming | `code` | Meaning |
| --- | --- | --- |
| 400 | `malformed_json` | Request body is not valid JSON |
| 404 | `model_not_found`, `session_not_found`, `tensor_not_found` | Unknown resource |
| 409 | `model_content_changed` | Local content differs from the session's pinned model fingerprint |
| 422 | `validation_error` | Invalid fields, types, identifiers, or constraints |
| 422 | `unsupported_representation`, `unsupported_rank`, `unsupported_size` | Unsupported logical materialization, operation rank, safe size, or uint32 count |
| 503 | `resource_exhausted` | Insufficient runtime memory/device/other resources |
| 500 | `internal_error` | Unexpected failure |

`openapi.yaml` enumerates applicable responses for each operation. Cancellation
of a syntactically valid but unretained/unknown operation UUID still returns
`204`, not `404`. Malformed identifiers use `422`. Cancellation is not an
error code. An execution failure after HTTP headers uses the same `Error`
object inside `ERROR_JSON`, including when no metadata exists yet.

Error `code` values are for program logic; `message` is human-readable.
Neither messages nor optional details may expose filesystem paths, tracebacks,
or private model-location information. All API responses require
`Cache-Control: no-store`. Allowed cross-origin streaming responses must
expose `X-Operation-Id` via CORS, including streams that terminate before META.

## Cross-field and allocation constraints

All shape dimensions, ranks, lengths, counts, indices, and offsets are
nonnegative safe integers, at most `2^53 - 1`. This bound applies to computed
products and sums too. The LMEX header's narrower uint32 payload-length bound
still applies. Reject unsupported sizes before allocating a destination;
never let floating-point rounding or integer overflow validate a size.
A mathematically valid safe size can still fail a runtime allocation with
`resource_exhausted`.

- Tensor descriptors require `rank == len(shape)` and `numel == product(shape)`.
  The empty product for scalar shape `[]` is 1; any zero dimension gives 0.
  Tensor metadata requires `byte_length == 4 * product(shape)`.
- Statistics require `count == finite_count + non_finite_count`.
  With zero finite values all finite-derived fields, including every percentile,
  are null. Otherwise they are finite numbers, `minimum <= maximum`, stddev is
  nonnegative, and percentiles are ordered within the finite range.
- Distribution sections occur exactly in row/column order, have shapes
  `[rows, 100]` and `[100, columns]`, and byte lengths four times their shape
  products. The row offset is zero; the column offset is the row byte length;
  total byte length is the sum. Domain endpoints are both null or both finite
  and ordered. Null domains require all-zero DATA counts. Summed row counts
  and summed column counts each equal the source finite count.
- Progress requires `completed <= total` when the positive `total` is supplied.

Schemas validate local structure; products, sums, ordering, source-dependent
constraints, and token span bounds must also be checked by implementations.
The shared fixtures include valid and invalid cross-field examples.
