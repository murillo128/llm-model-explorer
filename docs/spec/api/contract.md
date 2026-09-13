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

`POST /sessions` binds a new session permanently to one public `model_id`. A session does not change model during its lifetime. `GET /sessions/{session_id}` exists so a refreshed UI can reconnect while the in-memory backend session still exists. `DELETE /sessions/{session_id}` releases the session's ephemeral runtime state; it does not delete shared persistent artifacts.

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

Percentiles use linear interpolation equivalent to `torch.quantile(..., interpolation="linear")` over the finite logical values.

The default rendering transfer may use `p01` and `p99` as robust anchors, but that is a UI/rendering decision. The statistics contract itself does not apply a visual normalization.

If a tensor contains no finite values, finite-derived scalar fields are returned as JSON `null` in the stream metadata and `finite_count` is zero.

## Row and column distributions

`GET /sessions/{session_id}/tensors/{tensor_id}/distributions` is defined for rank-2 tensors only. A valid request for another rank returns an HTTP `422` error before streaming begins.

The proof-of-concept distribution artifact uses exactly **100 shared linear bins** over the tensor's complete finite-value range. All row profiles and all column profiles use the same bin domain so hover never changes the histogram coordinate system.

For finite `minimum < maximum`, zero-based bin index is:

`min(floor((value - minimum) / (maximum - minimum) * 100), 99)`.

The last bin therefore includes the maximum value. Non-finite values are excluded from histogram counts.

When all finite values are equal (`minimum == maximum`), every finite value is placed in bin index `50`.

The binary logical payload contains two contiguous `uint32` little-endian C-order sections:

- `row_counts` with shape `[rows, 100]`, mapping directly to the right-hand row-distribution surface;
- `column_counts` with shape `[100, columns]`, mapping directly to the bottom column-distribution surface.

Counts are raw histogram counts. Visual intensity normalization of those counts belongs to the UI and must remain stable while the tensor is open.

The stream metadata declares the domain, bin count, shapes, byte offsets, and byte lengths of both sections.

## Tokenization

`POST /sessions/{session_id}/tokenize` runs the real Hugging Face tokenizer associated with the session model. The browser never reimplements tokenizer logic.

The request contains the current text and optionally `add_special_tokens`, which defaults to `true`. The response echoes both values and returns the tokenizer sequence in order.

Each token exposes its numeric ID, tokenizer-native representation, decoded representation, special-token flag, and source offsets when the tokenizer provides them. `start` is inclusive and `end` is exclusive. Offset fields are omitted when no valid source span is available.

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

The proof-of-concept uses these status-code semantics:

- `404`: requested model, in-memory session, or tensor does not exist;
- `422`: the request is structurally valid but unsupported for the target resource, such as requesting rank-2 distributions for another rank;
- `5xx`: unexpected backend failure before the stream begins.

Error `code` values are stable machine-readable strings; `message` is human-readable and must not be used for program logic.
