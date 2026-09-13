# Binary streaming protocol

## Purpose

Long operations return progressive binary HTTP responses so the UI can consume useful data before the complete result has been produced or transferred.

The same framing is used for tensor transfer, tensor statistics, row/column distributions, and later large or progressive results such as activations and inference intermediates.

The protocol has no public version negotiation. Backend and UI implement the current accepted framing together.

## HTTP contract

The media type is:

`application/vnd.llm-model-explorer.stream`

Every successful streaming response exposes:

`X-Operation-Id: <uuid>`

The backend CORS configuration must expose `X-Operation-Id` to the browser. Implementations should also return `Cache-Control: no-store`; application-level artifact caching is owned by the backend and must not be confused with HTTP intermediary caching.

Before the response is started, ordinary HTTP status codes and JSON errors apply. Once the streaming response is started, terminal error/cancellation state is represented inside the framing described below.

## Byte order

All multi-byte integer fields in frame headers and all binary numeric result payloads are **little-endian**.

Tensor logical values are IEEE-754 `float32`. Distribution counts are unsigned 32-bit integers.

## Frame layout

A stream is a sequence of frames. Every frame starts with the same fixed **12-byte header**:

| Offset | Width | Field | Encoding |
| ---: | ---: | --- | --- |
| 0 | 4 | magic | ASCII `LMEX` (`4c 4d 45 58`) |
| 4 | 1 | type | unsigned 8-bit frame type |
| 5 | 1 | flags | unsigned 8-bit, must be `0` |
| 6 | 2 | reserved | unsigned 16-bit little-endian, must be `0` |
| 8 | 4 | payload length | unsigned 32-bit little-endian byte length |

The header contains no protocol-version field.

A receiver must reject a frame whose magic is not `LMEX`, whose flags/reserved fields are non-zero under the current contract, or whose declared payload cannot be read before the underlying HTTP body terminates.

The maximum payload of one frame is therefore `2^32 - 1` bytes, but implementations should emit much smaller data frames suitable for progressive consumption. Frame size is an implementation tuning parameter and is not an API semantic.

**HTTP/Fetch chunk boundaries are not frame boundaries.** A browser parser must handle a header or payload split across multiple `ReadableStream` chunks and may receive multiple frames in one network chunk.

## Frame types

The current frame type values are:

| Type | Name | Payload |
| ---: | --- | --- |
| `0x01` | `META_JSON` | UTF-8 JSON result metadata |
| `0x02` | `DATA` | Raw bytes of the logical result payload |
| `0x03` | `PROGRESS_JSON` | UTF-8 JSON progress metadata |
| `0x04` | `COMPLETE` | Empty |
| `0x05` | `ERROR_JSON` | UTF-8 JSON API error object |
| `0x06` | `CANCELLED` | Empty |

The first application frame in a successful stream must be `META_JSON`. The backend may delay that first frame while it computes information required by the metadata, but the HTTP response headers — including `X-Operation-Id` — are available before the first body frame.

After `META_JSON`, zero or more `DATA` and `PROGRESS_JSON` frames may appear. Exactly one terminal frame then ends the protocol-level stream: `COMPLETE`, `ERROR_JSON`, or `CANCELLED`. No bytes may follow a terminal frame.

If the HTTP transport itself is interrupted before a terminal frame, the result is incomplete and must not be treated as a successful operation.

## Common metadata

Every `META_JSON` payload is a UTF-8 JSON object with at least:

- `kind`: string identifying the result semantics;
- `byte_length`: total number of logical `DATA` payload bytes expected before successful completion.

`DATA` frames contain consecutive portions of one logical byte payload. They do not carry offsets. The receiver appends each `DATA` payload in arrival order. On `COMPLETE`, the accumulated data byte count must equal `META_JSON.byte_length`.

A receiver must treat a length mismatch as an invalid/incomplete result.

## Tensor result

For `kind: "tensor"`, `META_JSON` contains:

- `tensor_id`;
- `name`;
- `shape`;
- `dtype: "float32"`;
- `byte_order: "little"`;
- `layout: "c"`;
- `byte_length`.

`byte_length` is exactly `numel * 4`.

The concatenated `DATA` bytes are the complete logical tensor in C-contiguous order. Individual `DATA` frame payload lengths must be multiples of 4 so one `float32` element is never split across protocol frames.

The UI may progressively copy/upload complete received values without waiting for the terminal frame, but it must not treat missing regions as populated tensor values.

## Tensor-statistics result

For `kind: "tensor_statistics"`, the result contains no binary `DATA` payload and therefore has `byte_length: 0`.

The first `META_JSON` frame contains the complete statistics object:

- `tensor_id`;
- `count`;
- `finite_count`;
- `non_finite_count`;
- `minimum`;
- `maximum`;
- `mean`;
- `stddev`;
- `percentiles`, containing `p01`, `p05`, `p50`, `p95`, and `p99`;
- `byte_length: 0`.

Finite-derived scalar fields are JSON numbers, or JSON `null` when `finite_count == 0`.

A successful statistics stream therefore consists of `META_JSON`, optionally progress frames emitted before completion only if the metadata has already been produced, and `COMPLETE`.

## Tensor-distributions result

For `kind: "tensor_distributions"`, `META_JSON` contains:

- `tensor_id`;
- `rows`;
- `columns`;
- `bin_count: 100`;
- `binning: "linear-full-range"`;
- `domain_minimum`;
- `domain_maximum`;
- `dtype: "uint32"`;
- `byte_order: "little"`;
- `sections`;
- `byte_length`.

`sections` is an array with exactly these entries in this order:

1. `row_counts`: shape `[rows, 100]`, C-order;
2. `column_counts`: shape `[100, columns]`, C-order.

Each section declares `name`, `shape`, `offset`, and `byte_length`. `row_counts.offset` is `0`; `column_counts.offset` immediately follows the row-count section. Every section byte length is the product of its shape multiplied by 4.

The concatenated `DATA` stream is the concatenation of those two sections. Individual `DATA` frame payload lengths must be multiples of 4.

The binning algorithm and finite/non-finite handling are normative in `contract.md`.

## Progress result

`PROGRESS_JSON` is optional and advisory. Its UTF-8 JSON payload has:

- `completed`: non-negative integer;
- `total`: positive integer when known;
- `unit`: short string such as `bytes`, `elements`, or `rows`.

Progress frames must not change result semantics, offsets, shapes, histogram domains, or any other metadata already declared. Clients may ignore all progress frames without affecting correctness.

## Error frame

`ERROR_JSON` is terminal. Its payload uses the same JSON shape as the OpenAPI `Error` schema:

- `code`: stable machine-readable string;
- `message`: human-readable string;
- optional `details` object.

Bytes already received before an error are incomplete and must not be published by the client as a complete result.

## Cancellation frame

`CANCELLED` is terminal and has zero payload bytes.

A browser that aborts the HTTP request may not receive this frame because it intentionally tears down the transport. The frame is observable when cancellation is initiated separately while the response connection remains open.

Cancellation never implies that another consumer of deduplicated backend work has been cancelled.

## Minimal-copy requirement

The protocol is designed so `DATA` payload bytes can be appended directly into their destination binary allocation and, where browser/WebGL2 APIs permit, uploaded without numeric text parsing or application-owned format conversion.

The contract promises **minimal-copy**, not literal end-to-end zero-copy: browser networking, JavaScript runtime, WebGL2, drivers, and GPU uploads may impose unavoidable copies outside application control.

Metadata and progress are intentionally small JSON frames. Large numeric payloads remain raw binary.
