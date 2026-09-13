# Sessions and execution

## Sessions

The backend supports multiple concurrent sessions. Each session has a stable `session_id` and is permanently associated with one `model_id` for its lifetime.

For the proof of concept, logical session state is held in backend memory. A browser refresh or reconnect can reuse the session while the backend process remains alive. Backend restart may discard sessions. Persistent session recovery is not required.

Runtime state is separate from persistent artifacts. Temporary CPU/GPU buffers, in-flight work, and future inference runtime state belong to the session/runtime layer and may disappear without invalidating persistent cached artifacts.

## Long operations

Potentially expensive work is represented as an explicit long operation with an `operation_id`. Examples include logical tensor materialization, expensive statistics, and future inference steps.

A long operation is started by a request whose same HTTP response carries its progressive binary result. The `operation_id` is made available at the beginning so the UI can track or explicitly cancel the operation while consuming the response.

Fast operations such as metadata discovery do not need to use the long-operation mechanism.

Operations are not retained as historical records after completion. Once an operation successfully produces a persistent result, the durable object is the artifact rather than the completed operation.

## Cancellation

Cancellation is part of the operation model. A consumer may abort its HTTP request and may explicitly cancel by `operation_id`.

If work has not begun, cancellation prevents it from being scheduled. If GPU work is already executing, cancellation takes effect at the next safe boundary; the system does not promise interruption of an already-launched indivisible CUDA kernel.

## Shared work

Equivalent operations requested by multiple sessions are deduplicated when they would produce the same immutable artifact. Multiple consumers may subscribe to the same underlying production.

Consumer cancellation is independent. Cancelling one consumer must not cancel shared work while another consumer still needs it. The underlying operation may be cancelled when no active consumer remains.

## Concurrency and device scheduling

Multiple sessions, disk reads, cache hits, tokenization requests, and HTTP streams may proceed concurrently.

For the proof of concept, expensive GPU work is serialized through one execution queue per GPU device. This avoids unnecessary VRAM contention and complex CUDA scheduling while preserving concurrency for non-GPU work.

CPU and CUDA execution must produce the same logical API result.
