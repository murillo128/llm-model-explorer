# Sessions and execution

## Sessions

The backend supports multiple concurrent sessions. Each session has a stable `session_id` and is permanently associated with one `model_id` for its lifetime.

For the proof of concept, logical session state is held in backend memory. A required `model_revision` in create/get descriptors identifies the exact metadata snapshot captured by session pinning, including composition dependencies. Never attach a later catalogue token to an earlier pinned source. It is an equality hint only; existing content-change guards still reject stale access with 409. Model notifications never rebind a session.

A browser refresh or reconnect can reuse the session while that backend process remains alive. Backend restart may discard sessions. Persistent session recovery is not required.

Runtime state is separate from persistent artifacts. Temporary CPU/GPU buffers, in-flight work, and future inference state belong to the session/runtime layer and may disappear without invalidating cached artifacts.

## Prepared architecture lookup

Architecture preparation is application-scoped startup and demand-driven work, as defined in [backend architecture](architecture.md#blocking-architecture-preparation), not a session inference job. It must not change the session's immutable model binding or require a tokenizer. Establish prepared-result identity using the same content snapshot rules as [models.md](models.md).

A session may read only an architecture for its exact pinned content. Reject detected content changes rather than rebinding it to newer weights. A fresh valid session for content without a prepared result triggers preparation only on its architecture request, awaiting a shared terminal outcome for the exact pinned source. Listing, pinning and independent explorer requests do not wait for that analysis. Deleting a session releases its consumers/modal streams but does not delete shared architecture artifacts. No inference state, paused call stack, KV cache, or execution history is added by this increment.

## Long operations

Potentially expensive user-requested work is represented as an explicit long operation with an `operation_id`. Examples include logical tensor materialization, expensive statistics, and future inference steps.

A long operation starts with a request whose same HTTP response carries its progressive binary result. The `operation_id` is available at the beginning so the UI can track or cancel it while consuming the response. Fast metadata operations and prepared architecture retrieval do not use this numerical-operation mechanism.

Operations are not retained as history after completion. A successful persistent result is an artifact, not a completed operation record. Startup and demand-driven architecture analysis have no client operation ID; shutdown cancellation and artifact ownership apply at application lifetime instead. Cancelling an architecture waiter leaves shared preparation owned by the application until completion or shutdown.

## Cancellation

A consumer may abort its HTTP request and explicitly cancel by `operation_id`. If work has not begun, cancellation prevents scheduling. If GPU work is executing, cancellation takes effect at the next safe boundary; interruption of an already-launched indivisible CUDA kernel is not promised.

Closing architecture weight inspection uses this same consumer cancellation path. It must not cancel another consumer of a shared numeric artifact or remove the prepared graph.

## Shared work

Equivalent operations requested by multiple sessions are deduplicated when they produce the same immutable artifact. Multiple consumers may subscribe to the same production. Consumer cancellation is independent; shared work is cancelled only when no active consumer still needs it.

## Concurrency and device scheduling

Multiple sessions, disk reads, cache hits, tokenization requests, and HTTP streams may proceed concurrently after readiness. Expensive GPU work is serialized through one execution queue per GPU device. CPU and CUDA expose the same logical API result.

Static architecture analysis performs no GPU computation and must not acquire or hold a GPU compute slot. Its startup/demand lifecycle is distinct from future inference pause/resume scheduling, which remains outside this increment.

## Input embedding analysis lifetime

Input-embedding values, statistics and distributions are independent cancellable
consumers. Analysis gathers only the ordered requested float32 rows into one
request-owned CPU buffer using bounded reads. It finishes the input dependency
before acquiring the configured device queue, then reuses the existing native
statistics and histogram kernels. Source checks guard gathering, computation,
metadata and output delivery. CPU and CUDA retain the same mathematical contract.

The temporary source buffer and any uint32 result buffer belong to the consumer.
Cancellation, disconnect, session deletion and failures settle owned blocking work
before closing readers and releasing buffers. Successful completion also releases
them. These operations never publish per-prompt cache artifacts or retain token
sequences as history. They introduce no new sharing or artifact lifecycle. An
auxiliary failure leaves the independent value consumer usable.
