# Browser API boundary

Construct `ApiClient` with the `RuntimeConfig` returned by `loadRuntimeConfig`.
Its explicit methods cover the current OpenAPI endpoints. Paths preserve a
configured backend URL prefix and encode each resource identifier separately.
JSON responses and stream controls validate against generated OpenAPI schemas;
`validation.ts` additionally enforces the contract's cross-field constraints.
No React state, renderer, model math, or browser artifact cache lives here.

`npm ci` installs both the UI and the private generator workspace. Run
`npm run api:generate` after changing the accepted OpenAPI document, and commit
both files in `generated/`. `npm run api:check` compares regenerated content
without writing files and is part of `npm run check`. The generator's isolated
TypeScript 5 dependency satisfies openapi-typescript's peer contract while the
application continues using TypeScript 6. Default-valued optional request
properties remain optional; generation does not alter endpoint semantics.

## Progressive consumption

`streamTensor`, `streamTensorStatistics`, `streamTensorDistributions`, and
`streamInputEmbeddings`, `streamInputEmbeddingsStatistics`, and
`streamInputEmbeddingsDistributions` return
a `StreamOperation` immediately. `onOperationId` fires when headers arrive,
before any body bytes are required. `onMetadata` supplies validated dimensions
and sizes; `onData(bytes, byteOffset)` delivers contiguous raw little-endian
byte segments synchronously. Copy them directly into the chosen destination
or progressive upload path. Do not retain whole network chunks for later
concatenation. The decoder itself allocates no destination.

Network segments can split a four-byte value, and their buffer offsets need
not be aligned. Track populated bytes and expose/upload only complete elements.
Use `DataView.getFloat32(offset, true)` for float32 and
`DataView.getUint32(offset, true)` for uint32 when reading individual values;
a float32 conversion loses precision for large distribution counts. No numeric
conversion is needed when copying bytes into storage. Source-dependent
statistics and distribution correctness still belong to the backend; the
client validates layout, null-domain zero counts and matching section totals.

All callbacks describe provisional content. `operation.done` resolves with
`complete` only after COMPLETE, exact byte count and clean HTTP EOF. A terminal
frame followed by any byte is a protocol failure. Failed or cancelled content
must be invalidated by the consumer, including any provisional rendering.
Statistics can complete without DATA, and ERROR/CANCELLED can precede metadata.

`done` resolves to `{ kind: 'complete', metadata, byteLength }`,
`{ kind: 'backend', error }`, or `{ kind: 'cancelled' }`.
It rejects with `ApiFailure` for `http` (structured error and status), `protocol`
(invalid response), or `transport` (network interruption or incomplete EOF).
Ordinary JSON methods reject with these failures or `cancelled` on abort.

## Lifecycle and cancellation

Pass an `AbortSignal` for owner lifetime cancellation. Aborting stops fetch/body
consumption. Use `operation.cancel()` when explicit cleanup of a known backend
operation is needed: it aborts consumption and sends one idempotent DELETE with
an independent request signal. Repeated calls share that cleanup promise.
Await/catch it separately to observe a cleanup HTTP/network failure; `done`
still reports cancellation. Cancelling after a settled operation is a no-op.
Before headers there is no public ID to delete. Every request owns its own
operation handle, so cancelling one does not touch another.

Readers are cancelled and unlocked and external abort listeners removed on
all termination paths. A component may call `cancel()` during unmount and
handle its promise, or pass its lifetime signal; it must also handle `done`.
The UI and backend can have different origins. The backend must allow the UI
origin and DELETE preflights and expose `X-Operation-Id` to browser JavaScript.

## Verification

The decoder tests consume every shared wire fixture at every two-part split,
in one-byte chunks, coalesced, Fibonacci chunks and five seeded random chunkings.
They also test split UTF-8, an unaligned source, 2 MiB DATA with borrowed 64 KiB
segments, integer precision and malformed distribution counts. Shared schema
fixtures cover safe allocation arithmetic and token code-point spans.
The transport unit tests and `tests/api-transport.spec.ts` exercise the actual
adapter; the latter uses browser fetch against a separately bound HTTP server
with controlled headers/body timing, CORS and DELETE preflight handling.


`streamInputEmbeddings(sessionId, { token_ids }, options)` uses the generated
`InputEmbeddingsRequest` type/schema and POSTs through the same LMEX transport.
It snapshots ordered IDs and rejects an incorrect echo before exposing META or
DATA, including reordering, changed duplicates, and length mismatches. The
component never fetches or parses an embedding response itself.

The two embedding analysis methods accept the same ordered request and independently
return statistics or binary row/column counts. Each validates its distinct result
kind and exact ordered echo. Statistics include the requested matrix shape and
count; distributions include its row/column geometry. Keep the three operation
handles separate so auxiliary loading/failure does not gate valid matrix bytes.
Application-level Tokenizer composition is a separate integration step.
