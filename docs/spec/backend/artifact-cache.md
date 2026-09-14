# Artifact cache

## Purpose

The artifact cache stores complete, immutable, reconstructible backend results. It is shared across sessions so equivalent work is not duplicated. It is neither model storage nor session storage; original files remain authoritative and are not copied merely to create another authoritative copy.

## Storage

Use the local filesystem only, without a database, Redis, or external storage service. Each artifact key derives deterministically from all inputs that determine its content, including model fingerprint, operation, parameters, representation, and producer identity. Store payload bytes and a small validating manifest. Numeric tensors and structured architecture graphs are different artifact kinds, not different caches.

## Complete artifacts only

Chunks are transport units, not cache units. Only complete results become valid entries. Producers may write temporary payloads while streaming flushed bytes to consumers; publish the completed result atomically after successful validation. Failed/cancelled generation must not leave a valid partial entry.

A semantically partial architecture can be a completely constructed valid graph with explicit coverage diagnostics. It is cacheable under its own semantic identity. An interrupted graph is not such a result and must not be published.

## Deduplication and validity

Equivalent requests/sessions resolve to the same immutable artifact. Every model-derived artifact includes the checkpoint content fingerprint; content changes prevent reuse. Old results need not be deleted automatically. UI position, colors, zoom, expansion, selection, and session IDs do not affect graph or numeric content identity.

## Lifecycle

The cache remains disposable, with no automatic expiration, LRU, size limit, garbage collector, or management UI. With the backend stopped, deleting the entire configured cache must remain safe. Restart reconstructs required storage and prepared architectures. No session-persistence system is introduced.

## Concrete numeric store and disk layout

`backend/src/llm_model_explorer/artifacts.py` owns `ArtifactStore`. Model and cache directories must not overlap, including through symlinks. The concrete store uses a POSIX filesystem with advisory `flock`, directory descriptors, no-follow opens, and atomic same-filesystem rename. Configured paths stay stable during operation.

```text
<cache-dir>/
  <sha256-of-canonical-spec>/
    manifest.json
    payload.bin
  .tmp-<unique-writer>/
    payload.bin
    manifest.json
```

The current numeric `ArtifactSpec` snapshots sorted, compact, finite JSON and hashes it with SHA-256. Inputs include model fingerprint, source, operation, parameters, dtype, layout, shape, expected byte length, and producer identity. Preserve existing numeric key/manifest/byte semantics when adding structured results; do not mass-invalidate working numeric artifacts solely to implement graph caching.

Numeric manifests use format `1`, digest key, full specification, fixed `payload.bin`, exact length, and incrementally computed payload SHA-256. Existing manifest reads are bounded to 64 KiB and specifications to 32 KiB. Lookup validates structure, key, specification, representation, fixed payload name, regular-file status, and actual length. It does not hash the whole numeric payload on a hit. Its digest detects conflicting producer output, not same-length external edits under trusted storage assumptions. Malformed, missing, symlinked, and truncated entries are misses; permission/I/O failures propagate rather than authorizing deletion.

## Writer and reader lifecycle

`begin_write(spec)` creates a private temporary directory. `append(bytes)` retains no prior chunks and advances the available prefix only after flushing the full append. Expected length and available length are separate. `writer.open_reader()` gives an independent handle/cursor; bounded reads expose only flushed bytes. Empty live reads can mean no new bytes yet; consumers inspect `complete` and `aborted`. Wakeups, cancellation, and singleflight belong to the operation runtime.

Numeric `commit()` requires expected and actual byte counts to agree, syncs payload and manifest, and atomically publishes. A short advisory directory lock serializes publication, not generation or computation. An existing valid entry is reused only when payload digests agree; conflicts raise `ArtifactConflict`. Readers already attached to private files retain independent lifetimes through publication. `lookup` returns an independently owned reader or none.

`abort()` is idempotent, removes the private directory, and never reports incomplete bytes as successful output. Context exit aborts uncommitted writers. External truncation raises, rather than yielding a valid short result. Close readers and settle owned blocking work before releasing writers. Abandoned `.tmp-*` directories at startup are neither hits nor resumable results and may belong to another process; do not opportunistically delete them. A post-rename directory-sync failure may leave a complete published artifact with uncertain power-loss durability; no failure replaces an older valid result.

## Structured architecture artifacts

Extend this store locally to support a distinct `architecture_graph` kind with media type `application/json`. Persist deterministic finite UTF-8 JSON containing only the graph document; session/public model envelope and UI state are not cached. Reuse directory safety, independent readers, atomic publication, conflict handling, and disposal. Do not pretend JSON bytes are a float32 tensor, create a second independent cache, or require a fixed numeric shape.

Compute a domain-separated key *before graph generation* from model fingerprint, selected description ID/revision, analysis options affecting meaning, analyzer revision, and graph-schema/serializer revision. Exclude output length/digest from this precomputable input identity. Record actual byte length and payload digest in the completed structured manifest. Warm lookup must not rebuild the graph merely to learn its expected length. The graph's public opaque identity can derive from this analysis key without exposing the raw checkpoint fingerprint.

Keep legacy numeric manifests readable. A distinct structured manifest discriminator/format is permitted and must be validated independently; malformed or incompatible structured entries are misses during startup. Bound structured manifest/spec reads as above and graph payloads by the [API graph bound](../api/architecture-explorer.md#validation-limits-and-publication). On structured lookup, validate payload length/digest, JSON schema, identity, and reference closure before use. A compatible complete or partial graph is reusable; failure/unavailable diagnostics are startup state, not successful graph artifacts.

At publication, validate the full graph, size, byte count, final source snapshot, and digest before atomic visibility. Same-key divergent graphs are conflicts, not last-writer wins. Cancellation, failure, or source mutation never publishes a usable result. The preparer can continue with a localized model failure when storage remains usable; globally unusable storage remains a startup error.

After readiness, missing/corrupt/obsolete structured results are reported through the architecture capability state or pinned-content error, never silently regenerated by a read request. Changes to descriptions/schema invalidate graphs without changing the checkpoint. Renderer/layout-library changes alone do not, because layout is not cached semantic content.
