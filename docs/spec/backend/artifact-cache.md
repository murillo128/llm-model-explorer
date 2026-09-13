# Artifact cache

## Purpose

The artifact cache stores complete, immutable, reconstructible results produced by the backend. It is shared globally across sessions so identical work does not consume duplicate disk space or require duplicate computation.

The cache is not model storage and is not session storage. Original model files remain the source of truth and are never copied merely to create another authoritative copy.

## Storage

The proof of concept uses the local filesystem only. No database, Redis service, or external storage system is required.

Each artifact is identified by a deterministic key derived from the inputs that determine its content. Relevant inputs include the model fingerprint, operation, parameters, logical representation, and any other state required to reproduce the same result.

An artifact stores its binary payload plus enough filesystem metadata, such as a small manifest, to identify and validate it.

## Complete artifacts only

Chunks are transport units, not cache units. The persistent cache contains only complete artifacts.

While an artifact is being generated, the backend may write to a temporary file while simultaneously streaming produced bytes to consumers. The artifact becomes visible as a valid cache entry only after generation has completed successfully, preferably by atomically publishing or renaming the completed temporary output.

Cancelled or failed generation must not leave a valid partial cache entry. Incomplete temporary files may be discarded.

## Deduplication and validity

Different sessions requesting the same artifact resolve to the same cache key and reuse the same persisted bytes.

Every model-derived artifact is tied to the model content fingerprint. Replacing or modifying model content therefore results in different artifact keys and prevents stale results from being reused.

Old artifacts do not need to be deleted automatically.

## Lifecycle

The cache is disposable. The proof of concept has no automatic expiration, LRU policy, size limit, garbage collector, or cache-management UI.

It must be safe to stop the backend, delete the entire configured cache directory, and restart with an empty cache. The backend recreates any required cache structure on demand.

## Concrete store and disk layout

`backend/src/llm_model_explorer/artifacts.py` implements the store. The application
constructs one `ArtifactStore` from its configured cache directory and model root;
these directories must not overlap, including after resolving symlinks. The
concrete implementation currently requires a POSIX filesystem with advisory
`flock`, directory descriptors, no-follow opens, and atomic same-filesystem rename.
Configured paths must remain stable while the backend runs.

```text
<cache-dir>/
  <sha256-of-canonical-spec>/
    manifest.json
    payload.bin
  .tmp-<unique-writer>/
    payload.bin
    manifest.json       # written only when committing
```

Each artifact has one complete binary payload. Multiple representations or derived
results use independent artifact specifications. `ArtifactSpec` snapshots a JSON
specification into sorted, compact, finite JSON and hashes its UTF-8 bytes with
SHA-256. Key inputs are the caller-supplied model content fingerprint, source
identity, operation kind, computation parameters, output dtype, layout, shape,
expected byte length, and producer/algorithm identity. The producer identity is
internal invalidation metadata. Callers must exclude local paths, session IDs,
timestamps, and UI color/luminosity controls from computation parameters.

The manifest contains format `1`, the digest key, the complete specification,
fixed payload name `payload.bin`, exact byte length, and an incrementally computed
payload SHA-256. Manifest reads are bounded to 64 KiB; specifications are limited
to 32 KiB. Lookup checks the manifest structure, key, exact specification and
representation, fixed payload name, regular-file status and actual byte length.
It does not hash the whole payload on a cache hit. The payload digest detects
conflicting concurrent producer output; it is not a claim to detect same-length
external payload edits during lookup. The cache assumes trusted local storage.
Malformed, missing, symlinked or truncated entries are misses; permission and I/O
errors propagate rather than authorizing deletion of a potentially valid entry.

## Writer and reader lifecycle

`begin_write(spec)` creates a private temporary directory. `append(bytes)` writes
without retaining previous chunks and advances the available prefix only after
the complete append is flushed. Expected length and available length are separate.
`writer.open_reader()` opens an independent file handle and cursor; its
`read_available(max_bytes)` returns only flushed bytes, with an explicit positive
read bound. An empty live read means no new bytes yet. Consumers inspect
`complete` and `aborted` for terminal state; wakeups, cancellation routing and
singleflight ownership belong to the operation runtime.

`commit()` requires the expected byte count and actual file size to agree, syncs
the payload and completed manifest, and publishes the directory atomically. A
short advisory lock on the cache directory serializes publication across threads
and processes, including removal of invalid entries. It does not lock generation,
lookup or streaming, and is not an operation scheduler. An existing valid entry
is reused without mutation when its payload digest agrees; conflicting output
raises `ArtifactConflict`. Readers already attached to the private file keep
their handles through publication or reuse and release them with `close()` or a
context manager. `lookup(spec)` returns an independently owned reader or `None`.

`abort()` is idempotent and unlinks the writer's temporary directory. Writer
context exit aborts any uncommitted work, including exceptional or cancelled
production. Attached readers can consume the previously available prefix after
abort, but observe `aborted`, never successful completion. A short read caused by
external truncation raises rather than silently returning partial cached data.
The runtime must close every reader and let running blocking work finish before
releasing its owning writer context.

Startup ignores abandoned `.tmp-*` directories: they are never hits or resumable
work, and another process may still own them. Deleting the entire cache while the
backend is stopped removes these along with completed entries. Publication errors
before rename leave no completed entry; failure of the final parent-directory
sync may leave a fully published entry, whose persistence across power loss is
uncertain. Every visible completed entry still has a complete payload and
manifest. No failure path replaces an older valid entry.
