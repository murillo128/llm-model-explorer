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
