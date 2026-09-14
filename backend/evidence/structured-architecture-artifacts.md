# Structured architecture artifact evidence

The store adds `ArchitectureArtifactSpec`, `begin_graph_write`, and
`lookup_graph` in `artifacts.py`. A structured format-2 manifest records
`architecture_graph` / `application/json` in its specification, plus completed
length and SHA-256. It shares numeric publication locks, temporary-directory
ownership, conflict detection, independent readers, and disposal.

The input key snapshots fingerprint, selected description/reviewed source
revisions, scope, meaningful analysis options, analyzer/schema revisions, and
serializer revision. It needs no graph construction or expected output length.
Empty options preserve the graph core's `Producer.graph_id`; nonempty options
produce `spec.graph_id`, which construction must set on `GraphBuilder` before
allocating record IDs. Public model/session envelopes and renderer state are
not cache content. The caller supplies the already admitted `BindingContext`
and a final `check_source` callback that raises on pinned-source mutation or
cancellation; hashing, analysis selection, and preparation ordering remain
outside the store.

Both lookup and publication validate bounded UTF-8 bytes, actual length,
SHA-256, deterministic shared serialization, graph/schema identity, reference
closure, and inventory bindings through the shared graph validator. Only a
fully serialized graph can commit, including explicit partial coverage.
Unavailable results cannot commit. Warm lookup never runs analysis or repairs a
missing entry. Caller-owned readers must be closed.

## Reproduction

From `backend`, using the locked development environment:

```sh
uv run --locked pytest tests/test_architecture_artifacts.py tests/test_artifacts.py
uv run --locked pytest -s tests/test_architecture_artifacts.py::test_cold_warm_lookup_does_not_construct_graph
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
```

Local checks use Python 3.12 and the existing `/tmp/issue-43-venv` environment
with `PYTHONPATH=src` (pytest 9.1.1, Ruff 0.16.7, mypy 2.3.1, PyTorch
2.14.0+cpu). The focused tests cover:

- Cold publication, restart/warm lookup, source relocation with changed public
  directory-fallback identity, and complete cache deletion/reconstruction.
- Complete versus partial coverage; truncated serialization, cancellation,
  source mutation, and schema/reference/identity/inventory corruption, including
  corruption with a recomputed manifest digest.
- Exact byte bounds, oversized sparse files rejected before reading, oversized
  append, malformed/noncanonical/non-UTF-8 JSON, excessive nesting, manifest
  bounds, symlinks, FIFO/nonregular files, and private payload replacement.
- Permission/I/O/space errors, immutable same-key conflicts, parallel writers
  with independent live and published readers, owned temporary cleanup, and
  preservation of another writer's live and abandoned temporary directories.
- Fixed legacy numeric canonical specification/key and exact format-1 manifest,
  plus the existing numeric streaming, cancellation, publication, and
  cross-process conflict tests.

An illustrative synthetic-oracle run measured 7.9 ms for cold graph
construction/publication and 1.5 ms mean per warm validation over 20 hits.
Instrumentation required 20 validator calls and zero analyzer calls. This is
operation-count evidence and a local timing sample, not a large-model startup
benchmark or a timing threshold. Graph bytes are bounded at 32 MiB; the response
owner must account separately for the public response envelope's byte budget.

CUDA is not required by the store; local full-backend validation skips the
existing CUDA-only cases when no CUDA device is available. No model files,
cache payloads, or bulky generated evidence are committed.
