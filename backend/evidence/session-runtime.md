# Shared session/runtime validation

The runtime uses real temporary artifact files with deterministic asyncio/thread
barriers. No real model weights, runtime cache entries, or binary outputs are
committed. Numerical algorithms and binary framing belong to downstream modules.

Local validation (Python 3.12.14, locked dependency versions):

```text
ruff check backend: passed
ruff format --check backend: passed
cd backend
PYTHONPATH=src /tmp/issue-5-venv/bin/mypy: passed (26 source files)
PYTHONPATH=src /tmp/issue-5-venv/bin/pytest -q: 214 passed
```

Local environment setup exhausted available disk space when installing a second
PyTorch copy. Validation instead reused the existing `/tmp/issue-5-venv` dependency
environment with this worktree's `src` on `PYTHONPATH`; no dependency or lockfile
changes were needed. CI retains the normal `uv sync --locked` path and tests Python
3.12 and 3.14 plus wheel installation outside the source tree.

Coverage includes:

- Concurrent session/model isolation, refresh lookup, restart loss, pinned-content
  rejection, inventory mapping, safe errors and strict UUID/body validation.
- Simultaneous equal-key requests invoke one producer; distinct keys remain distinct;
  late consumers replay the prefix and cache hits receive fresh public UUIDs.
- One/all consumers leaving before launch, while producing, during blocked I/O,
  during commit, after commit and after operation removal.
- Cancellation during commit preserves a successfully published artifact.
- A slow cursor holds no full-payload queue and cannot prevent a fast reader from
  consuming a 4 MiB result. Reads are bounded to 256 KiB.
- Disconnects, including repeated task cancellation during a blocked read, retain
  the file until that read settles, then close it and remove the operation.
- Fake CUDA jobs serialize per device, queued cancellation skips launch, other
  devices/CPU progress independently, and active kernels retain their slot until
  completion even when the awaiting task is cancelled.
- Nested materialization/statistics dependencies complete without owning a GPU
  slot while waiting, and parent cancellation releases internal interests.
- HTTP session and operation DELETE cancel only their own consumers. Registries,
  readers and temporary directories are released after terminal response cleanup.

No real CUDA smoke was run. Dependency warnings concern optional NumPy absence
and upstream TestClient deprecations; they do not indicate failed checks.
