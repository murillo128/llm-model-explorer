# LMEX writer and HTTP adapter evidence

The shared writer/adapter is exercised by `tests/test_lmex.py` and
`tests/test_streaming.py`; the private fixture routes are registered only in tests.

- Every successful/error/cancelled shared wire fixture is reproduced byte for
  byte, including float32 special values, uint32 histogram counts, scalar and
  empty results, statistics, progress and terminal-first outcomes. All applicable
  shared metadata/progress/error schema fixtures are checked independently by the
  Python models. An additional non-ASCII tensor name checks literal UTF-8 output.
- Writer tests reject duplicate/missing metadata, post-terminal emission,
  misalignment, oversized frames, non-finite JSON, under/overruns and statistics
  DATA (including empty DATA). DATA memoryviews retain their original buffer.
- `test_socket_progress_two_consumers_delete_and_cache_hit` runs Uvicorn on a real
  loopback TCP socket. HTTPX observes operation headers while metadata is gated,
  then reads a complete first DATA frame while the producer is still held at an
  explicit event barrier and the artifact is not published. A second consumer
  receives the same prefix; DELETE cancels the first with CANCELLED while the
  second finishes successfully. A later cache hit delivers the same bytes without
  invoking the producer again.
- Real socket tests cover pre-header structured 422 errors, pre-META error and
  explicit cancellation, mid-DATA producer failure and socket disconnect. Each
  asserts reader closure and removal of operations, consumers and flights.
- Injected disk-read and publication failures, short/overlong and unaligned
  results never emit COMPLETE. Send failures at every response boundary stop
  immediately without appending a frame. Task cancellation also closes a pending
  metadata coroutine and releases runtime resources.
- A deterministic blocked ASGI sender leaves a real HTTP metadata request
  responsive. Delivery of a 1 MiB artifact uses buffers of at most 4096 bytes in
  this test, retains no response body, and reproduces a checksum across producer
  appends split inside individual values. Send failure releases only its consumer;
  a second consumer still receives the complete artifact.

These tests establish reusable transport behavior; they do not claim that product
numerical endpoints or a complete tensor feature have been integrated.

## Local validation

Python 3.12 with the repository-pinned Ruff 0.16.7, mypy 2.3.1 and pytest 9.1.1:

- Ruff check and format check pass.
- Strict mypy passes for all 31 source/test files (`--cache-dir=/dev/null`).
- Full source suite: **316 passed**.
- Wheel built with the declared Hatchling 1.32.0 build backend using
  `python3 -m pip wheel ./backend --no-deps --wheel-dir backend/dist`.
- Full installed-wheel suite, run from `/tmp`: **316 passed**.

The host filesystem was nearly full. Concurrent full-suite runs hit ENOSPC;
serial reruns after removing this task's temporary artifacts passed. The existing
local pinned environment was used directly because installing the uv executable
also hit that space limit. Warnings were the existing optional NumPy absence and
Starlette/AnyIO deprecations. No dependency or lockfile changes were needed.
