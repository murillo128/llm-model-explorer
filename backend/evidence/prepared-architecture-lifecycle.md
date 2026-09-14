# Prepared architecture lifecycle validation

This evidence covers application composition and prepared HTTP retrieval with
small local fixtures. It does not claim full-reference checkpoint acceptance,
GPU execution, or numerical inference equivalence.

## Implementation boundary

`ArchitectureService` registers the reviewed descriptions, discovers and pins the
startup catalogue, and prepares models sequentially before `open_services` yields.
Only input identity and binding context remain resident; successful graph bytes
live in the existing structured artifact store. `DescriptionRegistry.select`
allows precomputable cache lookup without graph construction.

The HTTP adapter guards the session's pinned content before and after lookup,
checks that the session still exists after blocking work, and serves the graph
with an exactly budgeted model envelope. Unavailable results remain startup state.
A cache miss after readiness never invokes analysis. The CLI forwards SIGTERM and
SIGINT to the startup stop flag before cancelling the lifespan task, closing the
race between signal delivery and the worker's publication check.

## Reproduction

From `backend/`, use the locked development environment:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
uv run --locked pytest tests/test_architecture_service.py
```

From the repository root, with the backend environment on `PYTHONPATH`/installed:

```sh
backend/.venv/bin/python -m pytest acceptance -q
api/.venv/bin/python api/validate_contract.py
```

Local validation used Python 3.12, PyTorch 2.14.0+cpu, FastAPI 0.141.1, and Uvicorn
0.52.4 in an existing environment matching the locked backend dependencies. Source
was selected explicitly with `PYTHONPATH`; the shared environment was not modified.

## Observed coverage

Backend lint, formatting, and strict types pass. The full backend suite passes:
**966 passed, 19 skipped** (CUDA unavailable), with two upstream TestClient
deprecation warnings. The final timing-only adjustment also passed the dedicated
lifecycle suite.

- All 25 dedicated lifecycle tests pass. A TCP test blocks the second model with
  a barrier and confirms that the service cannot accept a request before every
  candidate has finished. Four registered family fixtures produce complete
  graphs, including quantized metadata and a valid no-tokenizer V-JEPA 2 session.
- Warm startup and subsequent GETs succeed with graph construction replaced by
  a failing spy. Cold/warm response limits reserve the exact HTTP envelope. Logs
  distinguish fingerprint hashing, analysis, cache reads, and total preparation.
- Unsupported descriptions, partial graphs, analysis failures, and oversize
  results are terminal per-model outcomes; later candidates still complete.
  Entry-local storage failures are localized while a failed global storage probe
  aborts startup. Ambiguous catalogue identity also prevents readiness.
- Invalid/unknown/deleted sessions, changed pinned content, newly discovered
  content, source changes during GET/publication, deleted/corrupt/full-cache loss,
  request I/O failures, and memory exhaustion exercise their typed HTTP outcomes.
- Cancellation barriers cover analysis and publication. A separate CLI process
  receives actual SIGTERM while holding an unpublished graph, then exits without
  readiness or a surviving temporary/published graph. Workers settle before
  service teardown.
- Spies reject tensor materialization, PyTorch allocation/model calls, GPU queue
  acquisition, and network access during preparation. The tests use local files
  and never construct a Transformers model or load checkpoint Python code.
- Real HTTP acceptance passes: **16 passed, 2 skipped** (optional CPU/CUDA reference
  checks). Numeric acceptance observers now count format-1 artifacts separately
  from startup format-2 graphs, preserving numerical publication/cancellation
  assertions. Existing backend numeric tests use the same distinction.
- API validation passes: OpenAPI 3.1, 283 references, 88 instance cases, 112
  architecture cases, and 76 reproducible wire fixtures. Three actual HTTP
  responses (dense, Qwen3.5, V-JEPA 2) additionally passed the published
  `ArchitectureResponse` JSON Schema and independent semantic validator.

Full selected checkpoints, their cold/warm performance measurements, and GPU
capability acceptance remain the integrated-reference acceptance work. Cancellation
settles the current guarded hashing/analysis/cache phase at its safe boundary;
it does not interrupt a Python thread or an in-progress filesystem operation.
