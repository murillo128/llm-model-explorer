# Backend foundation

The installable Python service provides configuration, an app factory, lifecycle
ownership, CORS and dependency seams. `GET /models` lists local Hugging Face
checkpoints with lazy safetensors sources. Other product routers are added by
subsequent backend modules. The service does not load full models or serve a
frontend bundle.

## CPU installation and execution

Use Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/getting-started/installation/)
0.12.13 (the version used in CI). From the repository root:

```sh
cd backend
uv sync --locked --python 3.12
uv run --locked llm-model-explorer-backend --help
uv run --locked llm-model-explorer-backend \
  --model-root /absolute/path/to/existing/models \
  --cache-dir /absolute/path/to/separate/cache
```

`uv.lock` pins runtime and development dependencies. The explicit PyTorch CPU
index supplies CPU wheels on Linux/Windows; macOS uses the CPU wheel from PyPI.
This follows [uv's PyTorch index configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).
No NVIDIA driver is required. Model files must already exist locally; this
service does not download them. An empty readable model directory is sufficient
to start this foundation. For a runtime-only install, use `uv sync --locked --no-dev`.

Defaults are `--device cpu --host 127.0.0.1 --port 8000`. Stop with Ctrl+C.
For a separately hosted UI on a trusted network, explicitly configure the bind
address and each allowed browser origin:

```sh
uv run --locked llm-model-explorer-backend \
  --model-root /absolute/path/to/existing/models \
  --cache-dir /absolute/path/to/separate/cache \
  --host 0.0.0.0 --port 8000 \
  --cors-origin http://ui-computer:5173 \
  --cors-origin https://another-ui.example
```

Origins have no path or trailing slash. None are allowed by default. Credentials
are disabled and `X-Operation-Id` is exposed. Invalid paths, overlapping storage
(including resolved symlinks), unwritable cache, invalid ports or unavailable
requested CUDA produce a local diagnostic and nonzero exit. Validation may
create the cache directory and briefly write a disposable probe there. It only
checks the model directory itself and never writes model storage or reads weights.
Keep the configured filesystem paths stable while the service is running.

## Optional CUDA installation

The reproducible default environment is CPU-only. On a CUDA host, use a separate
environment and the matching accelerator index from
[PyTorch's installation instructions](https://pytorch.org/get-started/locally/).
Keep the PyTorch public version equal to the `torch` pin in `pyproject.toml`.
For example, after confirming that the CUDA 13.0 build supports your driver:

```sh
uv sync --locked --no-dev --python 3.12
uv pip install --python .venv/bin/python --reinstall 'torch==2.14.0' \
  --index-url https://download.pytorch.org/whl/cu130
.venv/bin/llm-model-explorer-backend \
  --model-root /absolute/path/to/existing/models \
  --cache-dir /absolute/path/to/separate/cache --device cuda:0
```

The accelerator wheel overrides the CPU lock only in this environment; running
`uv sync` or `uv run --locked` again restores the CPU environment. `cuda` selects
`cuda:0`; `cuda:N` selects that explicit logical device index (after any
`CUDA_VISIBLE_DEVICES` mapping). Unavailable CUDA fails rather than falling back.
Device validation probes driver/device properties but does not allocate tensors.
Real CUDA execution is not part of this foundation's CPU CI validation.

## Repeatable checks

Run inside `backend/` after `uv sync --locked`:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
uv build
```

Tests use temporary directories, synthetic weight sentinels and dependency
doubles. They check settings, filesystem separation/permissions, explicit device
selection, lifecycle cleanup, worker execution, CORS, installed CLI help, and a
real CPU server start/stop. POSIX mode tests are skipped when run as root or on
non-POSIX hosts; write-failure injection also checks filesystem failure reporting.
`.github/workflows/backend-ci.yml` runs these checks on Linux with Python 3.12
and 3.14, and installs the built wheel before testing it outside the source tree.

## Downstream module developer note

- Construct `Settings(...)` once, then call `create_app(settings, routers=[...])`.
  Configuration validation completes before app construction. There is no global
  app instance or import-time runtime initialization.
- Implement domain `APIRouter`s against `docs/spec/api/openapi.yaml`; it remains
  the normative HTTP contract. Generated OpenAPI/docs routes are disabled so an
  incomplete runtime schema does not compete with it.
- `dependencies.py` exposes `get_settings`, `get_blocking_work`, `get_catalogue`,
  `get_sessions`, `get_artifacts`, and `get_operation_delivery`. The catalogue and
  artifact slots provide concrete `ModelCatalogue` and `ArtifactStore` services.
  Sessions and operation delivery begin unset and fail loudly if accessed.
  Replace their `object` annotations as each module lands; no
  speculative business methods or provider interface is prescribed here.
- Supply an async context manager through `service_lifespan` to initialize domain
  resources, yield `Services`, and release resources in `finally`. Compose with
  `open_services(settings)` to reuse the owned worker pool. Tests may supply
  synthetic services or use FastAPI's `app.dependency_overrides`.
- Use `await blocking_work.run(callable, *args, **kwargs)` for blocking I/O or
  PyTorch/native computation. Request context variables propagate into workers.
  The pool is created during lifespan and drained off the event loop at shutdown.
  This seam supplies neither per-GPU serialization nor cancellation/deduplication:
  those belong to the later operation scheduler. Cancelling an awaiting coroutine
  does not stop a running Python thread.
- Use `with TestClient(app)` (or an ASGI lifespan manager) so services exist for
  requests and are released afterward. No services survive an app lifespan.

## Catalogue and pinned source integration

Discovery rules and fingerprint ownership are specified in
[`docs/spec/backend/models.md`](../docs/spec/backend/models.md). `GET /models`
runs discovery in the owned worker pool and returns path-free JSON with
`Cache-Control: no-store`. Invalid candidates appear in local logs. Conflicting
public identities fail the request with `422 validation_error`.

Sessions can retain the returned concrete `ModelSource` for their lifetime:

```python
# Inside an async session handler, using the existing dependency seams:
source = await work.run(catalogue.pin, model_id)
fingerprint = source.fingerprint  # internal artifact-key input; never serialize
descriptors = await work.run(source.tensors)
inventory = {"tensors": [d.model_dump(mode="json") for d in descriptors]}
```

Materialization consumes bounded flat CPU float32 chunks in C order. Execute
the iterator in blocking work, then hand its chunks to the operation's delivery
mechanism. Do not iterate synchronously on the async event loop:

```python
def produce(source, tensor_id, consume_chunk):
    for values in source.iter_tensor(tensor_id, chunk_elements=65536):
        consume_chunk(values)
    # Exhaustion includes a final snapshot check. Only now may a downstream
    # artifact producer mark the complete result publishable.
```

Empty tensors yield no chunks; their descriptors still carry their full shape.
Consumers own independent iterators and may close one without affecting another.
`ModelError.code`, `.status`, and its path-free message are available to future
HTTP/stream adapters; changed pinned sources use `409 model_content_changed`.
The source never repins automatically. `source.check_unchanged(rehash=True)`
also verifies the content digest explicitly.

The backend-only tokenizer seam exposes a guarded local directory. The tokenizer
module must use only snapshot-covered conventional assets, disable network and
repository-provided code, and retain the guard around loading/use in a worker:

```python
def tokenize_locally(source, text):
    # Transformers is a dependency of the tokenizer module, not the catalogue.
    from transformers import AutoTokenizer

    with source.local_directory() as directory:
        tokenizer = AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False
        )
        result = tokenizer(text)
    return result
```

The guard also belongs around use of a cached tokenizer so changed source assets
cannot be served through an old session. This example defines integration only;
session lifecycle, tokenization, binary framing, and artifact persistence are
implemented by their respective modules.

Catalogue tests generate tiny local checkpoints, including shards, scalar/empty
and rank-3 shapes, F16/BF16 conversions, malformed headers, identity collisions,
path escapes, and concurrent readers. They verify bounded header/content reads,
path-independent fingerprints, and mutation detection before/during work.
No real SmolLM2 checkpoint or CUDA smoke is required or claimed by these tests.

## Artifact store seam

`get_artifacts` now supplies the concrete `ArtifactStore` during the default app
lifespan. It uses the configured cache directory, persists across lifespans, and
requires POSIX filesystem primitives. See the owning
[cache specification](../docs/spec/backend/artifact-cache.md) for disk layout,
validation, failure semantics and platform requirements.

A synchronous operation-runtime producer can use the following lifecycle inside
`blocking_work.run(...)`. Every reader owns its cursor and must be closed; the
runtime decides how independent consumers are notified when more bytes arrive.

```python
from llm_model_explorer.artifacts import ArtifactSpec, ArtifactStore


def produce(store: ArtifactStore, model_fingerprint: str) -> None:
    spec = ArtifactSpec(
        model_fingerprint=model_fingerprint,
        source="example.weight",
        operation="materialize",
        parameters={},
        dtype="float32",
        layout="row-major-little-endian",
        shape=(2,),
        expected_bytes=8,
        producer="example-materialize:1",
    )
    cached = store.lookup(spec)
    if cached is not None:
        with cached:
            consume(cached.read_available(8))  # operation-owned transport
        return

    with store.begin_write(spec) as writer, writer.open_reader() as reader:
        writer.append(b"\x00\x00\x80\x3f")  # first synthetic float32
        consume(reader.read_available(4))  # usable before commit
        writer.append(b"\x00\x00\x00\x40")
        writer.commit()
        consume(reader.read_available(4))  # same handle survives publication
        assert reader.complete
```

`consume` above stands for operation-owned delivery. Exceptions or cancellation
must unwind the writer context; completed artifacts survive consumer/session
closure. No sessions, tensor computation, framing or singleflight scheduling are
implemented by this module. Cache tests use small controlled producers and real
temporary directories, including concurrent process publication, injected disk
failures, corruption, crash leftovers, and an 8 MiB stream whose traced Python
allocation peak stays below 1 MiB.
