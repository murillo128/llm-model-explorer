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
  Sessions and operation delivery are app-owned concrete services (see below).
- Supply an async context manager through `service_lifespan` to initialize domain
  resources, yield `Services`, and release resources in `finally`. Compose with
  `open_services(settings)` to reuse the owned worker pool. Tests may supply
  synthetic services or use FastAPI's `app.dependency_overrides`.
- Use `await blocking_work.run(callable, *args, **kwargs)` for blocking I/O or
  PyTorch/native computation. Request context variables propagate into workers.
  The pool is created during lifespan and drained off the event loop at shutdown.
  This seam supplies neither per-GPU serialization nor cancellation/deduplication:
  those belong to `OperationRuntime`. Cancelling an awaiting coroutine
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

## Session and shared-operation runtime

`session_routes.py` implements session create/get/delete, tensor inventory and
idempotent consumer-operation cancellation. `SessionRegistry` pins a `ModelSource`
on creation and checks the snapshot on subsequent reads; its public descriptor
contains only UUID `id` and logical `model_id`. The registry disappears at shutdown.
Deleting a session releases its consumers and preserves other sessions and artifacts.

`operations.py` coordinates coroutine producers over `ArtifactStore`. Construct an
`ArtifactSpec` with the session source fingerprint and the complete computation
identity. Use `SessionRegistry.subscribe(session_id, spec, producer)` for external
requests; it validates the session snapshot and prevents deletion/registration races.
Each call returns a distinct consumer UUID, including cache hits. The runtime's
lower-level `subscribe` is for callers that already own session validation.

A producer receives `ProducerContext` and must use these boundaries:

- `await ctx.io(callable, ...)`: blocking CPU/file work in the app executor.
- `await ctx.compute(callable, ...)`: blocking compute on the configured device,
  serialized per CUDA device. A CUDA callback must synchronize launched work before
  returning; no Python-level preemption of a running kernel is promised.
- `await ctx.append(chunk)`: flush a bounded bytes chunk to the private spool and
  wake readers. The runtime alone commits after successful producer return.
- `ctx.cancellation.check()` or `await ctx.cancellation.wait(event)`: cooperative
  safe boundaries for synchronous work or asynchronous barriers.
- `async with ctx.dependency(spec, producer) as consumer`: hold an internal,
  non-public interest in another artifact. Resolve dependencies **before** entering
  a compute callback. The materialization/statistics dependency must be acyclic;
  this runtime deliberately does not discover or schedule arbitrary workflows.

Producers must not block the event loop, spawn unowned work, or retain all emitted
chunks. Returning means all blocking work and source validation have finished.
Cancellation stops subsequent steps and queued jobs; active blocking work settles
before its writer is aborted or its device slot released. If commit has already
started, a successful publication survives cancellation.
Compute callbacks recheck cancellation inside the worker immediately before launch,
including when they waited in the shared thread pool after acquiring a device slot.

A future binary response adapter should acquire its consumer before sending
headers, put its UUID in `X-Operation-Id`, and own its entire response lifetime:

```python
consumer = await sessions.subscribe(session_id, artifact_spec, producer)
try:
    while chunk := await consumer.read():
        await send_data_frame(chunk)
    await send_end_frame()
except OperationCancelled:
    await send_cancelled_frame_if_connected()
except Exception as error:
    await send_safe_error_frame_if_connected(error)
finally:
    await consumer.aclose()
```

This is adapter pseudocode, not an additional HTTP endpoint or binary encoder.
The adapter must map exceptions to the accepted path-free error contract and emit
exactly one terminal frame. `read()` returns at most 256 KiB; callers read serially
and await delivery before asking for another chunk. Empty bytes mean successful
EOF; cancellation and failure raise distinct exceptions. Disconnection cancels and
drains an in-flight read before closing its file. Always `aclose()` when the
response ends, even if headers or the first frame fail; no operation history is kept.

Registration is atomic on the owning event loop. Readers replay from byte zero,
then follow the flushed prefix using independent file cursors. Notifications carry
no payload buffers. Slow network delivery holds neither the registry nor a GPU
slot. An abandoned producer stays registered until its blocking work settles;
new requests for that key wait for retirement before retrying. Lifespan shutdown
cancels consumers, drains producers, and then shuts down the blocking executor.

The deterministic tests in `test_operations.py` and `test_session_operations.py`
cover singleflight, late replay, cache hits, independent cancellation, blocked I/O,
commit races, slow readers, file/registry cleanup, fake per-device scheduling and
nested dependencies. Real CUDA smoke is optional and is not implied by these tests.

## Common LMEX delivery

`lmex.LMEXWriter` implements the accepted 12-byte framing and validates the
metadata variants, progress and error objects against the current contract.
It returns separate header/payload buffers; DATA uses memoryviews, is four-byte
aligned and bounded to 256 KiB by default. It rejects length overruns, incomplete
success, invalid ordering, non-finite JSON and control payloads above 1 MiB.

`streaming.LMEXResponse(consumer, metadata_factory)` owns one public runtime
consumer until the response ends. An endpoint first performs its known request
checks (session/tensor existence, rank, supported representation and size), then
subscribes and returns this response from a `LifecycleRoute` router. The async
metadata factory returns a plain JSON-compatible metadata dictionary; it may wait
for producer metadata, and must propagate metadata-production failures. It must
cooperate with task cancellation and use the blocking-work service for disk or
compute work. The response commits headers before invoking the factory.

```python
# Inside a domain endpoint after preflight and runtime subscription:
async def metadata_factory() -> object:
    return logical_metadata


return LMEXResponse(consumer, metadata_factory)
```

The adapter reads incrementally with runtime backpressure, carrying at most
three bytes between unaligned producer appends. Runtime EOF establishes producer
success and successful artifact publication before COMPLETE is sent. Pre-META
failures/cancellation use a terminal-only stream; subsequent failures terminate
the partial result. Unexpected exception messages are kept off the wire. A failed
ASGI send ends delivery without attempting another frame on a possibly truncated
transport. Explicit operation DELETE and socket disconnect release only that
consumer's interest; shared production can continue for another consumer.

Progress emission is optional: the common writer supports it, while this response
currently emits metadata, data and terminal frames only. No tensor, statistics or
distribution product endpoint is introduced by this adapter. See
[`evidence/lmex-streaming.md`](evidence/lmex-streaming.md) for transport evidence.

## Local tokenization

`POST /sessions/{session_id}/tokenize` accepts `{"text":"A😀é<special>"}`
and optional `add_special_tokens` (default `true`). It returns the exact text
and option, with ordered token indices, IDs, native strings, individual decoded
strings and special-ID flags. Valid offsets use Unicode code points in the
original text. Overlaps and literal special-token spans are preserved; inserted
`(0,0)` sentinels and unavailable/invalid offset pairs are omitted. Individual
decodings retain special tokens and disable cleanup; concatenation is not an
input reconstruction guarantee.

The lifespan-owned `TokenizerService` loads the session's actual Hugging Face
tokenizer with local files only and remote code disabled. It stages conventional
snapshot assets in a temporary directory for loading, excluding weights, Python
code and unrelated chat templates. File overrides must refer to these staged
assets. Model files remain read-only. A missing, malformed or unsupported
local tokenizer returns HTTP 422 `unsupported_representation`, with no fallback
to another model or download. Tokenizers needing uninstalled optional libraries
are unsupported in this environment.

Instances are reused by content fingerprint, with source checks before and after
loading/use. Changes require a new session (HTTP 409 `model_content_changed`).
The worker pool runs loading, encoding and decoding outside the event loop. A
per-instance lock protects the Hugging Face wrapper's internal mutable state;
request options are passed per call. No chat template, padding, truncation,
inference, operation ID, artifact or CUDA queue is involved. HTTP disconnects
and task cancellation release response interest and may cancel work that has
not started; an already running native encode finishes in its worker.

`tests/test_tokenization.py` generates tiny byte-level BPE and normalizing
WordPiece tokenizers locally and compares results directly with Hugging Face.
It also covers missing offsets, concurrent options, snapshot changes, prohibited
network/weight work, responsive metadata and disconnect/task cancellation.

## Logical tensor data

`GET /sessions/{session_id}/tensors/{tensor_id}/data` streams one complete
logical tensor through the common LMEX adapter. Safetensors F32 reads preserve
raw float32 bits without creating a persistent copy. F16/BF16 use the existing
native CPU source conversion in blocks of at most 65,536 elements (256 KiB of
logical output); configuring CUDA does not route this disk/conversion path
through a GPU. Shape remains generic, including scalars and zero dimensions.

`Services.logical_tensors` is a `LogicalTensorService`. Its blocking
`resolve(source, tensor_id)` returns a small `LogicalTensor` with `descriptor`,
`spec`, and `metadata()`. HTTP callers use `subscribe(sessions, session_id,
tensor_id)`. Numerical artifact producers use the same service directly:

```python
logical = await context.io(service.resolve, source, tensor_id)
async with service.dependency(context, logical) as consumer:
    while block := await consumer.read():
        # Feed bounded little-endian float32 bytes to the numerical computation.
        ...
```

Dependencies have no public operation ID and inherit parent cancellation.
Exhaust readers to validated EOF before declaring a numerical result complete.
F16/BF16 production shares the runtime's singleflight and complete-artifact
store, keyed by fingerprint, tensor ID, physical dtype and exact float32
little-endian C-order producer recipe. Warm reads replay disk bytes; independent
read guards also reject changed session snapshots on cache hits. F32 dependencies
own direct source cursors and use the same cancellation/session teardown.
The service does not compute statistics, distributions, or rendering transforms.
See [tensor data evidence](evidence/tensor-data.md) for validation and limits.

## Tensor statistics and distributions

`GET /sessions/{session_id}/tensors/{tensor_id}/statistics` and `/distributions`
use `tensor_analysis.py` and the same operation, logical materialization, cache,
and LMEX services as tensor delivery. Statistics work for arbitrary rank;
distributions reject non-matrices before streaming. Their numeric semantics and
wire layout are defined in [the API contract](../docs/spec/api/contract.md).

Each derived producer loads only its requested logical float32 tensor, exhausts
its internal materialization dependency, and then enters the common device
queue. The tensor-data endpoint never waits for a derived artifact. F16/BF16
conversion can be shared by all three requests; no HTTP self-request is used.
Native PyTorch sort provides exact percentiles without `torch.quantile`'s input
size restriction. Population variance uses float64; native bounded integer
exponent reductions preserve cancellation-sensitive means. Histogram binning
uses bounded float64 blocks and checked int64 accumulators before uint32 output.

The two kinds have separate fingerprint/recipe keys. The internal statistics
payload is 96 bytes (`<3Q9d`: counts, minimum, maximum, mean, stddev, percentiles).
The distribution payload has a 16-byte `<2d` domain prefix followed by row and
column counts. Internal NaNs encode absent finite fields. The response adapter
consumes these prefixes into schema-validated metadata, emits JSON nulls for
absent fields, and streams only the declared numeric DATA sections. Statistics
have no DATA frames. Prefixes are cache representation details, not LMEX bytes.
Warm reads decode the disk prefix without numerical recomputation. Source
checks, independent consumer cancellation, and atomic publication apply to both.

See [analysis evidence](evidence/tensor-analysis.md) for numerical tolerances,
edge cases, cancellation/queue tests, synthetic timing, and workspace limits.

## Input embedding rows

`POST /sessions/{session_id}/embeddings` accepts ordered `token_ids` and streams
only those input-embedding rows through LMEX as little-endian float32
`[token_count, hidden_size]`. Order, duplicate IDs, and empty sequences are
preserved. Preflight validates all IDs, matrix byte limits, and the 1 MiB META
limit before registering a consumer.

The initial resolver supports local `llama` / `LlamaForCausalLM` checkpoints
(including SmolLM2-135M Base) when configuration dimensions agree with the
architecture's input table. Missing/ambiguous architecture, custom model code
mappings, and incompatible tables return `unsupported_representation` without
using output weights or guessing from names.

F32/F16/BF16 row reads share the tensor path's guarded native conversion, with
at most 65,536 elements per block. Ascending adjacent rows are coalesced and an
immediately repeated bounded range reuses its block. Each operation owns its
reader, runs blocking reads in the worker pool, and uses existing cancellation
and session cleanup. No prompt-specific disk artifact is created. Session
pinning still hashes model assets in bounded reads; subsequent embedding lookup
reads only requested row ranges and checks the pinned snapshot throughout.

See [input embedding evidence](evidence/input-embeddings.md) for fixture coverage
and a reproducible optional local SmolLM2 comparison.

## Static architecture graph core

`llm_model_explorer.architecture_analysis` accepts guarded configuration and physical
and numeric inventory descriptors through `AnalysisInput.from_source`. It has no
runtime import of PyTorch or Transformers. Packaged family adapters explicitly
register a `Description` with reviewed model-type/architecture discriminators,
option/storage validation (`supports`), semantic producer revisions, and a `build`
callback. Consumers register packaged family descriptions explicitly; V-JEPA 2
and dense language registration are documented below.

Builders append full typed nodes, edges, parameters, symbols and repetitions;
`record_id(kind, semantic_key)` identifies an actual instance independently of its
label or insertion order. Adapters supply ordered group children and explicit
edges, including boundary ports. `unknown` and missing native bindings produce
localized diagnostics and partial coverage. `finish` checks the complete graph
against the admitted inventory. Registry analysis returns complete, partial or
unavailable without an HTTP/model/session envelope. Filesystem handles, loading
capabilities, and numerical tensors are not passed to description callbacks.
Adapters select architectural facts from configuration rather than copying raw
checkpoint strings into public records.

The API specification remains authoritative. `architecture_analysis/records.py`
is generated from the published OpenAPI, with no runtime schema-file dependency:

```sh
uv run --locked python scripts/generate_architecture_records.py
uv run --locked python scripts/generate_architecture_records.py --check
uv run --locked pytest tests/test_architecture_analysis.py
```

Construction and serialization enforce a 32 MiB maximum and reject overflow
without truncation. A response producer must reserve its own envelope bytes via
`byte_limit` and bound the entire response. File reads, startup scheduling, cache
publication, HTTP retrieval, and family semantics remain their consumers' work.
See [graph-core evidence](evidence/architecture-graph-core.md) for the independent
oracle and the limits of this fixture-only validation.

### V-JEPA 2 packaged description

The shared registry can now register the bounded `visual_encoder_predictor`
description explicitly, without importing Transformers or constructing a model:

```python
from llm_model_explorer.architecture_analysis import AnalysisInput, DescriptionRegistry
from llm_model_explorer.architecture_analysis.vjepa2 import register_vjepa2

registry = DescriptionRegistry()
register_vjepa2(registry)
result = registry.analyze(AnalysisInput.from_source(source, tokenizer_available=False))
```

It expands the selected Transformers V-JEPA 2 encoder and predictor, preserves
native parameter identities and higher-rank metadata, and diagnoses missing
weights as partial coverage. Registration is a startup consumer seam; it does not
add HTTP or cache lifecycle. See [source review and fixture evidence](evidence/vjepa2-description.md)
for the exact revisions, described path, reproduction, reusable no-tokenizer
fixtures and the distinction from later full-local-checkpoint acceptance.

### Dense language architecture descriptions

`architecture_analysis.register_dense_descriptions(registry)` registers the
reviewed Qwen3 dense and SmolLM2/Llama descriptions in a caller-owned
`DescriptionRegistry`. Pass guarded `AnalysisInput.from_source(...)` metadata;
these descriptions do not own startup, HTTP, or artifact-cache orchestration.
They expand all configured decoder instances and preserve native, tied, and
GPTQ parameter relationships. Unknown structural options return unavailable;
missing/incompatible parameters or unexplained storage yield explicit partial
coverage. Numeric inspection remains separate from architecture completeness.
See [dense description evidence](evidence/dense-language-descriptions.md) for
reviewed source/configuration revisions, supported variants, and validation limits.
