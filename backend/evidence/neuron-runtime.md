# Local neuron runtime evidence

Issue #317 implements the runtime and capability portion of the accepted
[backend contract](../../docs/spec/backend/neuron-execution.md). Public capture,
comparison, continuation orchestration and their LMEX result layouts remain the
following implementation increments. This report is fixture evidence, not complete
SmolLM2 checkpoint acceptance or CUDA numerical acceptance.

## Implementation boundary

- `neuron_execution.admission` checks the declared SmolLM2 Base identity, any
  declared reference revision, the reviewed configuration, all 272 native weight
  shapes and floating storage dtypes, and a loadable local tokenizer with the
  exact vocabulary range. The existing directory-name fallback is accepted only
  together with these checks. Local provenance declarations do not authenticate
  upstream weight bytes. Unsupported models retain their static capabilities.
- `GET /sessions/{session_id}/neuron-capabilities` reports all concrete MLPs,
  existing inventory IDs, gate/up rows and down columns, limits and actual
  execution environment. It does not construct or load a language model. No
  unverified graph-node association is supplied.
- The loader constructs only the packaged Transformers 4.57.6 `LlamaForCausalLM`
  with whitelisted configuration, eager attention and float32 weights. Meta
  construction avoids random initialization; the reviewed default RoPE buffer
  is recreated on the configured device. Guarded tensor readers populate native
  parameters without `from_pretrained`, remote code, pickle or downloads.
- `NeuronInput` performs common selection/revision/token preflight before a
  lease is queued. Downstream comparison code must additionally preflight its
  conditions, output sizes and continuation bounds. `NeuronRuntime.run` belongs
  inside `Consumer.prepare`, using `OperationRuntime.subscribe_source` with its
  session ID, so operation deletion, session deletion and disconnect own work.
  Result readers must preserve source guards throughout delivery and before EOF.
- One application instance is keyed by content fingerprint, pinned snapshot,
  sanitized configuration, adapter/runtime versions and effective dtype/device.
  A cancellable exclusive lease covers the complete callback, including hooks
  and all forwards. CUDA uses the existing device scheduler and synchronization;
  CPU leases also exclude concurrent hooks. Replacement drops the prior instance
  before allocation. The source is checked before/after loading and computation.
- Forward calls use evaluation/inference mode, exact prefix IDs, no autocast,
  no KV cache and no sampling/RNG state. Operation-owned hooks attach immediately
  before `down_proj`; cleanup settles blocking work before reuse. No prompts,
  logits or activations are added to the artifact cache or application history.
  OOM and numerical failures have safe contract errors; optional error details
  now survive both HTTP and LMEX adapters.

## Test plan and observations

| Contract / plausible defect | Primary proof and independent oracle |
| --- | --- |
| Correct real loading, RoPE, tied weights and forward | Seeded tiny local model saved by the real Transformers class; compare runtime logits to direct uninstrumented execution (`rtol=1e-6`, `atol=1e-7` on CPU float32), with duplicate prompt IDs preserved |
| Exact neuron point and concrete weight axes | Manual `SiLU(linear(x, gate)) * linear(x, up)` at the reference MLP input; explicit later-layer inventory row/column expectations |
| Production admission does not accidentally accept tiny/arbitrary Llama | Independently checked reference metadata supplies all 272 expected shapes; mismatched identity/revision/configuration, missing weights/tokenizer and runtime mismatch are rejected |
| No implicit full-model load; invalid requests allocate nothing | Weight reader/loader sentinels during capabilities and static inspection; selection, vocabulary, revision and size failures before the loader/operation |
| Shared mutable hooks or stale content contaminate another operation | Intervention followed by an independent baseline; two session consumers; unchanged weight bytes; repeated pins reuse once, changed content reloads and matches a separately updated direct model |
| Unsafe early cancellation releases active work | Controlled thread events during loading and an active hook, queue cancellation, repeated task cancellation, session deletion, disconnect and shutdown; next baseline equals the direct model |
| Exceptions/OOM/non-finite results poison reuse or leak diagnostics | Injected hook failures plus clean subsequent forwards; finite-value guard; HTTP and real TCP LMEX error-detail cases; controlled CUDA queue/synchronization OOM double; a retained-error/weak-reference regression proves failed models are released |

The tiny geometry override exists only in the test fixture. Production has no
test-model admission option. The CUDA scheduling/OOM double proves coordination,
not hardware execution. Existing operation/stream tests retain shared-consumer,
backpressure, cancellation and terminal behavior coverage.

Validation uses locked dependencies with an isolated writable uv cache. The focused
neuron suite passes 34 cases, alongside scoped sessions, operations, streaming,
app and CLI checks. The retained-error regression failed before the traceback
cleanup fix and passes after it. Ruff, format checks and strict mypy pass.
Dependency and packaging declarations are unchanged. Final revision-bound wheel
and cross-version results are recorded in the PR. From `backend/`:

```sh
uv run --locked pytest tests/test_neuron_execution.py tests/test_sessions.py tests/test_operations.py tests/test_session_operations.py tests/test_streaming.py tests/test_app.py tests/test_cli.py -q --durations=10
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
```

Product workflows explicitly defer CI for PRs targeting `codex/epic-issue-*`.
Scoped local checks belong to this child; the aggregate epic PR retains its
applicable full validation routing. No UI or full-checkpoint acceptance is claimed.
