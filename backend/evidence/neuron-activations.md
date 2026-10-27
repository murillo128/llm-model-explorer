# Bounded neuron activation observation

Issue #318 implements observation through the existing runtime lease and LMEX
consumer. Comparison/intervention orchestration belongs to the next increment.
This is small-fixture CPU evidence, not complete SmolLM2 checkpoint or CUDA
acceptance; checkpoint and browser validation remain integration work.

## Test plan

| Contract and plausible defect | Primary owner and independent expected result |
| --- | --- |
| Signed post-product values, token positions and requested columns; wrong gate/up tap or layer zero fallback | Backend capture: small MLP with explicit weights/inputs and scalar math oracle, including zero, negative values, duplicates, special IDs and noncontiguous reversed coordinates |
| Correct packaged Transformers seam and unchanged logits | Reuse #317's seeded native model fixture and MLP-input mathematical oracle; capture at layer 1 and compare observation-pass logits to uninstrumented execution at CPU float32 `rtol=1e-6`, `atol=1e-7` |
| Invalid input or incompatible runtime shape computes/allocates results | Common preflight suite retained; route validation sentinels and narrow actual-shape controls at the capture boundary |
| Capture buffer/hook survives failure or cancellation | Controlled failure/cancellation after the selected buffer exists, weak references and next clean capture/baseline; existing runtime session-isolation and teardown suite retained |
| Wrong descriptor, byte order, partial success or stale source at EOF | Focused real TCP endpoint test with independently calculated values, short DATA frames and terminal/EOF assertions; failure/cancellation/source-mutation controls reuse existing LMEX consumer/parser helpers |
| Backend metadata writer disagrees with published contract | Reuse independent activation cases and wire fixtures from `api/fixtures/neurons.json`; retain the shared LMEX framing suite |

Both mathematical fixtures live in `backend/tests/neuron_fixtures.py` for reuse
by comparison tests. No copied numerical oracle or production test-mode admission
is introduced. The capture retains only `[tokens, selected_neurons]` CPU float32
values, then bounded little-endian bytes; successful or failed results never enter
the artifact cache.

## Validation

The focused observation/runtime/LMEX portfolio passes 213 cases. Ruff checks,
format checks and strict mypy pass. Independent API validation passes all 81
neuron cases, eight neuron payload cases and the existing contract/wire fixtures.

The retained-error regression for content mutation immediately after capture
fails with the pinned `3ac4750` runtime: the result reader remains alive through
the error traceback. The same test passes with the explicit result release in
this change. The baseline run used an isolated source snapshot with the new
observation tests and only `runtime.py` restored to the pinned version; no shared
worktree was reset. Final-EOF source mutation is also rejected and releases the
reader. No claim is made that an incomplete DATA prefix is a successful result.

Reproduction uses Python 3.12 and the unchanged locked backend dependencies,
with `PYTHONPATH=src` when using a shared read-only dependency environment:

```sh
python -m pytest tests/test_neuron_activations.py tests/test_neuron_execution.py tests/test_lmex.py -q --durations=10
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Run `api/validate_contract.py` from the root with the separate environment built
from `api/requirements.txt`. Sibling operation/session/stream tests and the
installed-wheel smoke are included in final publication validation.

Validation results and exact publication target are recorded in the PR. The
repository workflows explicitly defer epic-child CI for `codex/epic-issue-*`
targets; the aggregate epic PR retains its applicable gates.
