# Bounded neuron comparisons

Issue #319 implements zero/scale/add at the existing post-product seam and
independent greedy continuations. This is CPU fixture evidence; complete local
SmolLM2 and optional CUDA acceptance remain separate integration work.

## Test plan

| Contract / plausible defect | Primary proof and independent oracle |
| --- | --- |
| Wrong selected coordinate, projection, operator or cumulative mutation | Shared explicit-weight MLP with scalar SiLU, projection and full-vocabulary softmax derived using Python math; zero, identity scale/add, scale-zero equivalence and nonzero delta |
| Union renormalization, omitted out-of-top-k probabilities, unstable ties | Deliberately different top candidates, ascending union, omitted vocabulary mass and equal-probability token-ID tie checks |
| Wrong absolute scope, mixed prefixes or stale condition state | Seeded packaged Transformers model with an independent MLP replacement oracle; prompt/continuation/all, exact prefixes, greedy length/EOS, and a subsequent clean baseline |
| Invalid options reach model loading | Comparison-specific route preflight sentinels; common selection validation retained from #317/#318 |
| Partial condition reported as success or retained after interruption | Controlled interruption during a non-base condition, hook and reader lifetime assertions; inherited runtime tests retain two-session and teardown ownership |
| Incorrect wire layout or late operation headers | One real TCP comparison scenario: exact META/float32 table and a pre-META terminal failure/cancellation while headers are already available |
| Python writer disagrees with independent contract | Extend existing shared neuron fixture tests to comparison metadata and wire records; no duplicate LMEX parser matrix |

Float32 fixture comparisons use `rtol=1e-6`, `atol=1e-7` for scalar-math
versus native kernels; the real Transformers oracle uses the same tolerance.
These are numerical tolerances, not claims of bitwise hardware reproducibility.

## Implementation notes

Each condition recomputes its exact prefix with no KV state, applying the hook
only to absolute prompt or continuation positions. The table retains distributions
from the shared prompt, even when generated branches diverge. Stable full-vocabulary
softmax is computed in float64 and converted to float32 before ranking; ties use
ascending token ID. Only `[condition, union candidate]` probabilities go on the
wire. Clients derive baseline deltas, without renormalizing the candidate union.
Greedy decoding uses logit argmax, also choosing the lowest ID for exact ties;
a dedicated tiny-logit case distinguishes that from ties introduced by float32
probability rounding. Prompt EOS never prevents an explicitly requested generation.

The comparison reader uses the existing embedding-analysis pattern: an internal
bounded header is read through the consumer before META, followed by DATA. All
generation and decoding happen inside the exclusive runtime lease. There is no
experiment cache, prompt log or new wire frame.

## Reproduction and handoff evidence

Use the locked backend environment (Python 3.12, CPU float32). When reusing a
read-only dependency environment, set `PYTHONPATH=src` from `backend/`:

```sh
python -m pytest tests/test_neuron_comparisons.py tests/test_neuron_activations.py tests/test_neuron_execution.py tests/test_lmex.py -q --durations=10
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest -m 'not extended' -q --durations=10
python -m pytest tests/test_tensor_analysis.py -m extended -q --durations=10
```

From the root, run `api/validate_contract.py` with the environment built from
`api/requirements.txt`, and `bash acceptance/check-wheel.sh ABSOLUTE_BACKEND_PYTHON`
for installed-wheel proof outside the source tree. Exact publication revision,
commands, outcomes and capability skips are recorded in the PR. Existing workflows
defer epic-child CI for `codex/epic-issue-*` targets; aggregate integration gates
remain required. No complete reference-checkpoint or CUDA claim is made here.
