# Tensor statistics and distribution evidence

The production endpoints and `tests/test_tensor_analysis.py` implement the
accepted numerical contract without modifying API schemas or renderer behavior.
Fixtures are generated locally; no model weights or cache artifacts are in Git.

## Numerical and endpoint coverage

- Mixed-sign asymmetric matrices, constant values, outliers, signed zero, empty
  dimensions, all-NaN, infinities, subnormal F32, and opposite-sign F32 extremes.
  Python `statistics.mean`/`pstdev` and independently interpolated sorted values
  provide reference results. Derived values use the contract tolerance
  `abs(actual - expected) <= 1e-7 + 1e-6 * abs(expected)`; minimum/maximum,
  counts, section bytes, and offsets are exact.
- Means retain small residuals such as `[max_f32, 1, -max_f32] -> 1/3`.
  PyTorch `frexp` and int64 scatter reductions sum at most 65,536 24-bit
  significands per exponent per block. Only the 277 exponent totals are combined
  in Python integers, then exact binary limbs are accumulated with `math.fsum`.
  There is no Python loop over weights. Population variance uses native float64
  Welford reduction; interpolation widens sorted F32 endpoints to float64.
- An in-memory descending sequence of **16,777,217** values checks the exact
  percentile interpolation and analytical population standard deviation above
  `torch.quantile`'s size limit. The implementation always uses native exact sort,
  so both small and reference-scale inputs exercise the same fallback algorithm.
- Histograms match an independent scalar oracle byte for byte. Every row and
  column sum matches its finite count, both totals match the tensor finite count,
  constant values use bin 50, and the maximum uses bin 99. Asymmetric dimensions
  also cross source work blocks and output frame boundaries. Shared golden
  uint32 byte-order and overflow fixtures are consumed directly.
- Rank-0/1/3 distributions and unsafe output sizes fail before response headers.
  Statistics accept these ranks. Real endpoints validate metadata and emit
  metadata-only statistics or exact row-first/column-second uint32 DATA.
  Logical F16/BF16 materialization is exercised alongside direct F32 inputs.
- Two derived artifacts survive application restart and serve new sessions
  without calling numerical production. Changed model fingerprints create new
  keys; stale sessions and sources changed during calculation are rejected.
- Real TCP barriers demonstrate tensor DATA delivery while both derived producers
  wait. Two statistics consumers share one producer with distinct operation IDs.
  Cancelling one preserves the other; cancelling all removes unfinished work.
  Materialization dependencies use the real device queue with CPU test callbacks
  to prove queue ordering, concurrent first DATA, and cancellation propagation.
  These scheduling tests are not represented as actual CUDA numerical evidence.
- A Linux subprocess warms the real app, then limits virtual memory to current
  usage plus 256 MiB. A native 800 MB `torch.empty` request confirms that the CPU
  allocator raises ordinary `RuntimeError` with `DefaultCPUAllocator`/ENOMEM.
  The real distributions endpoint for an empty `[0, 1000000]` F32 tensor then
  fails its 800 MB histogram allocation with `ERROR_JSON resource_exhausted`.
  Limits are restored in the subprocess; the pytest process is unconstrained.
  No distribution artifact or temporary spool remains. Only the recognized CPU
  allocator ENOMEM signature and typed PyTorch OOM are translated; other runtime
  errors, allocator names, and error numbers remain internal failures.
- MemoryError/native PyTorch OOM, source changes, and cancellation/failure after
  an internal cache prefix is written emit the proper terminal result, leave no
  valid partial artifact, and clean temporary files. Error payloads hide paths.

## Workspace and timing

One producer retains one requested CPU float32 tensor (`4*N` bytes); CUDA adds
its device copy while computing. Statistics additionally need the finite F32
selection, a transient F64 variance buffer, and native sort values/indices and
scratch. The largest buffers are O(N), confined to that tensor. Histogram state
uses `8*100*(rows+columns)` bytes of int64 accumulators, then a CPU uint32 result;
its other temporaries are bounded to 65,536 values. Output writes are at most
256 KiB. Concurrent independent derived kinds can each own this workspace;
there is no whole-model load, global RAM result cache, or approximation mode.
Allocation failures are reported as `resource_exhausted`.

Measured on 2026-09-13, Intel Core i7-11700K, Linux x86-64, Python 3.12.14,
PyTorch 2.14.0+cpu, two native threads. Each row below was a fresh process with
seed-9 normal F32 values. Time covers numerical production only, excluding input
creation, source I/O, cache publication, and HTTP. Peak RSS includes interpreter,
PyTorch and input; increase is above the post-input process high-water mark.
These are single synthetic measurements on a shared host, not hardware claims
or reference-model benchmarks.

| Kind | Shape | Input MiB | Time (s) | Peak RSS MiB | Peak increase MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Statistics | 576 × 1536 | 3.375 | 0.0661 | 263.8 | 31.2 |
| Statistics | 4097 × 4096 | 64.016 | 1.4724 | 689.5 | 396.1 |
| Distributions | 576 × 1536 | 3.375 | 0.0203 | 248.3 | 15.5 |
| Distributions | 4097 × 4096 | 64.016 | 0.3684 | 314.4 | 20.7 |

Reproduce a row from `backend/`:

```sh
OMP_NUM_THREADS=2 uv run --locked python scripts/benchmark_tensor_analysis.py statistics 4097 4096
```

## Validation

The locked Python 3.12 environment was installed successfully after the host's
full filesystem was freed. Ruff check/format, strict mypy, source pytest, and
installed-wheel pytest are the repository-native checks. Actual CUDA numerical
cases are capability-gated and **skipped** on this CPU-only host; queue tests
cannot establish GPU numerical equivalence. Existing warnings concern
Starlette/AnyIO deprecations.

Local final results: Ruff check/format and strict mypy pass. Source and isolated
installed-wheel suites each pass **446 tests**, with **16 CUDA cases skipped**
(61 passing derived-analysis cases added). The wheel was built with `uv build`,
installed without dependencies into `/tmp/issue-9-installed`, and imported/tested
from `/tmp` to verify the installed package independently of the source tree.
