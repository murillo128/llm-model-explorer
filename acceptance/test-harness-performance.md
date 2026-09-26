# Test harness efficiency evidence

## Workload and environment

The baseline is post-Architecture Explorer `fa5bc7840c123cf1adee79e57655bcdd78af3c6a`.
Historical issue timings were not used as a before sample. On this host the
unmodified UI has 1,005 unit tests, 672 component browser cases and 98 product
browser cases. Baseline and candidate use Node 24.14.0, npm 11.9.0, Playwright
1.63.0, Chromium 153.0.8010.12, SwiftShader, one browser worker, all existing
projects/viewports/DPRs, and the same warm locked dependencies. Python HTTP tests
use 3.12.14 and PyTorch 2.14.0+cpu. No checkpoint/CUDA capability is manufactured.

Node dependencies were copied from the installed checkout into the issue
worktree. The local Python environment uses the installed locked dependencies
and this worktree's source. The package was installed editable without dependency
changes to supply its declared CLI entrypoint before the final backend gate.
No installation time is included. Baseline startup
host load was 0.40/3.95/6.69; no competing heavy suite was observed or
started during measurements. Existing idle LAN/backend services remained running
on unrelated ports. Browser runs use host access, identical SwiftShader arguments
and isolated ports beginning at 14670. The earlier failing baseline was discarded
after its separate repair; only this corrected revision is compared below.

## Coverage preservation

| Area | Preserved verification |
| --- | --- |
| Fractional zoom | All five scales, both configured viewports, all 24,689 pixels / 98,756 channels, pending prefix, band limits, five complete magnifier buffers and upload/storage checks. The expected colors still come from the independent scalar oracle and exact cell inputs. |
| Fractional guides | All seven scales and three origins, exact guide membership, one-pixel thickness/intersection, every final RGBA channel within the original one-code tolerance, and resource checks. |
| Decoder | All 132 original wire fixtures; every split from 0 through N inclusive (75,436 two-chunk variants); 1,056 coalesced/single-byte/Fibonacci/five-seed variants; unchanged 118 schema cases and four standalone incremental/Unicode/uint32/rejection tests. Private prepared bytes preserve NaN payloads and signed zero. Every variant constructs a fresh decoder. |
| Pixel capture | The progressive product geometry/inspection/cleanup case explicitly enables capture for its matrix/profile/magnifier oracles. Every other product/architecture case retains screenshots, geometry, actions, lifecycle, native upload/error/context-loss observations and separately controlled scalar/count capture. |
| Isolation and CI | Fresh processes, sessions and mutable caches; unchanged workers, retries, timeouts, reference skips, stress cycles, check names, trigger filters and Python/DPR/viewport coverage. `--durations=25` preserves exit codes and PR/main/local ownership. |

Manifest comparison has no lost existing cases. The two pixel-case title paths
gain an empty Playwright option-group segment; their semantic titles are unchanged. Fourteen unit fault checks and
six browser probe cases are added. Fault checks cover first/middle/last values,
texture/word/chunk boundaries, missing/truncated/oversized evidence, dimensions,
offsets, tolerance, metadata, terminal outcomes, immutable preparation and oracle
errors occurring before expected rejection. Probe checks exercise real GL draws,
independent scalar/count/reader/texture/error/loss counters, capture off/on,
resize including same-size property/attribute and detached-canvas resets, DPR changes and canvas replacement.

The first optimized full component run exposed one unchanged scrollbar test's
post-`Home` synchronization race (677 passes / one failure). Native scroll was
already zero while the renderer still exposed its previous animation-frame
camera. The user-approved contract clarification permits an exact renderer-view
wait after the existing native-scroll wait. The original focus invariance,
geometry, resource and input assertions remain unchanged. Its initial trace and
report are retained; only the corrected complete run is eligible as the final
gate and full-suite comparison.

## Measurement procedure

All reports and traces are outside Git under `/tmp/issue-167-resume-evidence` for this
execution. The committed evidence is deliberately compact. Normal reproductions
use the setup in [README](README.md), then the commands below from the indicated
working directory. Run heavy suites sequentially on the same otherwise idle host.

```sh
# ui/: full unit/type/lint/build gate and three samples of each focused hotspot
npm run check
npm test -- src/api/lmex-decoder.test.ts --reporter=default --reporter=json --outputFile=/tmp/decoder.json
xvfb-run -a npm run test:browser -- tests/renderer.spec.ts --grep 'fractional zoom samples'
# ui/: unchanged complete component matrix
xvfb-run -a npm run test:browser
# repository root: real HTTP, production build and both browser DPRs
bash acceptance/check-integration.sh --main-ci
python -m unittest discover -s acceptance -p test_ci_entrypoints.py
# backend/: supported Python environments, with the current source on PYTHONPATH
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest --durations=25
```

The renderer's browser `Evaluate` duration is separate from the checking span.
Baseline checking spans the first through last expectation after evaluation;
the candidate records the exhaustive loop in `Compare every fractional-zoom
pixel`. Both timed runs use the same list/HTML reporters plus this observational
reporter, supplied as the last comma-separated `--reporter` entry. Set
`LMEX_TIMING_REPORT` to a distinct JSON file for each run. It observes existing
steps only and never modifies assertions, fixtures or timing thresholds.

```js
// /tmp/timing-reporter.cjs
const fs = require('node:fs');
class Reporter {
  onBegin(config) { this.rows = []; this.config = { workers: config.workers, projects: config.projects.map(p => ({ name:p.name, use:p.use })) }; }
  onTestEnd(test, result) {
    const steps = []; const visit = s => { steps.push({ title:s.title, category:s.category, duration:s.duration, start:s.startTime }); s.steps.forEach(visit); }; result.steps.forEach(visit);
    this.rows.push({ title:test.titlePath(), status:result.status, duration:result.duration, steps, attachments:result.attachments.map(a => ({name:a.name,path:a.path,body:a.body?.toString()})) });
  }
  onEnd(result) { fs.writeFileSync(process.env.LMEX_TIMING_REPORT, JSON.stringify({config:this.config,result,tests:this.rows})); }
}
module.exports = Reporter;
```

## Results

The repeated focused samples passed with identical workloads and reporters:

| Timed work | Baseline median (range), seconds | Optimized median (range), seconds |
| --- | --- | --- |
| Decoder file, all 254 original cases | 68.612 (68.460–69.330) | 1.403 (1.389–1.428) |
| Desktop renderer checking | 7.577 (7.489–7.727) | 0.008 (0.008–0.008) |
| Narrow renderer checking | 7.330 (7.301–7.852) | 0.008 (0.008–0.008) |
| Renderer browser evaluation, all six samples | 0.905 (0.899–0.910) | 0.907 (0.901–0.915) |

The checking span retains every original pixel/channel and magnifier comparison;
the candidate also asserts scenario and magnifier lengths. Browser evaluation is
reported separately and is not the source of the checking gain. Decoder expected
preparation and native exact comparisons reduce overhead without changing the
75,436 exhaustive splits or 1,056 other fragmentation variants. These targeted
medians have non-overlapping ranges. No millisecond threshold is added to CI.

The full UI check passed 1,005 baseline tests and 1,019 candidate tests. Its Vitest
wall time was 74.49s before and 6.98s after (single observations). All 13 built
application files have identical SHA-256 hashes. Fourteen helper unit cases and
six real-WebGL probe cases pass, including deliberately invalid evidence.

The complete component browser suite passed 672 baseline and 678 candidate cases,
with zero failures or skips. Wall time was 1,493.019s before and 1,486.086s after.
This includes six added probe cases; no full-suite speedup beyond this single
observation is claimed. All 16 focused native scrollbar cases also passed after
the approved synchronization correction.

Integrated production-browser acceptance passed 86 cases with the same 12
reference skips before and after: 1,086.994s before and 1,078.496s after. HTTP
acceptance passed 31 baseline / 34 candidate cases with the same eight optional
capability skips: 106.67s before and 106.85s after. The three additions verify
entrypoint ownership and failure propagation. All nine entrypoint unit tests pass.
Full-suite comparisons are single observed before/after runs, not stable
historical averages; neither demonstrates a large suite-wide speedup.

All 84 non-pixel browser cases report zero framebuffer readbacks, bytes read and
snapshot allocations. The two pixel cases still pass all their pixel oracles:

| Pixel case | Readbacks | Bytes read | Snapshot allocations | Pixel queries |
| --- | ---: | ---: | ---: | ---: |
| DPR 1 | 557 | 153,790,240 | 28 | 216 |
| DPR 2 | 590 | 392,462,576 | 30 | 216 |

These are current-document counters, not lifetime sums across navigation. The
baseline did not count readbacks, so no invented before-count is compared. Source
inspection establishes the prior default-on capture for non-pixel consumers.

The new phase attachments explain substantial residual costs. Median seconds
across 18 architecture and 68 product cases, respectively:

| Harness phase | Architecture | Product |
| --- | ---: | ---: |
| Fixture generation | 2.402 | 0.034 |
| Backend spawn-to-ready | 2.879 | 6.883 |
| Browser setup | 0.102 | 0.129 |
| Test body | 12.822 | 3.106 |
| Teardown | 0.169 | 0.180 |

Architecture fixture generation includes its separate subprocess; product fixture
generation is measured inside, and overlaps, backend spawn-to-ready. These phases
are not directly interchangeable or additive. Product spawn-to-ready totals
466.095s across 68 isolated processes; this includes readiness polling and does
not isolate Python import cost. Fresh process/session/cache isolation is retained.
Setup/process reuse remains outside this contract, not an implemented speedup.

Local validation setup errors are retained separately: running acceptance
lint from the backend directory changed Ruff's import classification, and the
initial dependency-only Python environment lacked the declared CLI executable
(1,369 passes / two missing-executable failures / 20 CUDA skips). Native lint
working directories and an editable no-dependency package installation resolve
these setup errors. All seven focused CLI checks then passed; no source or test
expectations were changed for either correction. A restarted Python-only command
also omitted the pinned Node path needed by one HTTP architecture oracle
(33 passes / one missing-Node failure / eight skips); restoring the same Node
24.14.0 path used by integrated acceptance corrects that command environment.

Backend Ruff, formatting and mypy pass on Python 3.12.14 and 3.14.7. The complete
pytest suite passes 1,371 tests with the same 20 CUDA skips on each version
(425.35s and 446.05s, respectively). Their collected case and skip identities
match. Acceptance lint/format and HTTP tests also pass on Python 3.14.7:
34 passes / eight unchanged skips in 159.36s. The Python-version runs are
correctness checks, not like-for-like optimization timing samples.

The initial diagnostic profile over all three candidate files was deliberately
interrupted after 57 passing embedding cases (308.67s); its partial report/profile
is retained and is not a completed gate. A bounded profile then passed four
representative cases in 56.93s:

```sh
# backend/: diagnostic sample; full suites above remain the correctness gates
python -m cProfile -o /tmp/backend-representative.prof -m pytest \
  'tests/test_embeddings.py::test_contract_requests[qwen3-embeddings-null-ids-F32]' \
  'tests/test_tensor_analysis.py::test_endpoints_and_disk_reuse[shape0-values0]' \
  'tests/test_tensor_data.py::test_exact_shapes_and_native_source_unchanged[shape0-F32]' \
  'tests/test_tensor_data.py::test_all_half_finite_widening_and_nonfinite_bits[F16-2]' \
  --durations=25
python -m pstats /tmp/backend-representative.prof
```

The 57.238s cProfile total attributes 51.245s cumulative to five calls to actual
architecture `_prepare_all`, including 39.441s in `json_chunks` and 15.016s in
`preflight` (nested cumulative costs overlap). The half-float expected-value
generator accounts for 0.031s, the settings fixture 0.002s, and the selected
statistics/histogram oracles less than 0.001s each. The partial profile likewise
points to production graph validation/serialization. This supports investigating
startup work separately, not caching calculated responses or sharing mutable
test state here. No Python fixture reuse is implemented or claimed.

Profiling adds substantial instrumentation overhead and other host Python work
was observed during this diagnostic phase. It is a bounded attribution sample,
not an exclusive-host benchmark or a claim about all Python tests. No profiled
timing is substituted for the repeated UI hotspot or full correctness evidence.
