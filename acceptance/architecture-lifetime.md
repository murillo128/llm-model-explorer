# Mounted Architecture layout lifetime — issue 149

The correction releases superseded layout results during nested isolation and
Back on the same mounted canvas. The preserved eight-cycle regression passes
without changing its actions, assertions or weak-worker observer. An additional
sixteen-cycle window checks bounded retention and whole-explorer teardown.

## Source and diagnosis

The uncorrected runtime is `f7764fd69fffde644dd484ba2f7ca851123f2206`.
The failing regression and observer come from acceptance #123 at
`78a7d3a6d427f28dd42420da6ef2b41fb5d5a2f2`. Only bounded lifetime,
delayed-session and concrete decoded-instance coverage is carried into this
repair; the incomplete acceptance report and other #123 changes are not imported.

The product correction is commit
`6b18768746b17da21f833cc22790785c52849b69`. Exact test hashes, source tree
identities, commands and observations are in [the receipt](evidence/architecture-lifetime/checks.json).

A fresh production run against the original runtime retained **7, 12, 17, 22,
27, 32, 37, 42** layouts after eight cycles. A tracing-disabled control retained
**7 through 82**, increasing by five on each of sixteen cycles. Every completed
worker terminated; one source graph remained. Unmount released both graphs and
layouts. The original eight-cycle assertion failed with `42 <= 12`.

The fresh no-trace heap snapshot confirms a strong path from the mounted
**Model overview** button's React click property through successive Canvas
closure contexts (`reveal`, `rememberAnchor`, inspection callbacks) to previous
`{ layout, options, invocation }` results. For an early retained result the
shortest strong path crossed 333 references. This confirms application callback
retention despite successful worker termination. Raw heaps and traces remain
outside Git.

## Correction and ownership

`useCanvasCallback` creates a per-hook event slot in a separate factory scope.
The stable event function retains that slot. A layout effect publishes the
handler for the committed render and clears it during cleanup; a speculative or
suspended render cannot overwrite it. The factory does not capture an earlier
Canvas invocation. Existing Canvas event callbacks use this seam consistently,
so independently memoized functions no longer link successive render contexts.
The two synchronous helpers that return values stay local to their render.

The source graph, layout algorithm, projection, worker cancellation, view/history
objects, React keys and scientific rendering paths retain their existing
implementations. The helper owns no global state, graph copies, history cache or
weak production references. Three unit tests cover current instance/argument
reads, a suspended navigation that must keep the preceding committed handler,
and StrictMode cleanup/session replacement with inert disposed handlers.

## Observed retention

| Production case | Retained layouts per returned view | After unmount |
| --- | --- | --- |
| Original runtime, eight cycles | 7, 12, 17, 22, 27, 32, 37, 42 | Separate control: 0 |
| Original runtime, sixteen cycles, trace off | 7 to 82, increment 5 | 0 |
| Correction, eight cycles, DPR 1 and 2, trace off | 2 at every cycle | Separate extended case: 0 |
| Correction, sixteen cycles, DPR 1 and 2, trace off | 2 at every cycle | 0 |

Two is an observation, **not an acceptance threshold**. The unchanged oracle
compares cycle eight with the identical returned state after two warm-up cycles;
the extended case also compares cycle sixteen with cycle eight. A corrected
heap snapshot contains two layout objects: current control props retain one,
and React's effect-cleanup state retains the other. It has no growing chain of
older Canvas callback contexts. Both become collectable on ordinary teardown.

Post-GC main-thread heap observations remain labelled separately from object
retention. In the initial paired runs, the baseline grows from 9,081,680 to
12,699,756 bytes over eight cycles, and the no-trace sixteen-cycle control from
8,834,924 to 16,274,160 bytes. Corrected DPR 1 reads 8,349,820 to 9,008,280 bytes
for eight cycles and 8,297,880 to 9,312,572 for sixteen. DPR 2 reads 8,344,096 to
8,993,544 and 8,336,624 to 9,320,080 respectively. This is not a claim of flat
total heap usage, peak browser RSS, GPU allocation or hardware-independent speed.

An extra heap diagnostic failed a comparison between the initial overview
camera and the subsequently returned, centered camera. Layout arrival precedes
the camera's animation-frame update, so the extended test now waits for the same two animation frames as the existing component harness
before taking camera snapshots. The original eight-cycle block remains
byte-for-byte identical to #123, and no retention assertion or GC observation
is relaxed. The diagnostic outcome and subsequent validation are retained in
the receipt.

## Preservation checks

- UI generated bindings, TypeScript, ESLint, production build and **734 unit
  tests** pass.
- The existing independent global comparison passes all **24** fixture/view
  combinations against `5d3522272cae1a7da8afcdf47017702b7698b36c`, comparing
  complete projection, boxes, ports, routes, source edge IDs and bounds; only
  elapsed layout time is excluded.
- All **40 focused component-browser cases** pass on the canonical desktop
  project, including controls, isolation, exact port/trunk interactions, concrete
  inspection, cancellation/error/retry, late results and whole-explorer teardown.
  Independent authored, partial, multi-stack and full-size synthetic fixtures
  preserve source/projection/geometry checks.
- All **14 production/live-backend cases** pass at DPR 1 and 2: the four
  reduced model families, exact native/decoded later-instance inspection,
  progressive cancellation, delayed actual session-response rejection and the
  original eight-cycle regression under normal tracing. The final extended
  sixteen-cycle control also passes at both DPRs with tracing off. Its matching
  uncorrected-runtime run still fails only the retained-layout assertion.

## Reproduction

Use isolated source snapshots, locked Node 24.14.0/npm dependencies and the
locked Python backend environment. For the original eight-cycle failure,
overlay only `ui/acceptance/architecture.spec.ts` and `architecture-probe.ts`
from `78a7d3a` onto the pinned original runtime. Build each UI before running.
Use the repository's live reduced-fixture service, not mocked graph responses.

```sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/backend/src"
npm --prefix ui run check
cd ui
UI_TEST_PORT=25620 xvfb-run -a \
  --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- architecture.spec.ts --project=dpr1 \
  --grep 'repeated nested return releases obsolete layouts'
UI_TEST_PORT=25660 xvfb-run -a \
  --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- architecture.spec.ts --trace off \
  --grep 'repeated nested return releases obsolete layouts|extended nested returns stay bounded'
```

Give each run separate output directories and ports; run one heavy suite at a
time. The receipt lists the additional focused browser commands. Heap captures,
raw logs, generated fixtures, caches and browser artifacts are local temporary
evidence, not committed runtime assets.

## Acceptance boundary

This is the corrective UI child, not completion of #123 or epic #124. Full local
checkpoint acceptance, remaining TCP suites, final exterior visual review and
integrated performance gates remain with #123. Reduced live-backend fixtures,
including native/GPTQ/NVFP4 inspection, do not establish actual-checkpoint or
CUDA acceptance. Final independent audit belongs to the normal review-ready
controller after this executor's handoff.
