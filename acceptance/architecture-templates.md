# Verified shared structures: focused acceptance

This receipt covers issue #125's optional shared-structure view. It complements
[the producer/API checkpoint](architecture-template-contract.md); it does not
replace full local reference-checkpoint acceptance.

The producer and exact metadata contract passed the required independent
semantic/API checkpoint before UI navigation was integrated. The UI uses a
component-local subset of existing immutable records and the existing isolation,
layout worker and canvas. On instance change, the view rebinds exact role-mapped
source references while retaining geometry/presentation identities. Numeric
inspection still uses the existing Tensor Explorer and its lifetime guards.

## Oracles and observed checks

- The independently authored full-attention browser fixture has instances at
  layers **0 and 2**, with the intervening linear-attention component outside the
  family. Its explicit Q/K/V member list and port/edge roles do not use the
  production template matcher.
- Unit tests validate that fixture, compare ordinary overview/exhaustive/isolation
  projections with metadata present/absent, require source records to remain
  unchanged, and test exact mapping, bounded navigation and invalidation.
- Browser tests compare layout invocation count, camera, every visible box/port
  coordinate and SVG route through neutral/concrete switching. Native pointer
  tests exercise the shared Q/K/V trunk and individual branches; pinned connection
  inspection resolves the chosen source edges. Tests exercise concrete entry,
  nonconsecutive next/previous, Back/View in model, and recovery after failed or
  cancelled worker requests. Controls are captured at 1440, 1178 and 390px.
- The built UI and real Uvicorn backend use two independently authored synthetic
  F16 layers, with exact logical values `100 * (layer + 1) + (offset % 29 - 14) / 8`.
  The production producer, API validator, native parameter lookup, binary stream
  and WebGL upload are used. Both DPR 1 and 2 pass: neutral inspection starts no
  tensor requests; layer 0 Q uploads all 144 expected values while streaming;
  changing instance dismisses the modal, cancels consumers and removes temporary
  artifacts; layer 1 Q uses its distinct tensor ID and all 144 expected values.
  Releasing the old producer cannot populate the replacement instance. Concrete
  interface inspection lists the real external source edges.
- Eight subsequent switches per DPR start no requests or layout work. Forced-GC
  observations retain one layout and two observed input graphs (the original
  graph and one component subset), with zero active workers and zero live numeric
  readers/textures after closing inspection. Observed final JS heap was about
  11.2MB at DPR 1 and 10.9MB at DPR 2; these are fixture measurements, not limits.
- The response-envelope test enforces the mandatory graph byte limit separately
  from optional metadata. A new service test covers rejecting an oversized warm
  immutable artifact, then cold rebuilding a smaller optional annotation set and
  reading it warm without changing coverage or source records. This corrects the
  former expectation that even an optional-only overflow must disable the model.

Validation: all 942 UI unit tests; generated API check, TypeScript, ESLint and
production build; all 42 focused backend service/template tests; backend mypy and
Ruff; acceptance-fixture Ruff. Browser regression and exact-head CI results are
retained in the PR checks. No source or numeric renderer algorithm was replaced.

## Reproduction and limits

Use the locked dependencies and commands documented in [README.md](README.md).
Focused commands from the repository root:

```sh
PYTHONPATH=backend/src backend/.venv/bin/python -m pytest backend/tests/test_architecture_service.py backend/tests/test_architecture_templates.py
npm run check --prefix ui
npm run test:browser --prefix ui -- tests/architecture-connections.spec.ts tests/architecture-isolation.spec.ts tests/architecture-inspection.spec.ts tests/architecture-controls.spec.ts
PYTHONPATH="$PWD/backend/src" xvfb-run -a npm run test:acceptance --prefix ui -- architecture.spec.ts --grep 'shared structure production'
```

The local Xvfb host driver crashed during GLX initialization. Running Xvfb with
`-s '-screen 0 1440x1000x24 -extension GLX -nolisten tcp'` restored the headed
SwiftShader tests; production/browser behavior and acceptance assertions were
unchanged. F16 fixture values are exactly representable. A preliminary F32 fixture
used the direct read path and did not reach the conversion-producer delay barrier;
the final test deliberately uses F16 to exercise cancellation of that producer.

These are synthetic fixture and live-service results. No approved complete local
reference-checkpoint manifest was available for a fresh actual-checkpoint run.
No model bytes, cache contents, bulky traces or generated screenshots are in Git.
