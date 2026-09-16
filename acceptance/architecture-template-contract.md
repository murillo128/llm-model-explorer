# Shared-structure contract and producer checkpoint

Issue #125 starts from accepted main `5bdc5d782c9a40d36fa475d7c026d27955d8094d`.
This intermediate checkpoint publishes optional annotations only; UI navigation
and final integrated acceptance follow after independent semantic/API review.

The complete source graph remains authoritative. Reviewed description factories
open component candidates explicitly and assign authored relative roles during
construction. Exact signatures verify complete containment, ports, edges,
attributes/formulas, logical parameter geometry and internal alias relationships.
Physical bindings, availability and numeric IDs remain instance-owned. No weight
reads, execution, model downloads, extra graph endpoint or layout changes occur.

The independent API fixture has two full components at source indices 0 and 2,
with Q/K ports of identical shape. Mutation cases reject missing/duplicate/foreign
mappings, wrong subtrees/instance parameters, swapped Q/K ports, different
operations/formulas/normalization/bias/shape/state, unknown metadata, alias changes
and malformed publication. Cross-language validators consume the same authored
cases through generated closed schemas. Explicit producer tests verify supported
descriptions, nonconsecutive hybrid membership, independent visual stacks,
unchanged ordinary records with annotation collection disabled, singleton fallback
and exact-budget fallback without coverage changes.

Validation on Python 3.12.14 and Node 24.14.0:

- Independent API: OpenAPI valid, 347 references, 185 architecture cases.
- Focused backend contract/grouping/template tests: 221 passed.
- Backend mypy: 76 source/test files passed; Ruff lint/format passed.
- UI API generation, TypeScript, ESLint, all 939 unit tests and production build
  passed. The two focused schema/context suites contain 378 passing checks.

These are fixture/static checks. They do not claim actual-checkpoint numerical
or browser acceptance. Full backend and subsequent integration results are
recorded in the PR/check evidence; no model/cache payloads are committed.

The full backend suite at the initial checkpoint passed 1,369 tests with 20
CUDA-only skips. Fresh intermediate review of `3cf123f03fa09c54f281ee72400d439bb1408d88`
found one material boundary defect: separators between optional template records
were undercharged. The correction charges those separators and checks six byte
limits around the complete two-template size, including one byte below it; source
records and coverage remain intact. All 16 producer-template tests pass after the
correction. No UI navigation was integrated before this review.

Fresh semantic/API review of corrected head
`583c2c361cde8bc90baabad8a84e1c021db635f6` returned **PASS**, with
`final-capable: no` (intermediate checkpoint only). The reviewer reran all 16
producer-template tests and eight adjacent byte-budget limits for each of the
dense and visual fixtures. At one byte below the complete dense document, the
producer retains one optional template and every original graph record with
complete coverage. No further material findings remained. UI navigation work
began only after that PASS; final acceptance belongs to the PR audit controller.
