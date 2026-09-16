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
