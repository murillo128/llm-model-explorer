# Architecture component isolation — issue 122

## Projection checkpoint

Activation base: `5d3522272cae1a7da8afcdf47017702b7698b36c` on
`codex/epic-issue-124`, including the #119 conservation baseline, #120 controls
and #121 authored component boundaries. The issue branch preserves the existing
default-branch runner-maintenance commit through a merge of that pinned base.

`scope.ts` resolves one source group/operation or the existing justified derived
MLP into source membership. `projection.ts` visits only those roots, with no
ancestor layout boxes. An external alias represents one exact original endpoint
in one crossing direction. Its ports and connection paths retain original
records; shapes and labels never establish signal identity. Forwarding remains
transparent only inside the selected scope, stopping at computational operations
and external endpoints. Exits and re-entries are separate external dependencies;
wholly external bypasses are explicitly excluded. The source graph is unchanged.

The projection exposes source membership and excluded source node/edge IDs.
Existing represented/hidden/filtered edge accounting remains a disjoint complete
partition. Declared source interfaces remain available for inspection. Consumption
is evaluated against the full source connectivity, independently of visibility.
An absent scope takes the unchanged global projection path.

At this checkpoint, the focused scope/projection/invariant/automatic-layout suite
passed **78 tests**. TypeScript, targeted ESLint and whitespace checks passed.
Tests reuse #119's source-reference and directed-path oracle and the actual ELK
geometry assertions. They cover authored/derived groups, leaf operations,
independent encoder/predictor components, partial/no-repetition graphs, equal-shaped
inputs, fan-out, separate K/V state, auxiliaries, residual exclusion and external
re-entry. Increasing surrounding layers from 4 to 48 leaves isolated Attention,
MLP and leaf-operation boxes, ports, routes and bounds identical.

```sh
cd ui
npm exec vitest run src/architecture-explorer/scope.test.ts \
  src/architecture-explorer/projection.test.ts \
  src/architecture-explorer/invariants.test.ts \
  src/architecture-explorer/auto-layout.test.ts
```

This is structural fixture evidence, not reference-checkpoint or browser
acceptance. Navigation is deliberately deferred until the issue-declared
independent semantic checkpoint is satisfied.

The first independent checkpoint found that external context aliases bypassed
`showContext: false`. The correction classifies those aliases using the original
source node kind while keeping exact endpoint identity. A new API-validated
context fixture proves identical reversible filtering globally and in isolation.
The external re-entry fixture now also passes normal API validation and retains
both external computation segments as excluded records. The revised focused
suite passed **79 tests**, plus TypeScript and targeted lint. A separate exact
comparison of 24 global graph/view combinations against the activation source
produced identical projection records, ELK boxes/ports/routes and bounds (excluding
elapsed layout time). No global expectations were changed.
