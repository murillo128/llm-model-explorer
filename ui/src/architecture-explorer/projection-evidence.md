# Issue 106: projection checkpoint

This historical intermediate checkpoint covered source-preserving view derivation
before the production canvas used it. The independent review returned
PASS_WITH_NOTES on a7367ce, allowing UI integration.
Final layout and browser evidence is recorded in
[the acceptance report](../../../acceptance/architecture-presentation-evidence.md).

## Inputs and boundaries

- Approved HTML downloaded from issue 106 and opened in Chromium before edits.
  SHA-256: `45bdc40c74192de94be1f73df5f1d1bf3150da0fa65b3bf2a4a6a493cae85e54`.
- Both binding design comments were read. Runtime changes remain inside the
  Architecture panel and retain the existing visual language.
- Original local graph: `7a11207126ee52d510e406921fa8d0890fd7aba9bd4e6e1b7b5449cb2de14379`,
  descriptor `transformers-qwen35-nvfp4`, revision 1, reviewed Transformers
  revision `2cba19507be799b7bef247ca6c1c4708bf881b5b`.
  Its local artifact digest matched its manifest:
  `3969cbe6065a19ec74e0b538d34d89b646d754ae7bac5494224433970c6dbc3c`.
- HTML, original graph, response replay and screenshots remain outside Git.
  Fixtures in this change are deliberately authored source topology with no
  checkpoint weights or copied cache records.

## Observed local structural reproduction

Source: 948 nodes, 1,329 edges, 333 parameters, 49 groups; 24 ordered instances
(18 linear attention, 6 full attention, L–L–L–F repeated six times).

| View | Visible nodes | Connections | Represented original edges | Internal/hidden | Explicitly filtered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overview | 13 | 8 | 40 | 1,247 | 42 |
| Four-instance example window | 17 | 17 | 44 | 1,243 | 42 |
| Expanded instance 0 | 20 | 17 | 52 | 1,235 | 42 |
| Expanded instance 3 | 21 | 20 | 54 | 1,233 | 42 |
| Exhaustive | 948 | 1,168 | 1,329 | 0 | 0 |

Connections compose boundary forwarding and may represent several original
segments. Counts therefore intentionally differ from original edge counts.
Every exhaustive path was checked for exact source-record identity and adjacent
endpoint continuity. No computational operation is traversed as a transparent
boundary. Original source records are unchanged.

Consumption follows source edges: instance 0 has unused declared `positions`
and `mask`; instance 3 has unused declared `current_mask`. Their real parameter
references and state ownership remain distinct. MLP grouping requires matching
parameter ownership and exact gate/up/SiLU/multiply/down topology.

## Before-image reproducibility

The built baseline is commit `0e8baebb94932755b2a084afaa4df3ec689e984d`.
Chromium captures use 1178×900 and 1440×900 viewports, DPR 1, fixed session identity
and deterministic replay of real local service responses. Ten captures cover
Architecture, native Tensor inspection, Tokenizer and switching through
Architecture. A second baseline run gave zero changed pixels and identical
geometry. Only the rounded Architecture panel interior is excluded from its
shell comparison; Tensor and Tokenizer comparisons retain the entire image.

The baseline Tokenizer prompt clears when returning from another explorer;
that pre-existing behavior is recorded separately from the populated prompt.

## Reproduce deterministic structural checks

```sh
npm --prefix ui test -- src/architecture-explorer/projection.test.ts
npm --prefix ui run typecheck
```

This checkpoint does not replace the repository final PR audit.
