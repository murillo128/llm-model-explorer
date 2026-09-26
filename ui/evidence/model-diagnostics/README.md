# Model diagnostic presentation evidence

## Reproduction

Use the locked Node 24.14.0/npm version and `npm ci` in `ui/`, then:

```sh
npm run check
UI_TEST_PORT=45195 npm run test:browser -- --project=desktop --project=narrow \
  model-diagnostics.spec.ts architecture-shell.spec.ts architecture-inspection.spec.ts
```

The browser suites use deterministic HTTP/model-owned and V-JEPA-style fixtures,
production preview plus the existing inspection dev harness, one Chromium worker,
SwiftShader, and isolated ports 45195/45196. They do not download models or claim
checkpoint acceptance. Full logs, traces and additional screenshots remain local
in generated test output.

## Behavior and visual checks

- Model-supplied provenance appears below the card header as a quiet disclosure.
  Its full explanation explicitly distinguishes structural/binding validation from
  equivalence to model code. Ambiguous interface mappings retain their exact
  message and affected component in the warning details and Model information.
- Response/graph copies deduplicate; identical messages at two distinct node
  locations remain separate. The authored fixture has three unique warnings.
- Dismissal removes only the inline findings. Session options → Model information
  retains complete messages, source context, capability, severity and counts,
  without node selection or a rendered graph. Opening it before graph retrieval
  explicitly says architecture is unobserved and makes no architecture request.
- A precise safe model-definition error retains its original message. Dismissing
  its error band preserves capability failure and backend-restart guidance.
- Thirty-three findings with long messages scroll inside a band bounded to 32%
  of the card. Both desktop and narrow layouts avoid document overflow. Keyboard
  disclosure/dismissal work; dismissal focuses the surviving card, and Escape
  from Session options restores its trigger.
- Notice/provenance disclosure, dismissal and Model information preserve the same
  mounted canvas, exact camera transform, layout count and request list. No tensor
  bytes, statistics, distributions or tokenization are requested by those actions.
- Dismissals survive explorer switches and repeated observations. A new graph
  makes its findings visible again. Store tests cover new severity, distinct
  locations, same-generation return, session/model/backend owner replacement,
  and late callbacks; shell tests also fence a delayed old architecture response.
- Existing inspection checks preserve progressive exact values, independent
  statistics/profiles, cancellation/resource cleanup, unavailable representations,
  StrictMode and graph-local recovery at both viewport sizes.

## Validation result

`npm run check`: passed API binding consistency, TypeScript, ESLint, **1,061 tests
in 52 files**, and production build. The build retains the existing advisory for
chunks larger than 500 kB. All **32 focused browser cases passed** in the final combined desktop/narrow run
(1.4 minutes).

The repository's UI, backend, API and application-acceptance workflows explicitly
exclude PRs targeting `codex/epic-issue-*`. Local evidence applies to this child;
the aggregate epic PR owns CI. No workflow or backend/API schema was changed.

## Captures

- [Desktop compact band](desktop-notices.png)
- [Narrow compact band](narrow-notices.png)
- [Desktop rediscovery](desktop-information.png)
- [Narrow rediscovery](narrow-information.png)
