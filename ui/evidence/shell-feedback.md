# Shell connection and notification evidence

## Reproduce

Use the locked Node/npm versions (`ui/.nvmrc`, `ui/package.json`), install with
`npm ci`, and run from `ui/`:

```sh
npm run check
UI_TEST_PORT=24194 npm run test:browser -- --project=desktop --project=narrow \
  tests/shell-feedback.spec.ts tests/shell-viewport.spec.ts tests/shell.spec.ts \
  tests/sessions.spec.ts tests/api-transport.spec.ts tests/tensor-explorer.spec.ts \
  tests/tokenizer-model-switch.spec.ts
```

The browser checks use controlled HTTP and stream fixtures, Chromium/SwiftShader,
and isolated production-preview/dev-server ports. No model download or real
outage is required. `shell-feedback.spec.ts` writes `operation-feedback.png` to
each project's test output directory; screenshots and full logs remain generated
local artifacts.

## Results

`npm run check` passed: generated API consistency, TypeScript, ESLint, 1,040
unit tests in 49 files, and the production build. The selected browser suites
covered 62 desktop/narrow cases: 60 passed in the combined run; the two catalogue
refresh cases passed on focused rerun after updating their obsolete top-strip
assertion. Their renderer/prefix/cancellation assertions were retained. The build
reports the existing advisory about chunks larger than 500 kB.

## Observed coverage

- Initial connecting, failed transport, repeated explicit retries, recovery and
  reachable HTTP/protocol/capability failures are distinct. Connection-only events
  create no toast. Cancellation remains a separate outcome.
- One failed stream creates at most one notification, including duplicate event
  delivery after dismissal. The visible stack is bounded at three. The local
  operation failure remains visible after its toast closes.
- Failed close retains its session and exposes a keyboard-operable Retry close;
  dismissing feedback preserves footer state and the existing close action.
  Expiration keeps persistent fresh-session guidance and reports reachable HTTP.
- Informational dismissal pauses on hover/focus. Errors do not expire. Model/backend
  replacement clears timers and fences obsolete requests, including an ignored
  abort and late failed close. StrictMode recovery completes correctly.
- Desktop and narrow browser assertions retain the 52px app bar, 28px footer,
  bounded workspace and no body/document overflow, including 280px width.
  Connection changes preserve adjacent metadata geometry. A zoomed matrix retains
  its exact camera, canvas identity, renderer count, scalar allocations, numeric
  request count and cancellations through toast display/dismissal and retries.
- Existing progressive-prefix, explicit/shared cancellation, session isolation,
  and tokenizer model A→B→A regression checks remain included.

The accepted workflow configuration defers UI/application CI for PRs targeting
`codex/epic-issue-*`; the aggregate epic PR owns those CI gates. This report records
local fixture evidence, not full local-model acceptance.
