# Tokenizer Explorer validation

## Reproduce

From `ui/`, using the pinned Node/npm versions:

```sh
npm ci
npm run check
npx playwright install chromium
npm run test:browser
```

For the focused interaction suite: `npm run test:browser -- tokenizer.spec.ts`.
The tests run at desktop 1440 × 900 and narrow 390 × 844 viewport sizes.

Local validation passed `npm run check` (176 unit tests, typecheck, lint, API
generation check, production build) and the complete browser suite (78 tests).

## Evidence and boundaries

- Unit checks cover code-point/UTF-16 conversion without normalization,
  transitive overlap grouping, typed special spans and absent/zero-width offsets.
- Browser tests use the production API client with contract-valid synthetic
  responses. Deferred transport promises intentionally ignore abort to verify
  reverse completion, A→B→A, late errors, and session/option generation fences.
- Browser typing, mid-string edits, caret/selection, undo/redo and actual
  clipboard copy/paste are exercised while decoration responses arrive.
  Composition events verify IME request suppression and completion handling;
  this does not automate an operating system's candidate-selection UI.
- Fixtures include empty source, leading/repeated spaces, tabs/newlines, accents,
  combining marks, non-BMP emoji, CJK, overlapping offsets, absent offsets,
  typed and inserted specials, and inert HTML-like input/native representations.
- Dense metadata spacing, long unmapped text, scrolling, keyboard focus, one visible
  source editor, black source and gray annotation styling have browser checks.
- Production shell tests exercise the actual typed HTTP request path with routed
  success and unsupported-tokenizer responses. Real model tokenization is reserved
  for the epic's final integration; this UI change contains no browser tokenizer.

`tokenizer.spec.ts` attaches standalone `tokenizer-explorer.png` and
`tokenizer-unicode.png` screenshots to the browser report. Compact standalone
desktop/narrow screenshots are retained beside this report. They show synthetic
fixtures, not claimed output of the reference model. CI uploads its full browser
report and attachments via the existing UI workflow.

## Local environment

Dependencies installed from the lockfile, with exact CodeMirror state/view/commands
versions added for document-independent decorations and editing history. A new Chromium download ran
out of host disk space; validation reused the already installed exact Playwright
Chromium revision 1243. Task-specific download/npm caches were removed. Chromium profiles use the task
RAM-backed temporary and report directories when host disk space is insufficient. Browser
tests used `UI_TEST_PORT=4273` to avoid concurrent worktrees' preview servers.
