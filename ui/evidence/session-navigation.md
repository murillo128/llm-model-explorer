# Session navigation validation

The shell uses controlled, contract-valid API fixtures; it does not require model
files, a running backend, or unfinished explorer implementations.

## Reproduce

With Node 24.14.0 and npm 11.9.0, from `ui/`:

```sh
npm ci
npm run check
npx playwright install chromium
npm run test:browser
```

## Coverage

- Component/controller tests cover model creation, recovery, expiration, retry,
  failed deletion, tab storage denial, backend-specific persistence, empty and
  invalid catalogues, descriptor metadata, unsupported ranks 0 and 3, and both
  explorer slots. Inventory rank 1 and rank 2 mount the same composition boundary.
- Delayed promises deliberately ignore abort: superseded recovery, catalogue,
  inventory and creation results cannot replace current state. Unused newly
  created sessions are released by their exact returned IDs.
- A → B → A selection, tool switching, backend replacement and request-channel
  replacement fence old callbacks. Data and statistics channels remain independent;
  cleanup affects only the disposed consumer. Existing browser transport tests
  verify consumer-specific operation cancellation over actual fetch streams.
- Chromium checks the production build at 1440 × 900 and 390 × 844. Two pages in
  one browser context independently create, refresh/recover, expire and delete
  sessions. Runtime backend replacement sends no recovery request to the wrong
  backend. Malformed responses and backend error details do not reveal paths.
- Keyboard tests exercise native disclosures, full-name tensor selection, tool
  navigation, skip navigation, and visible focus. Both viewport sizes check shell
  overflow and capture `session-navigation.png` in ignored test results; CI uploads
  browser reports/screenshots as evidence.

## Local environment

The first npm install exhausted host disk space. Its partial dependency trees were
removed. Validation reused matching dependencies from the issue-11 worktree and
Chromium from the prior UI run; task caches remained local. No dependency versions
or lockfiles changed. Fresh-install validation is also required by PR CI.

Explorer rendering, tokenization mechanics and live backend integration remain
separate work. No fixture values enter the production application.
