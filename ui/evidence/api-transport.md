# Typed API and LMEX interoperability evidence

Validation uses the current accepted OpenAPI document and
`api/fixtures/conformance.json`; no model files or running production backend
are required. The HTTP browser server is a controlled fixture producer.

- Node 24.14.0 / npm 11.9.0: clean `npm ci` succeeds; UI TypeScript 6.0.3 and
  generator TypeScript 5.9.3 resolve independently without peer overrides.
- `npm run check`: generation drift check, typecheck, lint, 158 unit tests,
  and production build pass.
- `UI_TEST_PORT=44111 npm run test:browser`: 18 tests pass across desktop and
  narrow Chromium projects. A port override avoids collisions with other local
  worktrees; default CI behavior still uses port 4173.
- Every shared valid/invalid wire case has the expected outcome under every
  single split position, coalesced input, one-byte chunks, Fibonacci chunks,
  and five seeded random chunkings, including unaligned backing-buffer offsets.
  Every shared schema case passes its expected validity and cross-field checks.
- A 2 MiB DATA frame is processed in borrowed 64 KiB segments. Callbacks retain
  the network backing buffer identity; no full-frame staging allocation exists.
  The nearly 4 GiB truncated fixture rejects without allocating its DATA length.
- Browser fetch observes operation headers before body delivery, data before
  the final chunk, and success only after EOF. Tests cover header CORS exposure,
  explicit DELETE preflights, duplicate cancellation, AbortController, socket
  closure, and continued success of a second concurrent consumer.
- Two consecutive generator runs produce byte-identical output. SHA-256:
  `types.ts`: `647e1cffa037196e18fbaf6072d1e4b68fde4fe4f20b786b7540e8266ea7fa35`;
  `schemas.json`: `0d8af3ba2fb76ff4bc68b3bccf6a0eee3b5811e2839f7c68e33de80e0b90544d`.

Destination storage and provisional-render invalidation remain consumer
responsibilities, documented in `ui/src/api/README.md`. These tests establish
wire and transport interoperability, not backend computation correctness.
