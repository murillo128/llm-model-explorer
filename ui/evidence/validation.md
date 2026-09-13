# UI foundation validation

Validated on Linux with Node 24.14.0 and npm 11.9.0 using the committed lockfile.
The shell is a foundation only: both explorer slots explicitly report that they
are not implemented. No backend connection or model operation is claimed.

| Check | Observed result |
| --- | --- |
| `npm run typecheck` | Passed, strict TypeScript including browser tests and tooling |
| `npm run lint` | Passed, zero warnings |
| `npm test` | 32 unit/component tests passed |
| `npm run build` | Static production build passed |
| `npm run test:browser` | 14 Chromium checks passed across desktop and narrow layouts |
| Runtime URL replacement | Real `dist/runtime-config.json` rewritten between two backend URLs; JS/CSS SHA-256 hashes unchanged; no backend requests issued |
| Startup failures | Missing file, missing/invalid URL, and malformed JSON show errors; retry recovers |
| Loading/empty states | Loading blocks explorer composition; ready slots show an explicit empty state |
| Keyboard | Skip link, explorer button focus/activation, visible outline, and workspace focus passed |
| Layout | No page overflow at 1440 × 900 or 390 × 844; screenshots visually inspected |
| Boundary inspection | `src/rendering/` contains only its boundary README and no React code; no model paths or filesystem API dependency in application source |

The screenshots are copied from the production-build browser acceptance run:

- [Desktop shell](neutral-shell-desktop.png)
- [Narrow shell](neutral-shell-narrow.png)

The displayed backend IP is a documentation/test address, not a live service.
Generated test reports, traces, dependencies, and production bundles are excluded
from Git. CI repeats lockfile installation and all checks from a clean checkout,
and retains browser evidence as an artifact.
