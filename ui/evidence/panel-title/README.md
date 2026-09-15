# Viewer title and inventory tooltip evidence

Baseline: `a7d8da51bdc426dedc60482ee16f17fe60518caa` (pinned integration
`352def0e002f83defe267f99dfc963278c48b564` reconciled with the launcher's existing
main baseline). After: source in the commit containing this report.

Captured and visually inspected with Chromium 153.0.8010.12, SwiftShader WebGL2,
DPR 1, and 900 CSS-pixel browser height. These are deterministic Tensor Explorer
fixtures, not model captures. Both runs deliver identical `[576,1536]` scalar
values and fixture statistics, with histogram counts derived from those values.

| Width | Tooltip before / after | Information before / after |
| --- | --- | --- |
| 1178 | [Before](before-1178-tooltip.png) / [After](after-1178-tooltip.png) | [Before](before-1178-information.png) / [After](after-1178-information.png) |
| 1440 | [Before](before-1440-tooltip.png) / [After](after-1440-tooltip.png) | [Before](before-1440-information.png) / [After](after-1440-information.png) |

The title now spans the panel above the bordered scientific card. Its height is
40px. At both widths its top moves from 77px to 64px and its left edge from 69px
to 56px; scientific content remains at `(69,117)` with unchanged dimensions
(`1080×730` and `1342×730`). The existing scientific padding is now owned by the
content card. The selected logical path, metadata, information and Fit width
appear on one row. The popover still anchors directly below its trigger.

The before tooltip is covered by the title; the after tooltip is completely
readable. [Before measurements](before-geometry.json) and
[after measurements](after-geometry.json) include browser hit testing across the
tooltip, changing from covered to uncovered at both widths. The fix isolates the
Tensor working surface's local stacking and raises only its sibling rail above
that context. Tooltip width is bounded by the measured workspace; native modal
top-layer precedence is retained.

## Reproduction and regression coverage

With locked UI dependencies installed, run a dev server on an isolated port:

```sh
npm run dev -- --port 4378 --strictPort
PANEL_EVIDENCE_URL=http://127.0.0.1:4378 node scripts/panel-title-evidence.mjs after /tmp/panel-title-evidence
```

Run the same script against the baseline checkout for the before images.
`npm run check` covers API bindings, typecheck, lint, 788 unit tests and build.
The focused browser command is:

```sh
UI_TEST_PORT=4379 npm run test:browser -- tests/tensor-header.spec.ts tests/tensor-inventory.spec.ts tests/matrix-explorer.spec.ts tests/tokenizer-embeddings.spec.ts tests/tokenizer-layout.spec.ts tests/architecture-inspection.spec.ts --project=desktop --project=narrow
```

Assertions cover title/body sibling structure and padding, fixed geometry under
scroll/status/metadata changes, long paths at 280px, preview/pin/Escape/touch,
source/canvas/allocation/upload lifetime, empty and unavailable full-width titles,
tooltip hit testing on hover and focus, focus restoration, modal precedence,
metadata-only tensor states, and the Architecture weight modal's shell, native tensor binding, exact values,
graph camera, cancellation and resource release. Only the intended title/card
offset assertion changes in integrated acceptance; numeric or screenshot
oracles are not relaxed. The downstream Tokenizer two-panel integration remains
owned by its separate issue.

Standalone numerical fixtures reserve their existing viewer dimensions separately
from the new card's border/padding. Their `--fixture-viewer-width` and
`--fixture-viewer-height` inputs preserve the original scientific area; the default
is explicitly checked at `(16,56)`, width `min(600, viewport − 32)`, height `220`.
Magnifier hysteresis/placement, wrapped readouts, zoom, centering, overlay controls,
selection and exact-pixel assertions remain unchanged. Full application title
tests independently verify the real card, its padding and full-width title.
