# Target-aware card navigation

Fixture browser evidence recorded on 2026-09-16 with Node 24.14.0,
Playwright 1.63.0 and the repository's Chromium/SwiftShader configuration.
This proves UI behavior against authored graphs, not checkpoint acceptance.

The screenshots show the desktop `components` fixture at these exact targets:

| Screenshot | Active root | Header actions |
| --- | --- | --- |
| [Model](navigation-model.png) | Model | Layer 3 and its concrete children offer Explore component |
| [Isolated root](navigation-isolated-root.png) | `layer-3` | Layer 3 offers View in model; Attention and MLP offer Explore component |
| [Nested isolation](navigation-nested-isolation.png) | `layer-3.attention` | Attention offers View in model; Q/K/V operations offer Explore component |

The pointer regression selects `layer-3`, double-clicks the navigation control
on `layer-3.attention`, then uses Space to explore the non-expandable
`layer-3.attention.Q`. Each Back restores the preceding options (including
expansion, filters and repetition window), selection, camera and layout boxes.
Selecting Attention and pressing Layer 3's root control returns to `layer-3`
without collapsing it. Header key events stay local so Space does not select
the child through React Flow before the preceding snapshot is saved. The
leaf-operation test separately verifies exact
centering on return to Model.

Shared-view coverage starts with disabled structure-only controls, chooses
`layer-2.attention` while geometry still uses the `layer-0.attention` anchor,
and explores the exact `layer-2.attention.Q`. Back restores the shared instance
and camera. Source/derived resolver tests cover independent target identity,
root versus descendant actions and unavailable synthetic/context cards.
Existing connection tests retain real pointer/keyboard port, branch and edge
targeting; layout failure, obsolete callbacks and graph replacement remain fenced.

Reproduce from `ui/`:

```sh
npm run check
UI_TEST_PORT=46360 npm run test:browser -- --project=desktop --project=narrow \
  architecture-card-actions.spec.ts architecture-isolation.spec.ts \
  architecture-connections.spec.ts architecture-controls.spec.ts
```

Browser reports additionally attach all three screenshots for each viewport
(1440 × 900 and 390 × 844) and the exact target IDs. Only the compact desktop
screenshots are retained here; generated reports/traces stay outside Git.
