# Compact architecture initialization and collapse framing

The baseline is `edf521388c553f5e4b46cf9ac9f14b291cdb2831`.
These are deterministic browser fixtures, not downloaded checkpoints or claims
that every supported model layout has been accepted. The visual fixture models
an encoder/predictor interface; the training fixture uses model-owned provenance.

The implementation opens only the projected outer Model boundary. It assesses
one first-level layout using visible boxes/routes and the current graph body.
`minimumOverviewScale = 0.8`, `overviewInset = 16` CSS pixels and
`maximumOverviewScale = 1` are shared, directly tested criteria. A too-large
candidate is never rendered: a single collapsed layout replaces it. A collapsed
model with an exceptionally tall interface can exceed the viewport at the
readable minimum and remains accessible through ordinary pan/zoom.

The baseline opened every source root group, including Encoder underneath an
inserted Model boundary. Layout also reserved all port rows above descendants
while separately placing those rows along the sides. The changes address these
independent causes: projected-root expansion, actual header/summary padding,
and initial top alignment. No source connectivity or route identity changes.
The old global collapse changed projection alone; the new command fits its
completed layout once, retains hidden source selection and rejects obsolete work.

## Observed before/after

`comparison.json` records exact boxes, viewport, transform and layout counts.
`initialization.json` records candidate/final visible bounds, fit scale, expanded
sets and layout/camera invocation/completion counts for eight fixtures on both
viewports. The 5290 × 206 wide candidate fits at 0.210 on desktop and 0.067 on
narrow; the 278 × 2966 fan-out candidate fits at 0.249 and 0.156. Each takes two
layouts and one completed camera action; compact candidates take one of each.
Graph-body dimensions are 1142 × 771 on desktop and 388 × 494 on narrow screens.
The visual fixture's initial expanded set changes from Model + Encoder to Model
only on desktop, or the collapsed Model on narrow screens.

| Fixture / viewport | Visible outer height before → after | Initial zoom before → after | Layouts after |
| --- | --- | --- | --- |
| Visual / desktop | 610 → 282 | 1 → 1 | 1 |
| Visual / narrow | 610 → 164 | 0.65 → 1 | 2 |
| 22 inputs / desktop | 1774 → 710 | 0.65 → 1 | 1 |
| 22 inputs / narrow | 1774 → 596 | 0.65 → 0.8 | 2 |

The many-port expanded fixture's first child moves from y=1154 to y=90 **inside
its container**, with all real interfaces retained. The final initial diagram
starts 16 CSS pixels below the graph body's top, independently of header height.

| View | Before | After |
| --- | --- | --- |
| Visual desktop | [before](before-desktop-interface-visual.png) | [after](after-desktop-interface-visual.png) |
| Visual narrow | [before](before-narrow-interface-visual.png) | [after](after-narrow-interface-visual.png) |
| Many ports desktop | [before](before-desktop-interface-many.png) | [after](after-desktop-interface-many.png) |
| Many ports narrow | [before](before-narrow-interface-many.png) | [after](after-narrow-interface-many.png) |

## Reproduction and coverage

Use the pinned Node/npm versions in `ui/`. `npm run check` covers projection,
layout geometry, exact threshold boundaries, retained identities, and complete
source/edge conservation. `architecture-overview.spec.ts` attaches candidate and
final bounds, viewport, scale, layout/camera counts and screenshots for compact
real-root and synthetic-Model fixtures, wide fan-out, model-owned training,
visual interfaces, many ports and hybrid repeated stacks. It also covers the
collapse command, retained hidden selection, scope/model replacement, user
camera precedence and a zero-to-nonzero viewport.

The existing browser camera, card, controls, interface, isolation, browser,
component and weight-inspection suites exercise Back, exact instances, full
expansion, ordinary toggles and resize invariants, failure/retry and native
weight inspection. Run the architecture browser specs on both `desktop` and
`narrow`, with an isolated `UI_TEST_PORT` and output directory. No checkpoint
weights or remote downloads are required.
