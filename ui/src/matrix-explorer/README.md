# Matrix Explorer

Import `MatrixExplorer`, `MatrixSource`, and `MatrixUpdates` from this directory.
The component imports its own composition/inspection stylesheet; application
palette tokens are optional. Give its parent a bounded, shrinkable flex height.
A compact `header` React slot and `label` supply local context. Fetching,
operation progress, cancellation and domain errors belong to the parent.

```tsx
const source = useMemo<MatrixSource>(() => ({
  descriptor: { shape: [rows, columns], rank: 2,
    numel: rows * columns, logical_dtype: 'float32' },
  subscribe(updates) {
    return subscribeToResult({
      onValues: (chunk, elementOffset) => updates.values(chunk, elementOffset),
      onStatistics: (statistics) => updates.transfer({ statistics }),
    }); // Return an unsubscribe function; restart the prefix on every subscription.
  },
}), [resultGeneration, rows, columns]);

return <MatrixExplorer source={source} header={<span>Input result</span>}
  onRowSelect={setActiveRow} onCellSelect={setActiveCell} />;
```

Keep `source` stable across hover, contextual UI and status updates. Its object
identity is the delivery generation: replace it when the result/shape changes.
Even an old callback invoked during unsubscribe is ignored. The component consumes
arrays synchronously; producers may release their chunks after the call. Offset
units are elements, and each surface accepts a consecutive C-order prefix starting
at zero. No accumulated chunk list or extra scalar copy lives in this composition.
The renderer's optional CPU array remains the exact inspection copy.

Omit `distributions` for matrix-only mode. When authoritative profiles exist,
set `distributions: true` and deliver `updates.distribution('rows', counts, offset)`
for `[rows, 100]` counts and `'columns'` for `[100, columns]`. Profiles use the
existing shared-domain, 100-bin uint32 count contract; the parent validates and
splits transport metadata/sections. Missing counts stay visibly unavailable.
Deliver `updates.distributionDomain({ minimum, maximum })` from the authoritative
distribution metadata to label both rulers and true finite extrema. The current
contract bins the complete finite range. Null endpoints mean no finite values;
omitting domain delivery leaves the scale explicitly unavailable. Domain labels
are independent from `updates.transfer(...)` and are fenced with their source.
Do not compute substitute statistics or distributions in the browser.

An optional initial `source.transfer` and subsequent `updates.transfer(...)`
accept the renderer's statistics, anchors and slope without scalar reuploads.
Callbacks `onCellSelect`, `onRowSelect` and `onColumnSelect` report exact native
indices on hover/focus changes (also for pending cells whose readout is unavailable)
and `null` on clearing. They fire only when their coordinates change, not on every
progress/transfer refresh. Callbacks receive the latest committed parent props
without restarting the source. They do not introduce pinning or remapping.

Rank-1 descriptors retain the existing horizontal strip behavior without profiles
or rank-2 inspection. Empty descriptors allocate no scalar textures. WebGL failure
has local visible feedback and `onRenderingStateChange`; replace the source to retry.
A subscription that throws must clean up its own partially started operations; the
component releases its viewport and lets the parent error boundary handle the error.

`tests/matrix-explorer.html` is a standalone synthetic consumer with no application,
model, tensor tree, API transport or global application stylesheet. Its browser
tests cover DPR 1/2, exact values and callbacks, local scrolling, A → B → A,
StrictMode and texture/display/program/buffer/CPU lifetime. Tensor Explorer's
existing browser suite covers full profiles, pixel colors, magnification and
native scrollbars through this same component.


`highlightedRow` accepts optional external row context. It draws a row-only guide
without moving focus, scrolling, inventing a cell readout, resubscribing, or
uploading scalar values. A new external row supersedes prior local inspection
and clears its cell readout. A subsequent local pointer/focus/key interaction
can inspect a cell again; linked consumers clear their external row when that
local selection is reported. Clearing the external row does not clear a newer
local cell.

`PanelHeader` composes identity, information, summary, status, and right-side actions in a fixed 40px scientific toolbar. Pass it through `MatrixExplorer.header`; status changes do not restart the source. `InfoPopover` supplies hover/focus previews and click/tap/keyboard pinning, with pinned-only close, outside/Escape dismissal, and focus restoration. Neither primitive depends on tensor transport or tokenizer state.
