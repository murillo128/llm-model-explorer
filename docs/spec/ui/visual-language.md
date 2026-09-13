# UI visual language

## Authority and references

This document defines the cross-cutting visual language for the browser UI. It owns shell styling, typography roles, spacing, neutral surfaces, common interaction accents, and the composition rules shared by explorers. Component-specific behavior remains owned by the dedicated UI specifications; when a visual rule here conflicts with a more specific accepted behavior in `tensor-explorer.md`, `tokenizer-explorer.md`, or another future component specification, the component specification wins.

The current visual reference is the light Matrix Inspector direction developed in Canva and materialized on the Miro dashboard:

- Canva design `DAHVAU_Dc3E`, based on the `LLM Model Explorer — Matrix Inspector Light_3.png` exploration.
- [Miro board — llm-model-explorer wireframes](https://miro.com/app/board/uXjVHnoDEYY=/).
- [Miro — current spec-compliant materialized dashboard](https://miro.com/app/board/uXjVHnoDEYY=/?moveToWidget=3458764683570368527).

Mockups are visual references. Written specifications remain normative for behavior, tensor semantics, exact geometry, and interaction.

## Design principles

The UI is **light, editorial, technical, and data-first**. The tensor or operation being inspected is the visual subject; application chrome is deliberately quiet. Use warm neutral page backgrounds, white working surfaces, graphite text, hairline borders, restrained shadows, and compact technical metadata.

Whitespace separates concepts more often than cards do. Do not turn every region into a rounded dashboard tile. A panel exists when it provides a real surface boundary, such as an editor, matrix viewport, magnifier, or inspector strip. Titles and metadata normally sit directly on the page background.

The interface should feel precise rather than decorative. Dimensions, tensor shapes, coordinates, token IDs, formulas, and model metadata use stable alignment and monospace typography where that improves reading. Scientific data must never be visually altered merely to make the layout prettier.

## Core tokens

The following values are the implementation baseline for ordinary UI chrome. They are project tokens derived from the accepted light direction; they are not claims about the exact internal color values of the Canva file.

```css
:root {
  --ui-bg: #f5f3ee;
  --ui-surface: #ffffff;
  --ui-surface-soft: #fbfaf7;
  --ui-text: #20201d;
  --ui-text-muted: #74716b;
  --ui-text-faint: #929089;
  --ui-border: #d7d3ca;
  --ui-border-strong: #cfcac1;
  --ui-accent: #c87922;
  --ui-accent-soft: #f3e1c9;
  --ui-shadow: 0 1px 2px rgb(43 39 31 / 4%);
  --ui-shadow-float: 0 10px 25px rgb(43 39 31 / 14%);
  --ui-app-bar-height: 52px;
  --ui-status-bar-height: 28px;
}
```

`--ui-accent` is the default warm amber interaction accent. It is suitable for cursor, hover, pinned-selection, active coordinate, and focus cues. It must not be used to encode scalar tensor magnitude. Tensor luminosity and semantic chroma remain governed by `rendering.md` and `tensor-explorer.md`.

Do not introduce a second general-purpose brand accent merely to make a screen more colorful. Additional hues require a domain meaning that cannot be expressed by the existing neutral/amber system and should be specified by the owning component.

## Typography

Use a neutral sans-serif for product text and a neutral monospace for technical metadata. The exact Canva font identity is not normative; the implementation should prefer an available UI stack rather than bundling a font solely to imitate a mockup. A suitable baseline is `Inter, system-ui, sans-serif` for UI text and `ui-monospace, SFMono-Regular, Menlo, monospace` for technical text.

Use these roles consistently:

- product identity and global navigation: `13 px`, semibold for identity and the active explorer;
- status bar: `12 px` text with `11 px` technical metadata;
- model/path metadata: `13 px` monospace with `20 px` line height, muted;
- section label: `11 px`, bold, uppercase, approximately `0.1em` tracking;
- ordinary body/control text: `14–16 px`;
- compact coordinates, IDs, and helper text: `10–12 px` monospace.

Avoid oversized marketing typography, multiple display faces, or a terminal-like wall of monospace. Monospace is a semantic tool for technical values, not the default font for the entire application.

## Application shell and panel geometry

The application occupies the browser viewport (`100dvh`). Its grid reserves exactly
`--ui-app-bar-height: 52px` for the top bar and `--ui-status-bar-height: 28px` for
the bottom status bar; the workspace frame receives all remaining height. These
80 px leave about 90.5–91.1% of an 840–900 px desktop viewport for the workspace.
Normal application states must not produce document/body scrolling. Scrollable
workspace panels have explicit shrinkable bounds; large scientific surfaces use
ordinary panel scrolling without scaling, resampling, or changing exact pixels.

The workspace frame uses `12 px` vertical and `16 px` horizontal padding on
desktop, and `8 px` padding at narrow widths. Working panels use a `1 px` hairline
border, approximately `8–10 px` corner radius, and at most the subtle base shadow.
Floating inspection surfaces may use the stronger floating shadow.

## Global application bar and status bar

One compact global bar owns product identity, Tensor Explorer / Tokenizer Explorer
navigation, the current model selector, adjacent refresh-model control, session
state, and secondary session actions. The active navigation item is the explorer
title. Do not repeat it as a large page heading or workspace title. The selector
is the primary visible model identity; do not repeat the model as a subtitle or
in the status bar. A workspace may retain an accessible explorer name without
another visible title.

Refresh models is an icon control with an accessible name and tooltip. Close
session belongs in the session-options overflow panel. This panel also exposes
backend connection context, session identity, and raw model metadata on demand.
Use semantic buttons and a labelled selector, visible keyboard focus, and an
expanded-state disclosure. Escape dismisses the panel and returns focus to its
trigger; moving focus outside also dismisses it.

The fixed bottom bar presents compact inline session state and, when supplied,
public architecture/model type, humanized decimal model size (for example
`272.4 MB`), and tokenizer availability. Raw byte counts and parameter counts
belong in the on-demand details. Loading, empty, error, retry, and storage notices
remain readable in a bounded workspace notice area, without expanding the chrome.

At constrained widths, shorten visible navigation labels to Tensor / Tokenizer
while retaining their full accessible names. Product identity and secondary
metadata remain available in session options when omitted from the compact bars.
Keep the model selector, refresh action, navigation, and overflow trigger usable
in one row. Never restore a tall hero or scale scientific data to fit the chrome.

Filesystem paths must never be shown. Model identity and tensor/module context
come from public descriptors defined by the application contracts.

## Tokenizer surface

The reusable tokenizer component follows `tokenizer-explorer.md` exactly. The editable prompt and its tokenization are one surface; do not repeat the same prompt in a second token row.

Ordinary tokens are rendered inline as gray `[` + black source text + gray `]`, with the token ID as muted gray metadata associated with that span, normally below it. Source whitespace remains inside the token span. Special tokens are gray because they are not user-entered text. Token identity is not represented with a rainbow palette or a different chip color per token.

When a compact inference layout needs less vertical space, compress this same component rather than replacing it with token pills or a parallel presentation.

## Matrix Explorer primitive

The reusable Matrix Explorer is the central scientific surface for rank-2 data and inherits its behavioral contract from `tensor-explorer.md` and `rendering.md`.

For a rank-2 tensor, the visual composition is:

- the exact matrix surface;
- a `100 px`-deep row-distribution surface immediately to the right;
- a `100 px`-deep column-distribution surface immediately below;
- a small visual gap, normally about `8 px`, between aligned surfaces.

The lower-right corner is not a fourth dashboard panel. It may remain empty; if a tiny bin-count note is genuinely useful, render it as quiet microcopy rather than a boxed control.

The matrix itself is not styled using ordinary CSS palette tokens. Each value maps to exactly one rendered pixel and scalar value controls luminosity. The page, borders, labels, and histogram chrome use the neutral UI system around that surface.

## Hover, selection, and magnification

Warm amber is the standard interaction chroma. Hovering a tensor cell links the corresponding row, column, and distribution scanlines while preserving the original scalar-derived luminosity. Do not paint opaque guide lines over tensor data and do not recolor a second copy of the tensor.

The primary pointer marker over a matrix is a small amber cross with a center gap. The hover readout uses compact monospace coordinates/value text. The current magnifier reference is approximately `170 × 170 px` and shows a `9 × 9` source neighborhood with nearest-neighbor/pixel-preserving enlargement. It is an inspection aid, not general zoom.

Pinned selection reuses the same amber semantics and adds persistence through a visible state/readout, not through a new unrelated color system. Keyboard focus must remain visible independently of chroma alone.

## Reusable operation composition

Computational screens should compose the same primitives instead of inventing bespoke panels for every operation.

**Inference Explorer** places the reusable inline tokenizer surface above the downstream scientific surface. The full version may expose richer status/hover detail; the compact version reduces spacing but does not introduce a second tokenizer representation.

**RMSNorm** places input and output Matrix Explorer surfaces in a clear before/after relation. The active token row maps one-to-one across both surfaces; the operation/formula sits between or below them as secondary information. The same amber interaction state links corresponding coordinates.

**Matrix Multiply Explorer** composes `A × B = C`, with `C` remaining an ordinary Matrix Explorer result. Hovering `C[i,j]` links row `i` in `A`, column `j` in `B`, and the exact result cell using the same semantic chroma while the scalar luminosity of every value stays unchanged. The formula/dot-product inspector is secondary to the matrices.

These compositions may be wider than a normal viewport. Preserve native matrix geometry and use normal scrolling rather than scaling all matrices down to fit.

## Loading, streaming, and incomplete data

The UI must be useful before all derived information has arrived. When tensor bytes stream progressively, render available pixels as soon as possible. Distribution/statistics surfaces may appear or refine later. Never fabricate values, histogram data, or a fully populated matrix while bytes are missing.

Loading treatment belongs to the shell, not the data. Use restrained text/progress state around an incomplete scientific surface; do not replace unknown tensor regions with decorative fake heatmaps that could be mistaken for real values.

## Accessibility and legibility

Primary text must have strong contrast against the light background. Muted metadata remains readable rather than becoming ornamental gray. Tiny `10–11 px` text is reserved for genuinely secondary coordinates/IDs; instructions and actionable labels should be larger.

Color must not be the only communication channel for an important state. Hover/pinned selection includes coordinates or other text state in addition to amber chroma. Focusable controls use a visible focus treatment. Scientific luminosity must remain readable for users who do not perceive the semantic hue.

Avoid dense overlays on the matrix. Inspection UI should move or flip sides when necessary rather than covering the data being inspected.

## Do not

Do not use a dark terminal-style application shell around the scientific view. Do not add saturated card backgrounds, gradient chrome, large decorative icons, or heavy shadows. Do not assign arbitrary colors to tokens. Do not rescale matrices to fit cards. Do not use general zoom/pan as a substitute for scrolling. Do not draw opaque row/column guide lines across tensor data. Do not modify tensor values to implement visual state. Do not maintain a second recolored tensor merely for hover/selection.

## Implementation boundary

React owns composition, semantic HTML, controls, text, and panel layout. The reusable WebGL2 renderer owns exact tensor pixels and shader-based visual transformations. The visual tokens in this document style the application shell; they must not leak into authoritative tensor mathematics.

Semantic interaction colors should reach the renderer as small state/uniform parameters. The renderer derives the final chroma at draw time while continuing to read the same immutable scalar tensor representation. This keeps the visual system consistent with the project's memory and one-value-to-one-pixel invariants.
