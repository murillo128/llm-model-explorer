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
}
```

`--ui-accent` is the default warm amber interaction accent. It is suitable for cursor, hover, pinned-selection, active coordinate, and focus cues. It must not be used to encode scalar tensor magnitude. Tensor luminosity and semantic chroma remain governed by `rendering.md` and `tensor-explorer.md`.

Do not introduce a second general-purpose brand accent merely to make a screen more colorful. Additional hues require a domain meaning that cannot be expressed by the existing neutral/amber system and should be specified by the owning component.

## Typography

Use a neutral sans-serif for product text and a neutral monospace for technical metadata. The exact Canva font identity is not normative; the implementation should prefer an available UI stack rather than bundling a font solely to imitate a mockup. A suitable baseline is `Inter, system-ui, sans-serif` for UI text and `ui-monospace, SFMono-Regular, Menlo, monospace` for technical text.

Use these roles consistently:

- product eyebrow: `12 px`, bold, uppercase, approximately `0.14em` tracking;
- screen title: approximately `30–32 px`, semibold, compact line height;
- model/path metadata: `13 px` monospace, muted;
- section label: `11 px`, bold, uppercase, approximately `0.1em` tracking;
- ordinary body/control text: `14–16 px`;
- compact coordinates, IDs, and helper text: `10–12 px` monospace.

Avoid oversized marketing typography, multiple display faces, or a terminal-like wall of monospace. Monospace is a semantic tool for technical values, not the default font for the entire application.

## Page and panel geometry

The current desktop composition reference is approximately `1736 × 872 px`. It is a design reference, not a fixed application viewport. The shell should adapt to the browser while preserving the exact native pixel geometry of scientific surfaces.

At desktop sizes, use approximately `40 px` horizontal page padding and `24–30 px` top padding. Primary vertical gaps are normally `14–20 px`; aligned scientific surfaces use tighter gaps around `8 px`. Working panels use a `1 px` hairline border, approximately `8–10 px` corner radius, and at most the subtle base shadow. Floating inspection surfaces such as the magnifier may use the stronger floating shadow.

Do not fit, scale, or shrink a tensor surface to satisfy these page margins. When exact data geometry is larger than the available space, the containing region scrolls as defined by `rendering.md`.

## Common screen header

Every explorer screen starts with the same quiet hierarchy:

1. `LLM MODEL EXPLORER` eyebrow;
2. one concise screen title;
3. one muted technical metadata line containing only context useful for the current view.

A small status chip may sit on the right when the state is actionable or materially informative, such as streaming, live tokenization, pinned inspection, or operation status. Do not create a permanent toolbar of decorative badges.

Filesystem paths must never be shown. Model identity and tensor/module context come from public descriptors defined by the application contracts.

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
