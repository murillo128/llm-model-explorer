# Tensor workspace containment

Reproduce from `ui/` using the pinned Node version:

```sh
npm ci
npm run check
xvfb-run -a npm run test:browser
```

`tensor-explorer-scrollbars.spec.ts` runs with headed Chromium and requires a
positive native scrollbar width. At DPR 1 and 2, its pane cases cover desktop
(1440×900), constrained desktop (780×640), and stacked panes (390×640), with a
long inventory and tensors that fit, exceed height only, exceed width only, or
exceed both axes. It checks fixed document/body and application bars, independent
pane offsets, no overflow in scientific wrappers, reachable final rows/columns,
exact device-pixel canvas geometry, and matching distribution origins/extents at
both scroll extremes. Existing short-matrix/vector cases also check resizing and
final-column pixel values without new scalar allocations.

The native matrix reserves its vertical scrollbar track even when no vertical
scrolling is needed. Its width includes that track; rendering uses the actual
client box. This also avoids Chromium's shortened horizontal scroll range observed
with an empty `scrollbar-gutter: stable` gutter.

`tensor-explorer.spec.ts` verifies streamed scalar/distribution values and aligned
pixels across texture bands at DPR 1/2. `matrix-inspection.spec.ts` covers exact
hover, keyboard scrolling, and unchanged scalar resources. `tensor-navigation.spec.ts`
checks that the metadata popover fits the scientific pane and its final content
remains reachable even at 280×400.

Geometry JSON and screenshots are attached to the Playwright report and retained
by the UI CI artifact. Generated reports, browser traces, and screenshots are not
committed here. Local validation passed the 239 unit tests and all browser cases
through the full suite plus focused reruns after geometry-assertion and popover
adjustments; exact published-head CI is recorded on the PR.
