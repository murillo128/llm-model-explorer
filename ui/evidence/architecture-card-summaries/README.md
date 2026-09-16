# Architecture card summaries and direct matrix inspection

The canvas now shows a supplied formula once beneath the title, exact own tensor
references, and explicitly mapped computational scalars. Names retain every path
segment after an unambiguous source module prefix; hover/focus and accessible
identity retain the original name. The matrix icon opens the existing inspection
modal with that exact parameter selected. Full descriptors and provenance stay
in the ordinary inspector.

The authored card fixture uses `LayerNorm(x; weight, bias, epsilon)`, a deliberately
nondefault `epsilon = 0.00003`, rank-1 weight/bias vectors, and a biased linear
operation with a rank-2 weight. These are UI fixtures, not checkpoint acceptance.
The application fixture matrix is the second repeated instance's asymmetric
3 × 2 tensor, with independently delivered statistics and histograms.

| Capture | Desktop (1440 px) | Narrow (390 px) |
| --- | --- | --- |
| Layer norm, dimensions off | [Capture](norm-desktop-dimensions-false.png) | [Capture](norm-narrow-dimensions-false.png) |
| Layer norm, dimensions on | [Capture](norm-desktop-dimensions-true.png) | [Capture](norm-narrow-dimensions-true.png) |
| Linear, dimensions off | [Capture](linear-desktop-dimensions-false.png) | [Capture](linear-narrow-dimensions-false.png) |
| Linear, dimensions on | [Capture](linear-desktop-dimensions-true.png) | [Capture](linear-narrow-dimensions-true.png) |
| Expanded group | [Capture](expanded-group-desktop.png) | [Capture](expanded-group-narrow.png) |
| Direct matrix inspection | [Capture](matrix-desktop.png) | [Capture](matrix-narrow.png) |

`architecture-summaries.spec.ts` checks actual DOM bounds, input/output hit targets,
unchanged camera/layout during hover and focus, full long text, the exact `+3 more`
count, and group-owned rows/port labels clear of children. Captures are secondary
to those assertions. Narrow overview captures use the graph's explicit Fit view;
individual cards remain available at readable scale through normal navigation.

`CardSummary.test.tsx` covers absent/nondefault/unknown constants, biased and
bias-free formulas, exact duplicate-reference removal, qualified/unqualified
names, missing/ambiguous module prefixes, own group parameters, long lists,
rank-1/rank-2 shapes and typed unavailable explanations. Inspection tests cover
independent activation, no numeric calls on summary hover, exact inventory IDs,
rank-1 statistics without distributions, rank-2 histogram delivery, Escape/focus,
close/model replacement, same-ID late callbacks and independent consumers.

Changing card sizes changes where shared fan-out segments end. The connection
browser test therefore uses the existing independent SVG geometry sampler to
choose an exclusive branch; a path's DOM owner alone does not establish that a
segment is exclusive. Shared trunks still highlight all represented branches.

## Local validation

`npm run check` passed API binding verification, TypeScript, ESLint, all 978 unit
tests in 41 files, and the production build. The subsequent completed-layout
annotation guard also passed typecheck, lint, build and the summary browser cases.

The focused desktop/narrow run passed 56 cases; its two exclusive-branch pointer
samples were corrected as described above. A focused rerun passed both corrected
connection cases and all four summary cases. Together these cover all 58 cases in
`architecture-summaries.spec.ts`, `architecture-inspection.spec.ts`,
`architecture-card-actions.spec.ts` and `architecture-connections.spec.ts`.

Reproduce from `ui/` with the locked Node and npm dependencies:

```sh
npm run check
UI_TEST_PORT=4260 npm run test:browser -- \
  tests/architecture-summaries.spec.ts tests/architecture-inspection.spec.ts \
  tests/architecture-card-actions.spec.ts tests/architecture-connections.spec.ts \
  --project desktop --project narrow
```

All six focused real-backend production checks passed at DPR 1: four deterministic
model-family graphs (SmolLM2, Qwen3, Qwen3.5 and V-JEPA 2), progressive close and
cancellation, and distinct shared-instance weights with cancellation. Available
weights open with one matrix-button activation; unavailable higher-rank buttons
retain their supplied reason and do not open or fetch. Their full descriptors
remain reachable in ordinary inspection. Native and packed logical scalar
oracles, independent profiles, exact tensor IDs and resource-release assertions
remain in place.

With the backend environment prepared as in `acceptance/README.md`, run:

```sh
UI_TEST_PORT=4260 xvfb-run -a npm run test:acceptance -- architecture.spec.ts \
  --project dpr1 \
  --grep 'deterministic production.*full graph|close during progressive|shared structure production'
```

The local sandbox's Xvfb crashed while loading the installed NVIDIA EGL vendor.
Selecting Mesa with
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json`
allowed these headful tests to run; no application check was disabled. The local
backend used the prepared Python 3.12 environment with this worktree's
`backend/src` on `PYTHONPATH`. Models and generated backend artifacts stayed in
temporary fixture directories.
