# Matrix gesture-history test input

This is the test-only repair authorized by issue #160's second design repair.
The Matrix runtime, 180 ms rule, renderer, history, specification, dependencies
and CI configuration are unchanged.

## Retained uncontrolled evidence

The [UI CI trace](https://github.com/murillo128/llm-model-explorer/actions/runs/35118726147/artifacts/10457052996)
for head `0e257689889ff2139addb907e1e7269cf43a3ca6`, DPR 2, records these
camera values. Both scale axes are equal; the view remains 1080 × 932 device
pixels (540 × 466 CSS pixels).

| Observation | Logical X | Logical Y | Scale |
| --- | --- | --- | --- |
| Initial | 0 | 0 | 10.8 |
| First burst | 1.5090741532405358 | 0.9603199156985228 | 14.578475121820839 |
| Second burst | 2.5407946115464175 | 1.6261085513897071 | 19.678883044217507 |
| Escape 1 | 1.5090741532405358 | 0.9603199156985228 | 14.578475121820839 |
| Escape 2 | 1.18891673772219 | 0.7430729610763688 | 13.457628690343716 |

Each burst dispatched deltas `[-60, -50, -40]`; the second used `ctrlKey=true`
after a 220 ms wall-clock wait. The first locator evaluation took 328.703 ms,
the second 19.567 ms. These are whole-call durations, **not** per-event clock
gaps. The retained trace did not capture handler clock samples. The second
Escape's scale matches `10.8 * exp((60 + 50) * 0.002)` within floating-point
rounding, demonstrating an extra checkpoint inside the first burst. Slow
processing between handler samples is consistent with that observation but
its precise interval was not measured. The failure does not establish a
violation of the production timing contract.

## Controlled input and observable history

The test-local driver temporarily supplies `performance.now()` during each
synchronous burst and restores the original own-property descriptor (or
inherited method) in `finally`. It starts at an integer clock value so the
180 ms boundary uses exact integer differences, then advances explicitly
before each real DOM wheel dispatch. Animation frames and asynchronous work
use the restored clock. No camera, history entry or gesture token is injected.

| Schedule relative to T (ms) | Adjacent gaps (ms) | Expected Escape sequence |
| --- | --- | --- |
| Wheel 0, 40, 80; ctrl-wheel 300, 340, 380 | 40, 40, 220, 40, 40 | First burst view, initial, initial |
| 0, 179, 358 | 179, 179 | Initial, initial, initial |
| 0, 180, 360 | 180, 180 | Initial, initial, initial |
| 0, 40, 221 | 40, 181 | View after event 2, initial, initial |

Both long coalescing sequences prove that the interval is measured from the
preceding event rather than the gesture start. The split case captures the
real intermediate view after event 2. All comparisons include the complete
view, including origins, both scale axes, DPR, viewport and scroll geometry.
The original two-burst case still requires both bursts to increase scale and
all three Escape results to match exactly. The modifier does not define the
boundary; the explicit 220 ms gap does.

Each browser report attaches the sampled clock values, event deltas/modifiers,
DPR, complete before/after event views, and all three observed Escape views.
Independent native wheel, ctrl-wheel and two-finger touch coverage remains in
the full Matrix suite.

The targeted run passed all 60 cases: four schedules × three DPRs (1, 1.25, 2)
× five repetitions, with two workers and no retries. All 225 captured clock
reads exactly match their scheduled input. Representative DPR 2 observations
from that run are below; the named views refer to the complete corresponding
camera above, including the intermediate view exposed by the old failure.

| Schedule | Observed clock samples (ms) | Observed Escape views |
| --- | --- | --- |
| Two bursts | 367, 407, 447, 667, 707, 747 | First burst, initial, initial |
| 179 ms | 364, 543, 722 | Initial, initial, initial |
| 180 ms | 408, 588, 768 | Initial, initial, initial |
| 181 ms split | 364, 404, 585 | Intermediate (scale 13.457628690343716), initial, initial |

The controlled split reproduces the extra checkpoint through real zoom and
Escape behavior. The controlled coalescing cases return exactly to the initial
view. This distinguishes the temporal inputs without relaxing either outcome.

The combined desktop/narrow regression run passed 130 cases (66 Matrix and
64 Architecture), including the exact optional-template camera equivalence,
initialization/Center precedence and nested card navigation. `npm run check`
passed contract generation, typecheck, lint, all 956 unit tests (including the
three camera-history cases), and the production build.

Reproduce from `ui/` with Node 24.14.0 and the locked Chromium/SwiftShader setup:

```sh
npm run check
UI_TEST_PORT=47000 npm run test:browser -- --project=desktop --workers=2 \
  --retries=0 --repeat-each=5 matrix-zoom.spec.ts \
  --grep 'coalesce into separate|wheel gesture timing'
UI_TEST_PORT=47010 npm run test:browser -- --project=desktop --project=narrow \
  --workers=2 --retries=0 matrix-zoom.spec.ts architecture-camera.spec.ts \
  architecture-card-actions.spec.ts architecture-isolation.spec.ts \
  architecture-connections.spec.ts
```

This repair changes only the temporal browser-test stimulus. The preceding
application acceptance evidence still applies to product behavior; the normal
PR workflows provide fresh full UI and application acceptance results for the
published candidate. Fixture success does not establish checkpoint or CUDA
acceptance.
