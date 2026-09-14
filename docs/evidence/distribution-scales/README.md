# Distribution scale readability

The numeric audit found no disagreement between current binning and drawing:
`tensor_analysis.distributions` uses float64 index arithmetic over the full finite
range, keeps uint32 counts, and places constants in bin 50. The renderer samples
these bins directly and uses `log1p(count) / log1p(axis_length)` only for intensity.
Robust p01/p99 anchors affect matrix luminosity, not histogram coordinates.
A near-zero band with distant outliers is therefore a valid distribution, not a
reason to rescale the bin axis. No API or backend production change was needed.

The UI now preserves `domain_minimum`/`domain_maximum` from distribution metadata.
Both rulers and true-range metadata use those authoritative full-range endpoints.
Zero uses its numeric fraction of the domain, not a fixed central coordinate.
Constant, nonfinite, missing metadata, late statistics, source replacement and
auxiliary failure retain explicit independent semantics.

Reproduce with the pinned Node version and installed backend dependencies:

```sh
cd ui
npm run check
npm run test:browser -- --project=desktop --project=narrow distribution-scale.spec.ts tensor-explorer.spec.ts matrix-explorer.spec.ts matrix-inspection.spec.ts
xvfb-run -a npm run test:browser -- --project=native-scrollbars
cd ../backend
.venv/bin/pytest tests/test_tensor_analysis.py -k 'native_numerics or endpoints_and_disk_reuse'
```

Backend deterministic cases verify exact domains and histogram counts through
native computation and HTTP/cache reuse. New symmetric/asymmetric and 200-value
outlier cases complement constants, nonfinite values and float32 extremes.
Browser fixtures check all row/column counts and density pixels against known
bin indices, exact accessible endpoints, quarter-domain zero, independent late
robust statistics, and unchanged uploads/labels after inspection. Existing
reference fixtures check stable labels through both scroll axes at DPR 1/2;
resource and native-scrollbar tests preserve alignment and fixed profile depth.
Screenshots are generated in Playwright artifacts, not committed here.

The pinned execution base has native scrolling and no camera zoom/pan controls.
This change does not add those controls: domain delivery is independent from
viewport and inspection updates, leaving that separately designed work intact.
