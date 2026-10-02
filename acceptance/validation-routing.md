# Validation ownership and change routing

`.github/scripts/validation_selector.py` is the single standard-library map used
by the four product workflows. Plans include complete paths, selected native
files, routine/full portfolio, potentially affected extended owners, Python
compatibility scope, reasons and exact tested revisions.

| Changed surface | Selected proof |
| --- | --- |
| Isolated existing backend `tests/test_*.py` | That pytest file. Known modules exporting helpers/oracles select all backend tests and genuine extended thresholds before the isolated-file rule. Other helpers and unknown/deleted/nested tests remain broad. |
| Existing matrix/tensor/renderer/distribution/magnifier/tokenizer/architecture browser spec | That native file plus complete UI fast checks. Unknown/deleted/nested specs remain broad. |
| UI unit test | Complete UI fast checks; ordinary tests remain included. |
| Split production spec | Its actual responsibility and corresponding real HTTP files. |
| `lora-reference.spec.ts` | Its extended browser cases plus TCP architecture, Base reference and independent LoRA/QLoRA reference owners. |
| `product-harness.ts` | All five tensor/tokenizer production spec consumers. |
| Isolated HTTP test file | That native pytest file; browser/build/Node/Chromium setup explicitly non-applicable. |
| Shared `acceptance/test_network.py` | All native HTTP tests, including model-root, export, embedding and actual-reference consumers; browser/build setup stays non-applicable. |
| Architecture UI production | Architecture component files, `architecture.spec.ts`, TCP architecture/native packages. |
| Tokenizer UI production | Tokenizer component files; transport, scientific, bindings and tokenizer-layout production files and their HTTP owners. |
| Rendering/matrix production | Matrix/native, tokenizer and architecture component owners plus complete integration. |
| Backend production/examples | Backend checks and real integration; backend production/support conservatively includes numerical threshold/allocator proof. Examples additionally select graph components. |
| API/generator/normative/shared/unknown/configuration | All applicable routine owners and potentially affected extended owners. |
| Operational Markdown/known runner infrastructure | Explicit product non-applicability; existing documentation/infrastructure checks. |

Maintain the finite shared-backend-module classification when adding imports of
helpers from a `test_*.py` module. Filename-based isolation applies only after
these shared owners; it does not prove a test module has no consumers.

`@extended` classifies genuine long/reference cases. Ordinary untagged tests run
by default. DPR 1 owns ordinary production cases; `@density` adds only meaningful
DPR-2 bridges. Desktop owns ordinary component cases; `@responsive` adds a narrow
bridge, and native scrollbars remain headed. Actual fractional DPR cases remain
native. Long architecture lifetime retains sixteen cycles; routine return/remap
uses three, with warmed object retention and exact identity assertions.

Routine traces are off, retries zero, with JSON, failure screenshots, logs and
compact resource/timing attachments. Request a diagnostic for one named failure:

```sh
cd ui
npm run test:acceptance -- scientific.spec.ts --project=dpr1 --grep 'source pixels' --trace on
```

Transport-only production cases use headless Chromium. Native focus, geometry,
pointer, IME and physical pixel owners remain headed with one SwiftShader worker.
There is no shared live service/cache state between independent tests. Separate
production/static and dev-harness servers are retained: both have actual consumers
and their small startup does not justify discovery or reuse machinery.

## Commands and portfolios

```sh
# Complete local proof: all non-retired cases, extended and configured references/CUDA.
bash acceptance/check.sh
# Complete integration, with all density projects; local contract checks included
# unless --main-ci delegates to the independent exact-target CI owners.
bash acceptance/check-integration.sh --main-ci
# Complete mandatory routine HTTP/build/discovery/browser/setup/teardown gate:
bash acceptance/check-integration.sh --routine --main-ci
# Use a complete changed-path array, or --github with native exact Git objects:
python3 .github/scripts/validation_selector.py --paths /tmp/changed-paths.json --output /tmp/validation-plan.json
bash acceptance/check-integration.sh --routine --main-ci --plan /tmp/validation-plan.json
python3 .github/scripts/validation_selector.py --plan /tmp/validation-plan.json --run browser --main
python3 .github/scripts/validation_selector.py --plan /tmp/validation-plan.json --run integration --main --extended
# Native backend selected files / threshold proof:
python3 .github/scripts/validation_selector.py --plan /tmp/validation-plan.json --backend-targets
(cd backend && uv run --locked pytest -m 'not extended')
(cd backend && uv run --locked pytest tests/test_tensor_analysis.py -m extended)
```

`--full` and manual workflow dispatch select all non-retired proof. An aggregate
epic PR selects the full mandatory **routine** portfolio plus affected extended
owners. A routine success is not reference/CUDA certification. Missing optional
capabilities remain skips; invalid supplied references and required-reference mode
still fail in their selected full/extended owners. No saving is assigned to skips.

Python 3.12 owns complete selected routine checks and version-independent
lint/format/mypy. Ordinary main Python 3.14 owns eighteen real installed-wheel,
CLI/server/session/dtype-shape runtime cases. Dependency/runtime/packaging inputs,
conservative fallback and manual mode select the fuller 3.14 wheel suite. This
reduces default cross-version confidence; it does not change supported versions.
Wheel installs have private temporary destinations, with locked dependency paths
read-only and explicit imported-package location/CPU-package assertions.

## Exact revisions, failure and CI truth

Plans use full Git history, the current PR base's merge-base and exact PR head or
push before/after, including all commits, deleted paths and both rename paths.
Tested merge identity must match the current base/head pair. Missing/truncated,
non-forward or unknown input selects broad routine and potentially affected
extended proof with a visible reason. Malformed explicit plans and empty required
groups fail. Native commands propagate failures; no empty-success flag is used.

Existing check identities, epic-child CI deferral, same-repository self-hosted
restrictions, least privilege and superseded-PR cancellation remain. The independent
API workflow owns schema/protocol plus exact fixture drift; the UI workflow owns
generated binding drift. Integration CI delegates these repeated checks on PRs
and main to those applicable exact-target gates. A missing/failed/cancelled selected
gate remains required and is never a product pass. `--main-ci` is this delegation,
not permission to waive final CI. Full local execution keeps both checks.

The application workflow retains its 25-minute limit. The combined routine gate
must complete within 1200 seconds, including selection and all setup/teardown;
900 seconds is an engineering target. Extended work is separately reported and
must pass when affected. No split command resets the routine budget. Builds are
fresh per checkout; no verdict, live cache or cross-workflow build reuse is added.
