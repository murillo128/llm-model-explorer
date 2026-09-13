# Browser UI foundation

A separately buildable React + TypeScript + Vite application. Tensor Explorer and
Tokenizer Explorer have reusable composition slots. The shell discovers models,
creates and recovers sessions, and presents a logical tensor hierarchy. The
[reusable inline prompt editor](src/tokenizer/README.md) is connected as the
standalone Tokenizer Explorer and calls the session's typed tokenization endpoint.
See its [interaction evidence](evidence/tokenizer.md). Tensor rendering is connected
separately through its slot.

The independent [exact-pixel WebGL2 renderer](src/rendering/README.md) is available
separately at `/renderer-demo.html` in both dev and production builds. The default
Tensor Explorer slot also composes it with progressive API streams and distributions.

## Develop

Use Node **24.14.0** (`nvm use` reads `.nvmrc`) and npm **11.9.0**.
From a clean checkout:

```sh
cd ui
npm ci
npm run dev
```

Open the URL printed by Vite. Development binds to `127.0.0.1` by default; use
`npm run dev -- --host 0.0.0.0` when a trusted remote browser needs access.

## Runtime deployment configuration

The UI loads `runtime-config.json` before mounting the application and before any
API initialization. `App` receives only validated configuration. The shipped
`public/runtime-config.json` uses `http://127.0.0.1:8000` for local development.
This is the **browser's** loopback address, so replace it with a reachable backend
address for a remote deployment:

```json
{
  "backend_base_url": "https://models.example.net:8443"
}
```

Build and inspect the static output:

```sh
npm run build
npm run preview
```

Deploy the contents of `dist/` to any static HTTP server. Change
`dist/runtime-config.json` on that server to switch the backend **without rebuilding
or changing the JS/CSS assets**. Reload the page after a change, or use the retry
button after a configuration error. The file is fetched with `cache: 'no-store'`;
configure any intermediary/CDN to serve it without caching as well. Keep the
runtime file available as JSON, including when the server has an HTML fallback.
For a UI hosted under a subdirectory, build with `npm run build -- --base=/explorer/`;
the configuration file is loaded from that same base. Changing the UI mount path
is a build setting; changing the backend URL is always a runtime setting.

`backend_base_url` must be an absolute HTTP(S) URL. A backend path prefix is allowed.
The loader trims surrounding whitespace, uses URL normalization, and removes all
trailing slashes. Relative URLs, credentials, queries, fragments, invalid JSON, and
missing configuration produce an actionable error and prevent application startup.
The file is public: include no model filesystem paths, credentials, or model
management settings. Only `backend_base_url` is consumed. No environment-specific
backend address is compiled into JavaScript.

The backend is an independent process and can run on another computer. Its CORS
configuration must allow the deployed UI origin. For an HTTPS UI, use an HTTPS
backend to avoid browser mixed-content restrictions. Catalogue and session requests
report loading, empty, failure, and expiration states with retry controls.
No Python or backend static-file server is needed to build or serve the UI.

## Validate

```sh
npm ci
npm run typecheck
npm run lint
npm test
npm run build
npx playwright install --with-deps chromium
xvfb-run -a npm run test:browser
```

`npm run check` combines type-checking, lint, unit/component tests, and production
build. On a fresh Linux host, `npx playwright install --with-deps chromium` also
installs Chromium's required system libraries and Xvfb. The `native-scrollbars`
project uses headed Chromium to verify non-overlay scrollbars; on Linux without
a display run the suite with `xvfb-run -a` as above. Playwright starts its own Vite
preview server on port 4173; run it after building and with that port free.
Renderer browser fixtures also use a Vite dev server on port 4174. They exercise
the source renderer; set `UI_TEST_PORT` to move both servers (preview uses that
port and renderer fixtures use the following port). Tests exercise
real WebGL2 pixel/resource behavior at DPR 1 and 2, including progressive bands,
late statistics, context reconstruction and failure paths. The standalone demo is
also exercised through the production build. See [renderer evidence](evidence/renderer.md).

Browser acceptance runs against production assets at 1440 × 900 and 390 × 844.
It covers both deployed backend URLs on the same assets, empty/loading/error
states, retry, keyboard activation, skip navigation, visible focus, and shell
overflow. Tests restore the deployed config file after modifying it and therefore
run with one worker. Generated screenshots/reports/traces stay in ignored
`test-results/` and `playwright-report/`; CI uploads them as artifacts.

The original [foundation evidence](evidence/validation.md) documents the initial
placeholder shell. See [session navigation evidence](evidence/session-navigation.md)
for the current shell and its fixture coverage.

## Source boundaries

- `src/api/`: runtime configuration, typed API client, and progressive stream transport.
- `src/app/`: startup gate, React composition, explorer slots, and shared CSS tokens.
- `src/components/`: screen header, metadata, status/error text, button, and working
  surface primitives, using semantic HTML and visible focus.
- `src/rendering/`: React-independent scalar WebGL2 renderer and native-scroll adapter.
  ESLint prohibits direct React/application imports and JSX here.

The accepted [UI architecture](../docs/spec/ui/architecture.md) and
[visual language](../docs/spec/ui/visual-language.md) own durable UI decisions.

The [browser API boundary](src/api/README.md) provides generated contract types,
validated JSON methods, incremental LMEX decoding, and cancellable streams.
Run `npm run api:generate` after accepted OpenAPI changes; the normal check
command rejects generated contract drift.

## Session and explorer composition

The active session ID is stored in `sessionStorage`, keyed by normalized backend
URL. Refresh attempts GET session; a missing session is visibly expired and the
stored ID is removed. Other failures preserve the ID for retry. Storage denial
keeps the shell usable and explicitly disables refresh recovery. Switching models
creates a new immutable session; it does not mutate or delete the previous one.
Close session deletes only the active session. A superseded POST is allowed to
finish so its newly created, unused session can be deleted by its returned ID.

`App` accepts optional `slots.tensor` and `slots.tokenizer` component types. Each
receives `ExplorerContextValue` as props (also available via
`useExplorerContext`): the shared `ApiClient`, session/session ID, selected tensor
descriptor, selection lifetime, and a guarded `reportStatus` callback. Unsupported
ranks remain selectable for metadata inspection but never mount the tensor slot.
Slots remount when selection, tool, or session changes. Backend changes replace
the entire controller and client. No shader resources belong to app navigation.

A child owns its request channels and operation handles. Create a `RequestChannel`
from `selection` inside the child's effect. Call `channel.begin()` for each new
request in that channel, use `request.guard(callback)` for every async state update,
and pass `request.signal` to the API. This protects repeated A → B → A requests,
including callbacks already queued when transport aborts. Independent data and
statistics channels can run concurrently. Never guard by tensor name alone.

For a stream handle, register `request.onDispose(() => {
void operation.cancel().catch(handleCleanupFailure); })`. Report loading,
streaming, complete, cancelled, failed, or expired-session via guarded callbacks.
Dispose the channel on effect cleanup; this aborts its current request and cancels
only the registered consumer. Also release any child-owned renderer resources in
that cleanup. A selection disposal fences every attached channel synchronously,
before the new explorer can accept results. Slot components must still implement
their own React effect cleanup for unmount and Strict Mode effect replay.

The shell consumes only validated public descriptor fields and never prints raw
backend errors/details. Logical hierarchy comes from `TensorDescriptor.path`;
full supplied names distinguish duplicate leaf labels. Native HTML disclosure
controls and buttons provide keyboard navigation, visible focus and nested-list
semantics without introducing a partial ARIA tree keyboard model.

## Progressive Tensor Explorer

The default Tensor Explorer slot now composes the typed incremental API client
with the independent exact renderer. Select a rank-1 or rank-2 tensor to start
its complete tensor and statistics streams; nonempty rank-2 also starts the
independent distribution stream. Each operation has its own visible completion,
failure and cancellation state. Received prefixes remain inspectable and visibly
incomplete after failure/cancellation. `Cancel loading` releases only this view's
consumers. Selecting another tensor or explorer disposes its owned resources.

Rank-2 has three data surfaces: matrix, row profiles to the right and column
profiles below. Native matrix scrolling keeps both profiles aligned. Counts stay
uint32 and are converted to density only by the shader; statistics change small
transfer uniforms without reuploading tensor values. Empty/unsupported tensors
and unavailable WebGL2 resources have explicit shell states. Hover, magnifier and
selection chroma are reserved for separate interaction work.

`tests/tensor-explorer.html` is a development-only browser fixture exercising the
real React composition, client/decoder and GPU renderer with manually paced,
contract-valid streams. Its tests capture screenshots and assert values, pixel
geometry, independent outcomes and resource accounting. See
[evidence/tensor-explorer.md](evidence/tensor-explorer.md).
