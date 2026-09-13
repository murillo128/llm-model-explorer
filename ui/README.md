# Browser UI foundation

A separately buildable React + TypeScript + Vite application. Tensor Explorer and
Tokenizer Explorer are **placeholder slots**: there is no model discovery,
tokenization, or backend request yet. Selecting an explorer changes
the shell's workspace slot only; no routing library is needed at this stage.

The independent [exact-pixel WebGL2 renderer](src/rendering/README.md) is available
separately at `/renderer-demo.html` in both dev and production builds. It is not yet
connected to the explorer slots.

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
future API initialization. `App` receives only validated configuration. The shipped
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
backend to avoid browser mixed-content restrictions. This scaffold displays
“Configured · connection not checked”; it does not claim a successful connection.
No Python or backend static-file server is needed to build or serve the UI.

## Validate

```sh
npm ci
npm run typecheck
npm run lint
npm test
npm run build
npx playwright install chromium
npm run test:browser
```

`npm run check` combines type-checking, lint, unit/component tests, and production
build. On a fresh Linux host, `npx playwright install --with-deps chromium` also
installs Chromium's required system libraries. Playwright starts its own Vite
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

[Desktop screenshot](evidence/neutral-shell-desktop.png) and
[narrow screenshot](evidence/neutral-shell-narrow.png) are compact visual evidence.
See [validation evidence](evidence/validation.md) for the captured results.

## Source boundaries

- `src/api/`: runtime configuration; future explicit API clients live here.
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
