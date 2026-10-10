# Automatic model hot reload acceptance

Issue #307 exercises the integrated #304–#306 contracts through real temporary
Safetensors files, the production backend/SSE routes, and the built browser UI.
`ui/acceptance/hot-reload.spec.ts` runs at DPR 1 with the existing different-origin
static server, native EventSource, headed Chromium and SwiftShader. No successful
notification, graph response, token result or numeric stream is fabricated.

## Reproduce and test ownership

After the dependency setup in [README.md](README.md):

```sh
npm --prefix ui run build
UI_TEST_PORT=23070 xvfb-run -a npm --prefix ui run test:acceptance -- \
  hot-reload.spec.ts --project=dpr1
```

Each case owns its backend process and temporary model/cache directories. The
fixture uses `reload/a` and `reload/b` logical IDs, tiny F32 weights, a deterministic
WordLevel tokenizer and a model-owned graph with nested parallel encoder stages.
The graph explicitly describes two uses of the same weight; repetition alone does
not imply weight sharing. The importer validates the exact bindings. This is
native Safetensors/model-defined coverage, not full reference-checkpoint or CUDA
acceptance, and does not prove mathematical equivalence to an executed model.

The compact test plan and independent oracles are:

| Boundary and plausible defect | Action and observable proof | Existing primary owner reused |
| --- | --- | --- |
| Disk/SSE/browser restoration accidentally retains old IDs or resets to stage zero | Atomically edit stage 1's projection label/formula with stable author IDs. Require a new session/revision and disjoint graph record IDs, retained stage-1 scope, expanded interior, selected projection, normalized anchor/zoom, Back, and a new-session weight request. A separate old session must return 409 `model_content_changed`. | #306 producer/remap matrices and component-camera proof |
| Catalogue-wide invalidation or unsafe fallback | Edit B while A's weight inspection is open; wait for real B observation and require A's session, events, camera and numeric requests unchanged. Remove the selected author ID with its parent intact and a same-label replacement; require the stage-1 parent. Corrupt/repair config and architecture separately; require waiting, same-ID recovery, and usable tensor data when only architecture is malformed. | #304 observer/admission/TCP tests; #306 ambiguity/namespace matrices |
| Reconnect loses a change, tabs share ownership, or delayed pinning overrides intent | End actual SSE responses and temporarily drop reconnect requests, edit A, then let native EventSource reconnect. Require exactly one replacement. Two contexts share one observer but own separate sessions. Hold real POST responses while switching model/explorer or closing; require abandoned sessions deleted and the other context preserved. | #305 controlled controller race matrix |
| Numeric consumers retain stale values or overwrite current input | Keep exact tensor identity while changing its twelve values from 1 to 9. Keep the mounted editor, selection and pending edits while swapping tokenizer IDs and changing the embedding table. Require IDs `[1,3,2]` and independently known uploaded embedding rows `[21..24,29..32,25..28]` from the new session. The existing default special-token option remains true. | #305 native editor undo/IME proof and consumer unit tests |

Existing native probes count retained graph/layout objects, workers, WebGL
textures and readers across the small replacement/recovery sequence. The fixture
server additionally reports session IDs, process identity, observer epoch,
subscriber count and whether its single worker is running. These are test-only
observations; they do not restore application state. Its disconnect control ends
real HTTP streams without manufacturing events. Closing all sessions/subscriptions
must settle ownership. This is bounded lifecycle evidence, not total heap/VRAM or
an unbounded memory benchmark.

The passive EventSource subclass only records received native frames. Delayed POST
responses come from `route.fetch()` against the actual backend. The browser
origin's configured backend URL and encoded slash-containing model query are
checked. Process identity and a per-document marker establish that the active
refresh sequence uses neither backend restart nor browser navigation.

## Manual edit-in-place workflow

1. Keep the backend running with a writable local model copy beneath its configured
   model root. Keep `architecture.json` author-local IDs stable for components that
   persist. Start the UI against that backend's configured base URL.
2. Select the model and Architecture Explorer, explore a nonzero concrete stage,
   expand an interior, select a component and pan/zoom. Record the current session,
   `model_revision` and graph ID from the network panel.
3. Write a valid revised definition beside the old file, then atomically rename it
   onto `architecture.json`. Change a visible label/formula as well as its declared
   revision when appropriate. Do not click Reload or navigate the document.
4. Observe `/models/events?model_id=...`: a `model-state` revision change must lead
   to replacement session/inventory/graph requests, fresh IDs and corresponding
   view restoration. Inspect a weight and verify its request uses the new session.
5. Remove a selected child while retaining its parent; expect a nearest-compatible
   ancestor notice. Invalid checkpoint metadata waits for repair without switching
   to another model. Malformed architecture is a local capability failure: Tensor
   and Tokenizer availability follow their own assets. Repair under the same
   logical ID to recover automatically.

Use disposable copies: the backend treats model files as read-only, while the
external author/test performs mutations. Identity changes are remove/add, not
inferred renames. Metadata observation settles changes; it is not an immediate
transaction across an arbitrary multi-file export. Reference models, optional
CUDA, exhaustive layout/family matrices and production deployment load are outside
this fixture proof.

## Reverse-proxy smoke

Use an already available proxy; no proxy installation is part of acceptance. For
nginx, place the following directives in the location forwarding the backend:

```nginx
proxy_pass http://127.0.0.1:8765;
proxy_http_version 1.1;
proxy_set_header Connection "";
proxy_buffering off;
proxy_cache off;
gzip off;
proxy_read_timeout 45s;
```

The backend sends `X-Accel-Buffering: no`, `Cache-Control: no-store`, an initial
`retry: 2000`, named model-state frames and comment heartbeats about every 15 s.
Nginx consumes the buffering header by default; use
`proxy_pass_header X-Accel-Buffering;` if inspecting it downstream. Keep every
proxy/load-balancer idle/read timeout above the heartbeat interval and verify
that compression or another intermediary does not coalesce frames.

With a selected model and the proxy base URL, use an unbuffered client:

```sh
curl -N -i --get --data-urlencode 'model_id=reload/a' \
  -H 'Origin: http://127.0.0.1:4175' http://127.0.0.1:8766/models/events
```

Confirm the configured CORS origin, initial frame before EOF, a changed-revision
frame after an atomic file edit, and at least one idle heartbeat. Disconnect and
verify cleanup in the fixture server's `/__test/state` (zero subscribers and no
observer worker after the last connection), or use server-side lifecycle
observations in a deployment. The production CLI has no `/__test` routes.
Reconnect to obtain current state; event IDs do not imply replay history.

Performed locally on 2026-10-10 with already installed nginx **1.24.0**, isolated
loopback ports, the real fixture backend and a 45 s proxy read timeout: initial
state arrived after **1.00 s**, changed state after **3.01 s** from connection, and
a heartbeat after **15.00 s**, all before EOF. Old/new revisions differed within
one epoch; `X-Accel-Buffering: no` and the allowed CORS origin were observed.
Disconnect returned subscribers to zero and stopped the observer without changing
the backend process. These are observations, not performance thresholds. TLS,
remote hosts, chained proxies and production load were not tested. Direct
ASGI/TCP checks alone would not establish this proxy evidence.

Exact execution revision, selected aggregate commands and outcomes are recorded
in the delivery PR; bulky reports, weights and caches remain outside Git.
