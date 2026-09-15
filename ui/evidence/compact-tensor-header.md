# Compact tensor header and distribution ruler

Issue [#107](https://github.com/murillo128/llm-model-explorer/issues/107).
The [desktop reference](compact-tensor-header-desktop.png) shows the production
UI at 1440×900 CSS pixels, DPR 1, using the deterministic local acceptance
fixture `model.layers.0.mlp.down_proj.weight` (`[576,1536]`, F16).

The capture uses built `ui/dist` assets served by
[`acceptance/static.mjs`](../acceptance/static.mjs), and the production backend
through [`acceptance.server`](../../acceptance/server.py). All three scientific
operations had completed. Chromium 153.0.8010.12 ran headed under Xvfb with
SwiftShader and native scrollbars; Node was 24.14.0, Python 3.12.14 and PyTorch
2.14.0+cpu. This is fixture evidence.

## Observations

- One 40 CSS-pixel header contains the emphasized public logical path,
  information control, secondary shape/dtype, and reachable rightmost `Fit width`.
- No persistent `Full-range bins` description is rendered. Activating the labelled
  information dialog exposes `Full-range bins`, true finite minimum `-1`, true
  finite maximum `1`, and the existing tensor metadata.
- Header bottom is Y=117; the matrix and row histogram both start at Y=151.
  The retained 24-pixel numeric ruler and 10-pixel scientific grid gap account
  for that separation. The descriptive metadata row is absent.
- The horizontal row ruler and histogram both span X=1311…1411 (100 pixels).
  The low label starts at X=1311, the high label ends at X=1411, and zero is
  centered at X=1361, matching the authoritative `[-1,1]` domain.
- Endpoint captions occupy separate text rows so decimal bounds remain readable
  when the 100-device-pixel histogram is 50 CSS pixels wide at DPR 2. Zero uses
  the existing gap below the axis. In this capture, low occupies Y=117…128,
  high Y=129…140, and zero Y=140…150, all above the histogram at Y=151.
  The caption rectangles do not overlap; header, ruler and data geometry remain
  unchanged.
- The matrix and row histogram have the same Y extent (151…722). The document
  remains 1440×900 with no document overflow. The capture had no page errors and
  loaded JavaScript exclusively from production `/assets/` URLs.

The capture script asserted the header count, absent permanent description,
information-dialog content, and ruler/label alignment and non-overlap against the
actual row canvas rectangle. The screenshot supplements the repeatable geometry checks in
[`tensor-header.spec.ts`](../tests/tensor-header.spec.ts) and
[`distribution-scale.spec.ts`](../tests/distribution-scale.spec.ts).

## Reproduce the reference

Build with `npm run build --prefix ui`. In separate terminals at the repository
root, start the existing acceptance backend and static server:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  backend/.venv/bin/python -m acceptance.server \
  --port 8875 --origin http://127.0.0.1:4275
```

```sh
cd ui
LMEX_STATIC_PORT=4275 LMEX_BACKEND_PORT=8875 node acceptance/static.mjs
```

Open `http://127.0.0.1:4275` in a 1440×900, DPR-1 browser, select
`acceptance/fixture`, expand `model → layers → 0 → mlp → down_proj`, and select
`weight`. Wait for tensor, statistics and distributions to finish, then move
the pointer off the scientific surfaces before capturing.

The local run reused an existing backend interpreter with this worktree's
`backend/src` on `PYTHONPATH`. The host's NVIDIA EGL initialization crashed
under the sandbox; setting
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json`
for `xvfb-run` selected the working software display path. No application or
acceptance configuration changes were needed.
