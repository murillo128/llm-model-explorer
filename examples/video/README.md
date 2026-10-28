# Native static video inspection

Export the selected local LTX/Wan files into a self-contained inspection package,
then select it in Tensor, Tokenizer or Architecture Explorer. The native graph
describes the base single-scale pipeline and its mathematical components. It
does not generate, upload, play or save video.

## Selected references

| Export profile | Denoiser input | Other required inputs | Semantic modes |
| --- | --- | --- | --- |
| `ltx-0.9.8` | `ltxv-2b-0.9.8-distilled.safetensors` | Embedded VAE; PixArt T5 text encoder and tokenizer | Text or initial image + text |
| `ltx-0.9.6` | `ltxv-2b-0.9.6-distilled-04-25.safetensors` | Embedded VAE; the same T5 assets | Text or initial image + text |
| `wan-ti2v-5b` | `wan2.2_ti2v_5B_fp16.safetensors` | Wan 2.2 VAE, scaled-FP8 UMT5, UMT5 tokenizer | Text or initial image + text |
| `wan-t2v-a14b` | Both `wan2.2_t2v_{high_noise,low_noise}_14B_fp8_scaled.safetensors` | Wan 2.1 VAE, scaled-FP8 UMT5, UMT5 tokenizer | Text |
| `wan-i2v-a14b` | Both `wan2.2_i2v_{high_noise,low_noise}_14B_fp8_scaled.safetensors` | Wan 2.1 VAE, scaled-FP8 UMT5, UMT5 tokenizer | Initial image + text |

[`video_reference.json`](../../backend/src/llm_model_explorer/video_reference.json)
is the executable package's data-only reference register: immutable repository
revisions, per-artifact sizes/content hashes, license declarations, source URLs,
header inventories and verified component configurations. It contains no tensor
values or graph nodes. Code interpretations are pinned to
[LTX-Video `4b2d053`](https://github.com/Lightricks/LTX-Video/tree/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f),
[Wan2.2 `1ea34ff`](https://github.com/Wan-Video/Wan2.2/tree/1ea34ff48f87168174e12956e200b1d908b1c5ff)
and [ComfyUI `79cf64c`](https://github.com/Comfy-Org/ComfyUI/blob/79cf64c21a1dba50749945ba9fe2e3a692f45438/comfy/utils.py).
Those implementations were consulted as format/semantic references; the exporter
does not import them. Checkpoint license terms remain applicable to local copies
and redistribution. Exported shards preserve embedded license notices and the
package includes `LICENSES.txt` and source provenance.

## Offline export

Use Linux with atomic `renameat2(RENAME_NOREPLACE)` and the locked backend
environment (`uv sync --locked --project backend`). The export command has no
network path. Select the exact files already available locally; filenames alone
never verify their role. Source hash/config mismatches fail before publication.
Provide an existing output parent outside every input directory. The final
destination must not exist. Account for both original files and one original-dtype
package copy; no decompressed FP16/FP32 weight copy is created.

```sh
backend/.venv/bin/python examples/video/export.py \
  --profile ltx-0.9.8 \
  --denoiser /local/ltx/ltxv-2b-0.9.8-distilled.safetensors \
  --text-encoder /local/pixart/text_encoder \
  --tokenizer /local/pixart/tokenizer \
  --output /local/models/ltx-098-inspection

backend/.venv/bin/python examples/video/export.py \
  --profile wan-ti2v-5b \
  --denoiser /local/wan/wan2.2_ti2v_5B_fp16.safetensors \
  --vae /local/wan/wan2.2_vae.safetensors \
  --text-encoder /local/wan/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
  --tokenizer /local/umt5-xxl \
  --output /local/models/wan-ti2v-inspection

backend/.venv/bin/python examples/video/export.py \
  --profile wan-t2v-a14b \
  --high-noise /local/wan/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors \
  --low-noise /local/wan/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors \
  --vae /local/wan/wan_2.1_vae.safetensors \
  --text-encoder /local/wan/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
  --tokenizer /local/umt5-xxl \
  --output /local/models/wan-t2v-inspection
```

For I2V use `wan-i2v-a14b` and both I2V experts. For LTX 0.9.6 change the profile
and its combined checkpoint together. The T5 directory needs both indexed
`model-0000{1,2}-of-00002.safetensors` shards and `config.json`; tokenizer files
are selected from the pinned register. Wan uses the conventional local UMT5
tokenizer files instead of exposing its embedded `spiece_model` as a tensor.

Output contains `config.json`, namespaced Safetensors shards and index,
tokenizer assets, `video-provenance.json` and license notices. Original files
are read-only. Every tensor retains its original role/name mapping. Staging is
unselectable, checked again for source mutation, and published atomically without
overwriting an existing package. Fresh exports have no `architecture.json`.

```sh
backend/.venv/bin/llm-model-explorer-validate-architecture \
  /local/models/wan-ti2v-inspection --json
backend/.venv/bin/llm-model-explorer-backend \
  --model-root /local/models --cache-dir /local/model-cache --device cpu
```

The public model identity distinguishes release, variant and representation;
the ordinary validated encoding suffix also identifies the mixed Wan package's
FP8 component. Architecture preparation uses metadata and bounded hashing, and
opening one weight uses bounded logical reads. Convolution weights keep their
real rank and report `unsupported_rank` in the 1D/2D matrix inspector.

## Numerical meaning and evidence

Scaled E4M3FN weights return `float32(q) * float32(weight_scale)`. The legacy
`scaled_fp8` marker and activation scale are never weight multipliers. Native
F16/BF16/F32 values remain native logical tensors. The frontend consumes only
canonical float32 through the existing binary protocol.

[`evidence.md`](evidence.md) separates small-fixture conformance, actual complete
reference preparation, numerical correspondence and browser observations.
Neither a matching header nor fixture success certifies a complete checkpoint.
To measure an export or native validation, prefix its command with
`/usr/bin/time -v`; keep large models, generated graphs, caches and logs outside
Git. No CUDA, full-model downloads or upstream inference runtime is needed by
ordinary CI.

For an exported complete package, the local reference checker starts the existing
TCP harness, validates the native graph, tokenizes text and compares whole selected
weight streams against bounded slices of the original Safetensors files:

```sh
backend/.venv/bin/python examples/video/check_reference.py \
  /local/models/wan-ti2v-inspection --source-root /local/reference-inputs \
  --output /tmp/wan-ti2v-reference.json

npm --prefix ui run build
LMEX_VIDEO_REFERENCE_PACKAGE=/local/models/wan-ti2v-inspection \
  LMEX_TEST_PORTFOLIO=full UI_TEST_PORT=4331 \
  LMEX_EVIDENCE_DIR=/tmp/video-browser \
  xvfb-run -a npm --prefix ui run test:acceptance -- video.spec.ts --project=dpr1
```

The source root must contain one copy of each original filename needed by the
numeric comparison. Reference browser checks are explicitly optional in ordinary
CI; use `LMEX_TEST_PORTFOLIO=full` with the package variable to run them locally.
