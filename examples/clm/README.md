# CLM static inspection export

This offline exporter prepares the complete native CLM-v0.1-8B reference for
the existing Tensor, Tokenizer and Architecture Explorers. The original `.pt`
directory is not an admissible Explorer model. This is static inspection;
no encoder inference, generation, serving endpoint or quality claim is added.

The inspected upstream source is
[`Contrastive-LM/CLM@bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`](https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7).
`heads.py` defines the configurable MLPs, LayerNorm, projection normalization
and `min(exp(float32(logit_scale)), 100)`; `embedder.py` normalizes encoder
vectors with `norm + 1e-12`; `engine.py` scores candidates and applies request
temperature before candidate normalization. Tokenization is an external Explorer
capability. Each encoder invocation has its own token/position/mask interfaces
and sequence length; last-token pooling uses right-padded lengths. Parameters
are shared, while hidden states and attention dependencies are independent.
The stored vocabulary output head is inspectable but absent from executed
decision operations. The graph retains the model-supplied, unverified-forward
notice.

## Immutable reference inputs

All three upstream repositories use Apache-2.0. Keep their licenses and provenance.
The following pins are the accepted reference preparation; do not substitute
Qwen3-8B-Base, another encoder size, quantization or pooling rule.

| Input | Revision | Selected assets |
| --- | --- | --- |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | All five native Safetensors shards, index, config, tokenizer JSON/config, vocabulary/merges, LICENSE, README |
| `Contrastive-LM/CLM-v0.1-8B` | `e939398d4556fcd9400c76fa8c5a513202f42b0a` | `CLM_v0.1-8B.pt`, config, LICENSE, README |
| `Contrastive-LM/CLM` source | `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` | `src/clm/heads.py`, `embedder.py`, `engine.py`, LICENSE |

The five encoder shards total 16,381,516,776 bytes. The head is 75,557,149 bytes;
its upstream SHA-256 is
`b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`.
Config, tokenizer and metadata add approximately 16 MiB. Preflight the selected
download bytes and free disk before downloading. Shared-shard export requires
about 200 MiB additional free space; `--copy-shards` additionally requires the
complete encoder size. Leave room for the graph cache and check artifacts.

## Reproduce outside Git

Use the locked backend environment (including PyTorch and Safetensors) and an
installed `hf` CLI. These download commands are operator/development preparation;
the exporter and running backend do not download anything. Choose an external
working directory, with encoder and output inside the same model root:

```sh
clm_work=$(mktemp -d /tmp/clm-reference-XXXXXX)
mkdir -p "$clm_work/models/qwen" "$clm_work/head"
df -h "$clm_work"
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --include '*.safetensors' --dry-run
hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt \
  --revision e939398d4556fcd9400c76fa8c5a513202f42b0a --dry-run
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --include '*.safetensors' --local-dir "$clm_work/models/qwen" --max-workers 2
hf download Qwen/Qwen3-8B config.json model.safetensors.index.json tokenizer.json \
  tokenizer_config.json vocab.json merges.txt LICENSE README.md \
  --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir "$clm_work/models/qwen"
hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt config.json LICENSE README.md \
  --revision e939398d4556fcd9400c76fa8c5a513202f42b0a --local-dir "$clm_work/head"
hf cache verify Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir "$clm_work/models/qwen"
hf cache verify Contrastive-LM/CLM-v0.1-8B \
  --revision e939398d4556fcd9400c76fa8c5a513202f42b0a --local-dir "$clm_work/head"
PYTHONPATH=backend/src HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  backend/.venv/bin/python examples/clm/export.py \
  --encoder "$clm_work/models/qwen" --head "$clm_work/head/CLM_v0.1-8B.pt" \
  --output "$clm_work/models/clm" \
  --encoder-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --head-revision e939398d4556fcd9400c76fa8c5a513202f42b0a
PYTHONPATH=backend/src backend/.venv/bin/python -m llm_model_explorer.validate_architecture_cli \
  "$clm_work/models/clm" --json
PYTHONPATH=backend/src:. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  backend/.venv/bin/python examples/clm/check_reference.py \
  --encoder "$clm_work/models/qwen" --head "$clm_work/head/CLM_v0.1-8B.pt" \
  --model-root "$clm_work/models" --evidence "$clm_work/acceptance"
```

Default export uses confined shard symlinks to the unchanged local encoder.
Run the backend with their common model root. Keep both directories together;
for independent relocation use `--copy-shards` when exporting, or dereference
the symlinks when copying. Content identity excludes absolute/resolved paths.
No input file is written. The output's resolved location must be outside the
encoder, including symlink-parent and traversal aliases. Staging uses a private
container under the writable output parent; its nested package is invisible to
catalogue discovery. Canonical import validates against that common confinement
root before atomic rename publishes the complete directory on the same
filesystem. An output parent beneath a read-only ancestor is supported. Existing
destinations are refused. Failed conversion/import removes only its own staging
container.

`clm-provenance.json` records the input pins, per-file SHA-256 and size, complete
tensor names/shapes/dtypes, source and exporter revision, head configuration and
namespace mapping. Encoder names are unchanged; heads are mapped to
`clm.state_head.*` and `clm.action_head.*`, with `clm.logit_scale` separate. The
combined index accounts for every selected tensor once. Native dtypes/values are
preserved, and encoder weights are never materialized as a numerical model.
Config preserves the Qwen text/input-table layout while adding a distinct CLM
identity and a required-sidecar marker. Missing/mismatched head bindings or loss
of the sidecar cannot produce a bare-Qwen fallback under that identity.

Only primitive configuration, native tensor dictionaries and a finite scalar
scale are admitted from the bounded `.pt` file, using explicit
`torch.load(weights_only=True, map_location="cpu", mmap=True)`. No unrestricted
pickle fallback, safe-global extension, remote code import or vLLM is used.
The head schema rejects extra/missing tensors, dimension conflicts and unsupported
options. Before replacing the encoder identity, the exporter rejects any
non-null `_name_or_path` / `name_or_path` declaration other than `Qwen/Qwen3-8B`,
and any `_commit_hash` / `revision` declaration differing from the selected
immutable revision. All declarations must agree, including secondary fields.
Provenance preserves those declarations and records whether repository/revision
binding came from configuration or operator selection. The pinned upstream Qwen
config omits these fields, so revision/checksum verification above remains the
operator's proof of upstream identity; compatible geometry alone is insufficient.
Reduced untrained test fixtures are not the reference checkpoint.

## Validation ownership

The existing dense-Qwen suite owns attention/MLP/RoPE/layer correspondence; the
model-owned importer suite owns generic schema, hierarchy, binding and resource
limits. Added CLM cases cover the remaining composite risks:

| Contract and plausible defect | Independent assertion | Owner |
| --- | --- | --- |
| Head options, wrong namespace or dtype, duplicate encoder copies | Exact original tensor bytes/dtypes, independent two-layer inventory, shared parameter consumer identities | `backend/tests/test_clm_export.py` |
| Wrong activation/norm/residual/order/scaling | Float64 scalar equations transcribed from pinned source versus evaluation of exported head dependencies; tolerances 2e-6 for vectors/probabilities, 2e-4 for scaled scores | Same backend test |
| Unsafe or incomplete export | Restricted-load rejection, incompatible geometry or repository/revision declarations, absent/extra tensors, duplicate base storage; no selectable destination after failure | Same backend test |
| Input aliases, unauthorized staging or premature publication | Complete input tree/bytes unchanged; real non-root directory permissions in shared/copy modes; real catalogue/validator observation, validated device/inode retained after publication, failure cleanup | Same backend test |
| Nested validation escapes confinement | Default parent rejects shared shards outside it; explicit root admits confined sharing and rejects external package/shard targets | `backend/tests/test_model_defined_service.py` |
| Real consumer/lifecycle, accidental checkpoint loading/network/fallback | Backend startup/cache and endpoints, exact binary bytes and embedding rows; runtime load/network traps, unavailable damaged definitions, relocation/content invalidation | Same backend test |
| Production rendering/navigation | Select CLM, expand a projection group, inspect its real native matrix | `ui/acceptance/clm.spec.ts` |

Run focused checks from the repository root:

```sh
PYTHONPATH=backend/src HF_HUB_OFFLINE=1 backend/.venv/bin/python -m pytest backend/tests/test_clm_export.py backend/tests/test_model_defined_service.py -q
(cd backend && .venv/bin/ruff check tests/clm_fixtures.py tests/test_clm_export.py)
backend/.venv/bin/ruff check --config backend/pyproject.toml examples/clm
(cd backend && .venv/bin/mypy src tests ../examples/clm/export.py)
```

The browser test uses the reduced deterministic package by default and the
operator-selected complete package when `LMEX_CLM_REFERENCE_MODEL_ROOT` is set.
Build the production UI before either run, using the repository's pinned Node
environment and installed Playwright Chromium. On Linux the configured headed
browser requires a working display, for example `xvfb-run -a`:

```sh
(cd ui && npm run build && xvfb-run -a npm run test:acceptance -- clm.spec.ts --project=dpr1)
(cd ui && LMEX_CLM_REFERENCE_MODEL_ROOT="$clm_work/models" \
  xvfb-run -a npm run test:acceptance -- clm.spec.ts --project=dpr1)
```

The reference run must be reported separately. See [acceptance evidence](evidence.md)
for the observed actual reference and the validation limits.
