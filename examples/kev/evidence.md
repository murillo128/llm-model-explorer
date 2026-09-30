# Kev-0.8B reference evidence

Observed on 2026-10-01 (Europe/Madrid), on Linux x86_64 with the locked
Python 3.12 backend, CPU PyTorch 2.14.0, Node 24.14.0 and Playwright Chromium
1243 using Xvfb/SwiftShader. No backbone construction, inference or GPU was
required. The reference inputs and large outputs remain outside Git.

## Immutable inputs and producer

- Kev Hub revision: `9a45d25eb2ab761841196625383fa1dff0e56c1e`.
- Required Qwen base revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Inspected Kev source revision: `45923b7a3460b6d36358e2e143455902c1eb856b`.
- Exporter SHA-256: `88fc5939b980c1f9aa6d80f47e3e35e960330ac70b645acaab9e1fff08f04da2`.
- Qwen description: `{"analyzer_revision": "static-graph-core-3", "description": "transformers-qwen35-nvfp4", "revision": "2", "schema_revision": "6d4f6ea778590ff68d919adb59962eb97951240df7b521b3ca08c4846fd68b88", "source_revision": "transformers/2cba19507be799b7bef247ca6c1c4708bf881b5b"}`.
- Base content fingerprint: `2207f5619dd73c19fe6c00806701d8796909a7c8cee10cdcb229355bd9fadba3`.
- Model identity: `jaredpalmer/kev-0.8b-inspection@9a45d25eb2ab761841196625383fa1dff0e56c1e`.
- Licenses: Kev source and checkpoint Apache-2.0; Qwen base Apache-2.0.

The base config omits repository/revision identity fields. The selected base
revision was checked against `head.pt`, training configuration and provenance,
and against immutable Hub file hashes. The head is the published calibrated
file, not the older head hash/raw temperature retained in training provenance.

All 11 downloaded base files and 10 downloaded Kev files match the pinned Hub
tree: LFS SHA-256 for large files and Git blob SHA-1 for small files. Non-model
`.gitattributes` and the upstream training log were not needed. The CLI checksum
check with `--fail-on-extra-files` reported its own `.cache/huggingface` metadata
as extra files; the independent Hub-tree digest comparison above established the
selected file checksums without changing any inputs.

| Base selected input | Bytes | SHA-256 |
| --- | ---: | --- |
| `LICENSE` | 11343 | `50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32` |
| `config.json` | 2907 | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `merges.txt` | 3353259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `model.safetensors-00001-of-00001.safetensors` | 1746942600 | `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c` |
| `model.safetensors.index.json` | 50900 | `ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a` |
| `tokenizer.json` | 12807196 | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |
| `tokenizer_config.json` | 16712 | `e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de` |
| `vocab.json` | 6722759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

| Kev selected input | Bytes | SHA-256 |
| --- | ---: | --- |
| `head.pt` | 2103999 | `f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6` |
| `adapter_model.safetensors` | 43338624 | `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a` |
| `adapter_config.json` | 1273 | `748acb2cda88454cb1ba69d745ba336f3fcb5486eac349e90960c8b8d8d3e854` |
| `training_config.json` | 1767 | `3ec3d92146248265652d4387f8eb65ee8defaed4dd9173b049e7b5193045ece6` |
| `provenance.json` | 4221 | `fce039fbda7de19451c9b5d3065aee2c5e063e1562a8bc18c72e297fbd15e9d8` |
| `tokenizer.json` | 19989325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | 1128 | `8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45` |
| `README.md` | 18516 | `5c0408069bf3882c758ccc10a8351a4fba6a66e5cdce0ae6b3c320f4a50c0afb` |

| Inspected source | SHA-256 |
| --- | --- |
| `LICENSE` | `6b08bb37982c233aa12bcbdf19106da12f3f4fcf800773ea42e28ebeddd34fb8` |
| `kev/checkpoint.py` | `f3edb4d159c0aa12d7006fc599fe6ebc3f2c288b3b69890c289f7dac451dd03d` |
| `kev/model.py` | `2634ffe7747d69473fb6596942c248e5df2586df3610bd1adb47e7e9acd99f96` |

## Static import and production lifecycle

The exporter and the installed
`llm-model-explorer-validate-architecture <package> --json` both returned
`status=valid`, `coverage=complete`, no diagnostics. The actual backend CLI
performed startup discovery/preparation and served the package alongside its
native base. HTTP architecture retrieval retained `model_defined` and the
model-supplied notice.

| Observation | Result |
| --- | --- |
| Inventory / graph parameters | 864 (488 original base + 372 factors + 4 pointer tensors) |
| Adapted projections | 186 |
| Hybrid layers | 24: 18 linear attention, 6 full attention |
| Graph nodes / edges | 1915 / 2694 |
| Sidecar / JSON response bytes | 2133369 / 5554811 |
| Cold startup | 8.496 s |
| Warm startup | 5.433 s, same graph ID, cached artifact reused |
| Pointer dimension / temperature | 256 / 2.3510958125672174 |

Graph ID: `d659bae4d108099a2e3fdc945d8ab4173d4fbc5362ea54bcd488bd6cf1ec9b9c`.
Definition SHA-256: `f1d97cb69f821a492110e734c4e78fe6c3066497b47c8d401ff245f10b1d5566`.

The sidecar remains below 8 MiB and the prepared response below 32 MiB.
Complete coverage is within the declared text option-decision scope. Visual/MTP
weights remain preserved inventory and context; their forward paths are not
claimed. Hybrid question rows and their KV/conv/recurrent state remain isolated;
no packed-mask or option permutation-invariance claim is made.

## Independent original-value observations

Eight complete production binary streams matched the corresponding original
Safetensors/head values byte-for-byte after the API's canonical float32 conversion.
All streams ended successfully, including both query/key biases.

| Original/exported sample | Native dtype | Shape | First four float32 values |
| --- | --- | --- | --- |
| `model.language_model.norm.weight` | BF16 | `[1024]` | `[-0.77734375, 2.75, 2.640625, 3.0625]` |
| `model.language_model.layers.0.linear_attn.in_proj_a.weight` | BF16 | `[16, 1024]` | `[-0.02392578125, 0.06787109375, 0.09326171875, 0.078125]` |
| `kev.lora.base_model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight` | F32 | `[16, 1024]` | `[-0.028449399396777153, 0.00716873724013567, 0.013701829127967358, 0.011302702128887177]` |
| `kev.lora.base_model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight` | F32 | `[6144, 16]` | `[-0.004132404457777739, -0.0026709996163845062, 0.004493480082601309, -3.457834100117907e-05]` |
| `kev.head.q.weight` | F32 | `[256, 1024]` | `[-0.02298840321600437, 0.02146698348224163, 0.030700622126460075, -0.01633305847644806]` |
| `kev.head.k.weight` | F32 | `[256, 1024]` | `[0.0059750983491539955, 0.01535598374903202, 0.0017799054039642215, -0.02098538726568222]` |
| `kev.head.q.bias` | F32 | `[256]` | `[0.004845152609050274, -0.010690426453948021, 0.028925402089953423, -0.016478851437568665]` |
| `kev.head.k.bias` | F32 | `[256]` | `[0.0003076594730373472, 0.017372656613588333, -0.01931297592818737, -0.013722769916057587]` |

Actual tokenizer input: `Hello world, España 😀`. Original and HTTP token IDs
were `[9419, 1814, 11, 54614, 87209]`. Ordered embedding lookup additionally
checked all delimiter IDs and a repeated token row against the original table.
All returned rows matched exactly. Delimiters: `{'<|fim_prefix|>': 248060, '<|fim_middle|>': 248061, '<|box_start|>': 248049, '<|box_end|>': 248050, '<|fim_suffix|>': 248062}`.

## Production browser and deterministic regression proof

`ui/acceptance/kev.spec.ts` passed with the complete real export at DPR 1 and 2
(17.2 s and 18.6 s; 37.1 s including setup). It selected Kev, expanded layer 0's
adapted `linear_attn.in_proj_a`, opened its native A factor, then expanded the
pointer head and opened `q.weight`. Both numeric responses succeeded and both
matrix canvases rendered without alerts. The captured real screenshots were
visually inspected. The same download-free fixture path passed at DPR 1 and 2
(10.3 s and 11.5 s; 23.0 s including setup). Screenshots/traces stay outside Git.
The sandbox could not expose Xvfb to Chromium; these headed runs used the
permitted host execution path with temporary X servers and isolated ports.

- Focused backend suite: 178 passed across Kev, Qwen3.5, PEFT, LoRA,
  model-owned architecture and CLM.
- Backend Ruff lint/format and mypy: passed (104 typed source/test files).
- `npm run check`: API binding freshness, types, lint, 1,019 unit tests,
  production build passed.
- Export/check-reference scripts: Ruff lint/format with backend configuration.
- Mutation proof separately changes base bytes, factor bytes, head bytes,
  config and sidecar; relocation retains content identity and pinned mutations
  fail rather than silently refresh.
- Independent fixture algebra: unmerged projection `[7,18,0,0,0,0]`, pointer
  logits `[0.5,1.25]`, probabilities approximately `[0.3208213008,0.6791786992]`.
  Graph assertions bind each query/key bias and factor to its consuming operation.
- Negative controls include unsafe/object metadata, missing head/calibration,
  invalid temperature, base/revision conflicts, unknown targets/options,
  missing/duplicate/orphan factors, incorrect shapes and unsupported hybrid flags.
- An explicit RoPE-factor conflict failed its rejection test before the fix
  (`DID NOT RAISE`) and passes afterward; omitted factors still inherit the
  selected base's declared RoPE value.

The existing backend/API/UI/application PR workflows exclude
`codex/epic-issue-*` bases. This child uses the accepted scoped local validation;
the final integration PR still owns its aggregate CI gates. No CI waiver or
workflow change was introduced. The only acceptance configuration change adds
the deterministic Kev browser path to the existing suite.

## Limits

These observations establish static exploration of the pinned native Kev-0.8B
package. They do not verify a full backbone forward pass, quality, Jev equivalence,
CUDA behavior, other Kev sizes, quantized variants, arbitrary FEATURE_EXTRACTION
adapters, or inference serving. Only the exporter reads the restricted upstream
head payload. No weights, runtime caches, full graphs or private paths are
committed. Reproduction commands are in [README.md](README.md).
