# Kev-0.8B static inspection package

This offline tool prepares a complete, unmerged decision-model package for the
existing Tensor, Tokenizer and Architecture Explorers. The server reads only
native Safetensors and data-only JSON. It does not load `head.pt`, download models,
construct Kev/Qwen modules, or execute the graph. The original PEFT directory
uses `FEATURE_EXTRACTION` and remains outside generic `CAUSAL_LM` admission.

The validated reference is `jaredpalmer/kev-0.8b` at
`9a45d25eb2ab761841196625383fa1dff0e56c1e`, requiring
`Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
Kev's own training provenance pins the inspected source to
[`45923b7a3460b6d36358e2e143455902c1eb856b`](https://github.com/jaredpalmer/kev/tree/45923b7a3460b6d36358e2e143455902c1eb856b).
The source, base and Kev reference are Apache-2.0. See [evidence](evidence.md)
for hashes and observed coverage. These pins are part of this recipe, rather than
an instruction to select the latest upstream checkpoint.

## Prepare and export

Use the repository's locked backend environment. Keep weights and generated
packages outside Git. Preflight with `hf download --dry-run` and `df -h` first:
the selected native base and Kev assets need about 1.8 GB, plus conversion/cache
space. Do not fetch larger models or alternative weight formats.

```sh
uv sync --locked --project backend --python 3.12
# MODEL_ROOT is an operator-selected directory outside the checkout.
hf download Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --include '*.json' --include '*.safetensors' --include '*.txt' \
  --include 'LICENSE' --include 'README.md' --local-dir "$MODEL_ROOT/base"
hf download jaredpalmer/kev-0.8b \
  --revision 9a45d25eb2ab761841196625383fa1dff0e56c1e \
  --include '*.json' --include '*.safetensors' --include 'head.pt' \
  --include 'README.md' --local-dir "$KEV_CHECKPOINT"
backend/.venv/bin/python examples/kev/export.py \
  --base "$MODEL_ROOT/base" --kev "$KEV_CHECKPOINT" --output "$MODEL_ROOT/kev" \
  --base-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --kev-revision 9a45d25eb2ab761841196625383fa1dff0e56c1e
backend/.venv/bin/llm-model-explorer-validate-architecture "$MODEL_ROOT/kev" --json
backend/.venv/bin/python -m llm_model_explorer \
  --model-root "$MODEL_ROOT" --cache-dir "$ARTIFACT_CACHE"
```

The output must be new and outside both resolved inputs, including when an output
path uses symlink parents or `..`. Shared base shards must stay
inside the configured model root. Use `--copy-shards` when preparing an independent
portable package; copying is bounded by file, without resident full-weight loading.
The exporter builds the package beneath a temporary container inside the resolved
output parent. The container has no model configuration, so catalogue scans cannot
select the nested package. Validation uses the output model root to confine shared
shards; only a valid package is atomically renamed to the final directory on the
same filesystem. The temporary container is removed on success or failure. The
exporter never modifies either input. Do not move a shared package without its base; a
copied package can relocate without changing its content fingerprint.

`head.pt` conversion explicitly uses `torch.load(weights_only=True,
map_location="cpu", mmap=True)` with no safe-global additions or unrestricted
fallback. Only the four native query/key weights and biases and validated primitive
metadata are accepted. Missing calibration, incompatible base revisions, unknown
adapter targets/options, absent/duplicate/orphan factors, wrong geometry,
question-side LoRA, trained special embeddings and hybrid option isolation fail
before publication. The tool requires the pinned single adapter Safetensors
layout; sharded adapters, full-weight Kev, quantized bases and larger Kev sizes
have no support claim.

## Tensor inventory and provenance

Every base tensor retains its exact name, dtype and values. The index references
its unchanged shard under `base/<original-shard>`. Adapter tensors become
`kev.lora.<exact-original-PEFT-key>`, and pointer tensors become
`kev.head.<exact-original-head-key>`. These reserved namespaces are checked against
the base inventory. No factor is merged, no base matrix is rounded again, and no
virtual effective/delta tensor is created. Tensor Explorer can inspect base, A,
B, query/key weights and biases independently.

`kev-provenance.json` retains the original `FEATURE_EXTRACTION` adapter metadata,
exact local input hashes, selected revisions, head calibration, namespace mapping
and each checked target's orientation/scaling. `config.json` preserves the Qwen
text/tokenizer/input-table layout and uses the distinct identity
`jaredpalmer/kev-0.8b-inspection@<Kev revision>`. Its `kev_inspection` declaration
requires the sidecar; a missing sidecar cannot become a bare Qwen graph. The
package's ordinary content fingerprint covers all base shards, exported factors
and head, config, provenance and `architecture.json`. Modified sources invalidate
pinned sessions and caches through the existing lifecycle.

The bare base configuration has no revision declaration on this reference.
The operator must supply the exact downloaded base revision; it is checked against
`head.pt` and the available training/provenance metadata. This is a selected-input
binding, not a claim that a directory name certifies its contents. Reproduction
checks all recorded input hashes and compares original values to real API streams.
The original head file's hash binds all metadata, including training-only fields;
private upstream training paths are not copied into the public graph or report.

## Decision computation and graph

The view describes the actual text decision path in `encode`, `rows_of`,
`forward_rows_batch`, `PointerHead`, and checkpoint loading at the inspected source.
All 24 real hybrid layers remain explicit, with six full-attention layers and
18 Gated DeltaNet layers. Every adapted linear operation has a group containing
its ordinary base projection, `A`, `B`, multiplication by `alpha/r`, and residual
addition. Factors use `[rank,input]` and `[output,rank]`; the base uses
`[output,input]`. Untargeted normalization, gating, convolution, recurrence,
attention and MLP paths retain the existing reviewed Qwen3.5 topology. The graph
reuses that description's source pin without adding native Qwen3.5 packaged
runtime admission.

Each question runs in an independent causal row containing the same state prefix
and its own instruction/options. Prior and next KV, convolution and recurrent
state are per row. Zero prior state represents the complete state+question row;
serving can reuse the state prefix by copying its caches independently into each
question row. A packed block-causal attention mask cannot isolate Gated DeltaNet
recurrence across questions and is not used by this reference. The source's
attention-only packed path is a separate configuration-dependent path. Options
within each question remain sequential; this hybrid reference does not implement
option-isolation/permutation invariance.

The pointer query reads the final decision-token hidden state; keys read each
option's closing-delimiter hidden state. Both receive the source's float32 hidden
states and separate weight/bias projections. Per-question scores are
`dot(k(option), q(decide)) / sqrt(head_dim) / temperature`, followed by an
option-axis softmax. The reference pointer width is 256 and inference temperature
is `2.3510958125672174`. Outputs are probabilities over supplied options, whose
counts can differ by question. The vocabulary output projection is not a decision
operation. All original base tensors, including unused visual/MTP tensors, remain
in the inventory; visual/MTP components are contextual rather than executed text
operations.

Typed-request serialization and tokenization remain auxiliary tooling outside
the architecture. Kev reuses these existing delimiter tokens in order:
`<|fim_prefix|>` (state), `<|fim_middle|>` (question), `<|box_start|>` (option start),
`<|box_end|>` (option end), `<|fim_suffix|>` (decision). No embedding rows are added
or rewritten. Kev's encoder escapes user-entered delimiter syntax before
serialization. The existing Tokenizer Explorer still inspects its user's actual
text; it is not a Kev request editor.

The importer retains `model_defined` and the visible model-supplied notice.
Static import and exporter correspondence tests do not certify backbone inference,
training quality, a Jev reproduction, or universal PEFT compatibility.

## Reproduce validation

```sh
(cd backend && uv run --locked pytest tests/test_kev_export.py \
  tests/test_qwen35_architecture.py tests/test_peft_adapters.py \
  tests/test_lora_architecture.py tests/test_model_defined_architecture.py \
  tests/test_clm_export.py -q)
OMP_NUM_THREADS=2 backend/.venv/bin/python -m examples.kev.check_reference \
  --base "$MODEL_ROOT/base" --kev "$KEV_CHECKPOINT" \
  --model-root "$MODEL_ROOT" --evidence "$NEW_EVIDENCE_DIRECTORY"
(cd ui && npm run check)
(cd ui && UI_TEST_PORT=25100 xvfb-run -a npm run test:acceptance -- kev.spec.ts)
(cd ui && LMEX_KEV_REFERENCE_MODEL_ROOT="$MODEL_ROOT" \
  UI_TEST_PORT=25100 xvfb-run -a npm run test:acceptance -- kev.spec.ts)
```

The fixture browser test generates native local assets and runs the real backend
and production bundle at DPR 1 and 2. The reference invocation selects the actual
exported checkpoint and opens a hybrid LoRA factor and a pointer weight through
ordinary architecture navigation and numeric streams. Large outputs, screenshots
and runtime caches remain outside Git. Aggregate integration gates retain their
existing CI ownership; fixture success alone is not actual-reference acceptance.
