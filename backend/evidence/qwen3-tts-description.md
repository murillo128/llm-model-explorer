# Qwen3-TTS Base and unmerged LoRA inspection

The native producer covers the speech-language components of
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`. It exposes both the original checkpoint and a
separate Base + LoRA composition, preserving the original tensors and additional
A/B factors. It does not merge weights or synthesize audio.

## Reference and source provenance

- Base config/weights: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` at
  `fd4b254389122332181a7c3db7f27e918eec64e3`.
- Adapter: `aguken-ai/Qwen-3-TTS-12Hz-1.7B-Base-hi-LoRA-Finetuned` at
  `b84766274af71506758ccf45a9965bd1d934297b`, `adapters/hindi_female`.
- Reviewed implementation: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e),
  `qwen_tts/core/models/modeling_qwen3_tts.py` and
  `configuration_qwen3_tts.py`. Upstream code/model and selected adapter declare
  Apache-2.0 (license included beside the reference config). The small packaged
  reference JSON is the upstream Base config;
  no upstream inference code is vendored or executed.

The selected Base has 28 Talker layers (width 2048) and five code-predictor layers
(width 1024). Both have 16 query heads, eight KV heads and head dimension 128.
Text embeddings are `[151936, 2048]`, distinct from the `[3072, 2048]` first-code
table. The fifteen predictor embedding tables have shape `[2048, 2048]` and its
heads `[2048, 1024]`. The biased projection bridges Talker to predictor width.
The text projection contains two biased linear layers with SiLU between them.

Source correspondence: `Qwen3TTSTalkerModel` owns text/codec embeddings and the
Talker stack; `Qwen3TTSTalkerForConditionalGeneration` owns the text projection,
first-code head and predictor. `Qwen3TTSTalkerCodePredictorModel` owns its own
layers/norm and the residual code tables, whose width is the Talker width.
`Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` projects the input
prefix and selects a head by generation step. Both decoder-layer implementations
apply input RMSNorm, Q/K-normalized GQA, residual, post-attention RMSNorm, gated
SiLU MLP and residual. Talker `get_rope_index` repeats equal text positions on all
three axes; the reviewed multimodal rotary selection consequently reduces to
ordinary split-half rotation for that path.

The adapter has rank 16, alpha 32 and 231 targets: seven attention/MLP linear
projections in all 33 layers. Its 462 factors bind with scale 2. Reference audio
and `speaker_embedding.safetensors` are voice-conditioning assets, not LoRA
factors, and are not included in the adapter inventory. No synthesis claim is
made for this inspection composition.

## Explicit limits

The graph is always **partial**. Prompt/reference conditioning, speaker-encoder
interior, sequential code generation/embedding feedback and the separately loaded
speech tokenizer/waveform decoder are unexpanded. They are not silently replaced
with a text language-model generation loop. The model-owned sidecar precedence
is unchanged. Speaker tensors retain native rank; convolution weights are not
flattened into matrices.

Selection checks the pinned configuration's structural options, with bounded
smaller compatible dimensions for deterministic tests. It rejects custom-voice,
voice-design, altered rotary/sliding attention and unknown structural options.
This is not a claim of support for every Qwen3-TTS or every PEFT adapter variant.
Text-table resolution is a separate capability and never selects an audio table.

## Local preparation and validation

Place a complete official checkpoint in a direct child named
`Qwen3-TTS-12Hz-1.7B-Base` under the configured model root. This basename preserves
the matching rule when upstream config lacks an organization-qualified identity.
In a separate direct child, place only the selected adapter's original
`adapter_config.json` and `adapter_model.safetensors`. Retain provenance/license
records outside the runtime inventory. Downloading is an explicit operator action;
the application remains offline, local, lazy and read-only.

Full local reference inspection observed separate selectable entries with no
adapter diagnostics, 480 native tensors in the Base and 942 in the composition.
The corresponding graphs contain 1,218/2,373 nodes and 480/942 parameter bindings,
with only the declared speech-pipeline coverage diagnostic. No full weights were
materialized to construct these graphs.

`tests/test_qwen3_tts.py` owns independent asymmetric reduced geometry, exact text
rows and unchanged base/factor values, both concrete stacks, LoRA operation
bindings, higher-rank availability and unsupported metadata/factor rejection.
The existing dense/LoRA/PEFT/embedding suites retain their independent shared
behavior and lifecycle checks. The initial focused run passed 328 cases; after
adding the base/config geometry guard, all 42 Qwen3-TTS/PEFT cases passed.
Repository-wide Ruff and strict mypy passed, as did the production frontend build
and 18 installed-wheel/CLI/runtime checks outside the source checkout.

Real HTTP checks passed for both downloaded variants: catalogue/session creation,
14-token multilingual tokenization, two exact-width 2048-column embedding rows,
native norm and adapter-factor binary streams with successful terminal frames,
and both prepared partial graphs. CORS was checked with the deployed frontend
origin. A real Chromium/WebGL2 session selected both models, expanded the graph,
opened a native Q/K norm and a real LoRA A factor in the Matrix Explorer modal,
and used the text tokenizer with no page errors. These observations establish
inspection only, not an executed speech model or audio quality.
