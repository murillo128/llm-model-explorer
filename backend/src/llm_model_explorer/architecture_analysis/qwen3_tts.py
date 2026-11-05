"""Reviewed speech-language components; generation/audio interiors stay explicit gaps."""

from ..qwen3_tts_config import SOURCE_REVISION, TTSConfig, checked, linear_dimensions
from . import records as r
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .dense import DenseGraph, shape
from .dense_config import DenseConfig
from .validation import require


class SpeechGraph(DenseGraph):
    def __init__(self, inputs: AnalysisInput, builder: GraphBuilder, config: TTSConfig):
        self.tts = config
        super().__init__(inputs, builder, config.talker)

    def linear_dimensions(self) -> dict[str, tuple[int, int]]:
        return linear_dimensions(self.tts)

    def port(
        self, name: str, dims: r.ArchitectureShape, output: bool = False
    ) -> r.ArchitecturePort:
        if dims is not None and self.layer_prefix == "talker.code_predictor.model":
            dims = [
                r.ArchitectureSymbolDimension(kind="symbol", name="C")
                if isinstance(d, r.ArchitectureSymbolDimension) and d.name == "S"
                else d
                for d in dims
            ]
        return super().port(name, dims, output)

    def stack(self, prefix: str, config: DenseConfig) -> str:
        self.layer_prefix, self.c = prefix, config
        hidden = shape("B", "S", config.hidden)
        positions = self.node(
            prefix + ".positions",
            prefix,
            "sequence_positions",
            {"x": hidden},
            shape("B", "S"),
            formula="positions = cumulative valid-token count - 1",
        )
        self.edge(prefix, positions, source_port="x")
        mask = self.node(
            prefix + ".mask",
            prefix,
            "causal_mask",
            {"x": hidden},
            shape("B", 1, "S", "S"),
            attributes={"causal": True},
        )
        self.edge(prefix, mask, source_port="x")
        rotary = {}
        for fn in ("cos", "sin"):
            rotary[fn] = self.node(
                prefix + "." + fn,
                prefix,
                "rotary_" + fn,
                {"positions": shape("B", "S")},
                shape("B", "S", config.head_dim),
                attributes={
                    "theta": config.theta,
                    "layout": "split_half",
                    "position_axes": "equal text axes"
                    if prefix == "talker.model"
                    else "one codebook axis",
                },
                formula=fn + "(concat(f, f)); f = positions * theta^(-2i/head_dim)",
            )
            self.edge(positions, rotary[fn], "positions")
        previous = prefix
        instances = []
        for index in range(config.layers):
            layer = self.layer(index)
            self.edge(previous, layer, source_port="x" if previous == prefix else "out")
            self.edge(mask, layer, "mask")
            for fn in rotary:
                self.edge(rotary[fn], layer, fn)
            instances.append(
                r.ArchitectureRepetitionInstance(
                    node_id=self.nid(layer), index=index, variant=prefix
                )
            )
            previous = layer
        norm = self.norm(prefix + ".norm", prefix, hidden, config.hidden)
        self.edge(previous, norm)
        self.edge(norm, prefix, "out")
        self.group(prefix, "speech", [self.port("x", hidden), self.port("out", hidden, True)])
        self.b.add_layer_repetition(
            r.ArchitectureRepetition(
                id=self.b.record_id("repetition", prefix),
                parent_id=self.nid(prefix),
                label="Talker layers" if prefix == "talker.model" else "Code predictor layers",
                instances=instances,
            )
        )
        self.layer_prefix = "model"
        return prefix

    def embedding(self, key: str, vocab: int, width: int, source: str) -> str:
        node = self.node(
            key,
            "speech",
            "embedding",
            {"x": shape("B", "S")},
            shape("B", "S", width),
            parameters=(self.parameter(key + ".weight", (vocab, width)),),
        )
        self.edge(source, node)
        return node

    def build(self) -> None:
        b, t = self.b, self.tts
        b.add_symbol("B", "Symbolic batch size; no inference performed.")
        b.add_symbol("S", "Talker conditioning/audio-frame sequence.")
        b.add_symbol("C", "Sequential residual-codebook predictor prefix, independent of S.")
        b.add_symbol("A", "Reference audio samples.")
        text = self.node("text", "speech", "text", {}, shape("B"), kind="input")
        audio = self.node(
            "reference_audio", "speech", "reference_audio", {}, shape("B", "A"), kind="input"
        )
        tokens = self.node(
            "text_tokens",
            "speech",
            "tokenizer",
            {"text": shape("B")},
            shape("B", "S"),
            label="Text tokenization and prompt formatting (context)",
        )
        self.edge(text, tokens, "text")
        embedded = self.embedding(
            "talker.model.text_embedding", t.text_vocab, t.text_hidden, tokens
        )
        projected = self.connect_linear(
            embedded,
            self.linear(
                "talker.text_projection.linear_fc1", "speech", t.text_hidden, t.text_hidden, True
            ),
        )
        activated = self.node(
            "talker.text_projection.silu",
            "speech",
            "silu",
            {"x": shape("B", "S", t.text_hidden)},
            shape("B", "S", t.text_hidden),
        )
        self.edge(projected, activated)
        projected = self.connect_linear(
            activated,
            self.linear(
                "talker.text_projection.linear_fc2", "speech", t.text_hidden, t.talker.hidden, True
            ),
        )
        speaker_params = tuple(
            b.native_parameter(name, name, shape(*storage.shape), self.provenance())
            for name, storage in self.inputs.bindings.physical.items()
            if name.startswith("speaker_encoder.")
        )
        self.used_storage.update(
            name for name in self.inputs.bindings.physical if name.startswith("speaker_encoder.")
        )
        speaker = self.node(
            "speaker_encoder",
            "speech",
            "speaker_encoder",
            {"audio": shape("B", "A")},
            shape("B", t.talker.hidden),
            parameters=speaker_params,
            label="Speaker encoder (unexpanded)",
        )
        self.edge(audio, speaker, "audio")
        conditioning = self.node(
            "conditioning",
            "speech",
            "speech_conditioning",
            {
                "text": shape("B", "S", t.talker.hidden),
                "speaker": shape("B", t.talker.hidden),
                "audio": shape("B", "A"),
            },
            shape("B", "S", t.talker.hidden),
            label="Prompt / reference codec conditioning (unexpanded)",
        )
        self.edge(projected, conditioning, "text")
        self.edge(speaker, conditioning, "speaker")
        self.edge(audio, conditioning, "audio")
        talker = self.stack("talker.model", t.talker)
        self.edge(conditioning, talker)
        self.c = t.talker
        head = self.connect_linear(
            talker,
            self.linear("talker.codec_head", "speech", t.talker.hidden, t.talker.vocab, False),
        )
        sampled = self.node(
            "first_codec",
            "speech",
            "codec_selection",
            {"logits": shape("B", "S", t.talker.vocab)},
            shape("B", "S"),
            label="First codebook selection (generation context)",
        )
        self.edge(head, sampled, "logits")
        codec = self.embedding(
            "talker.model.codec_embedding", t.talker.vocab, t.talker.hidden, sampled
        )
        prefix = self.node(
            "predictor_prefix",
            "speech",
            "codebook_prefix",
            {
                "hidden": shape("B", "S", t.talker.hidden),
                "first_code": shape("B", "S", t.talker.hidden),
            },
            shape("B", "C", t.talker.hidden),
            label="Per-frame sequential predictor prefix (context)",
        )
        self.edge(talker, prefix, "hidden")
        self.edge(codec, prefix, "first_code")
        # The predictor's embeddings are Talker-wide; projection precedes its stack.
        self.layer_prefix = "talker.code_predictor.model"
        if t.talker.hidden != t.predictor.hidden:
            prefix = self.connect_linear(
                prefix,
                self.linear(
                    "talker.code_predictor.small_to_mtp_projection",
                    "speech",
                    t.talker.hidden,
                    t.predictor.hidden,
                    True,
                ),
            )
        predictor = self.stack("talker.code_predictor.model", t.predictor)
        self.edge(prefix, predictor)
        self.layer_prefix = "talker.code_predictor.model"
        heads, embeddings = [], []
        for index in range(t.code_groups - 1):
            key = f"talker.code_predictor.lm_head.{index}"
            heads.append(self.parameter(key + ".weight", (t.predictor.vocab, t.predictor.hidden)))
            embeddings.append(
                self.parameter(
                    f"talker.code_predictor.model.codec_embedding.{index}.weight",
                    (t.predictor.vocab, t.talker.hidden),
                )
            )
        selected = self.node(
            "codebook_head",
            "speech",
            "indexed_linear",
            {"x": shape("B", "C", t.predictor.hidden)},
            shape("B", "C", t.predictor.vocab),
            parameters=tuple(heads),
            formula="logits = x @ W[generation_step].T",
            label="Select one residual codebook head",
            module_name="talker.code_predictor.lm_head",
            attributes={"head_count": float(t.code_groups - 1)},
        )
        self.edge(predictor, selected)
        self.layer_prefix = "model"
        feedback = self.node(
            "residual_codes",
            "speech",
            "sequential_codebook_generation",
            {"logits": shape("B", "C", t.predictor.vocab)},
            shape("B", "S", t.code_groups - 1),
            parameters=tuple(embeddings),
            label="Residual code selection / embedding feedback (unexpanded)",
            module_name="talker.code_predictor.model.codec_embedding",
        )
        self.edge(selected, feedback, "logits")
        self.layer_prefix = "model"
        codes = self.node(
            "audio_codes",
            "speech",
            "concat",
            {"first": shape("B", "S"), "residual": shape("B", "S", t.code_groups - 1)},
            shape("B", "S", t.code_groups),
            kind="output",
            label="Audio codec IDs (not text vocabulary logits)",
        )
        self.edge(sampled, codes, "first")
        self.edge(feedback, codes, "residual")
        self.group("speech", None, [])
        b.diagnose(
            r.ArchitectureDiagnostic(
                code="speech_pipeline_context",
                message=(
                    "Partial static speech-language view: prompt/reference conditioning, "
                    "speaker encoder, sequential generation/feedback and external speech "
                    "tokenizer/waveform decoder are unexpanded. "
                    "No audio inference or voice cloning is performed."
                ),
            )
        )
        if set(self.inputs.bindings.physical) - self.used_storage:
            b.diagnose(
                r.ArchitectureDiagnostic(
                    code="unrecognized_storage",
                    message="Storage outside the reviewed Qwen3-TTS components remains unmodeled.",
                )
            )


def build(inputs: AnalysisInput, builder: GraphBuilder) -> None:
    config = checked(inputs.configuration)
    require(config is not None, "Unsupported Qwen3-TTS Base configuration.")
    assert config is not None
    SpeechGraph(inputs, builder, config).build()


def register(registry: DescriptionRegistry) -> None:
    registry.register(
        Description(
            Producer("qwen3-tts-base", "1", SOURCE_REVISION),
            "language_model",
            frozenset({"qwen3_tts"}),
            frozenset({"Qwen3TTSForConditionalGeneration"}),
            lambda inputs: checked(inputs.configuration) is not None,
            build,
        )
    )
