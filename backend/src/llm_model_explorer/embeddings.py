"""Explicit input-table resolution and ephemeral bounded row delivery."""

import hashlib
import json
from dataclasses import dataclass
from uuid import UUID

from .artifacts import ArtifactSpec
from .lmex import LMEXWriter
from .materialization import CHUNK_ELEMENTS, _TensorReader
from .model_files import ModelError
from .operations import Consumer
from .sessions import SessionRegistry
from .tensor_source import MAX_SAFE_INTEGER, ModelSource, TensorDescriptor, safe_integer


def unsupported() -> ModelError:
    return ModelError(
        "unsupported_representation", "A supported input embedding source could not be resolved."
    )


def resolve_input_table(source: ModelSource) -> TensorDescriptor:
    """Resolve an architecture-owned input table in the actionable logical inventory.

    Require agreement between local config and the exact architecture-owned
    tensor key/dimensions. Custom code mappings and ambiguous architectures are
    unsupported; output heads and similarly named weights are never fallbacks.
    """
    config = source.configuration()
    if config.get("auto_map"):
        raise unsupported()
    mappings = (
        ("llama", "LlamaForCausalLM", "model.embed_tokens.weight", None),
        ("qwen3", "Qwen3ForCausalLM", "model.embed_tokens.weight", None),
        (
            "qwen3_5",
            "Qwen3_5ForConditionalGeneration",
            "model.language_model.embed_tokens.weight",
            "text_config",
        ),
    )
    matches = [
        (name, nested)
        for model_type, architecture, name, nested in mappings
        if config.get("model_type") == model_type and config.get("architectures") == [architecture]
    ]
    if len(matches) != 1:
        raise unsupported()
    name, nested = matches[0]
    dimensions = config if nested is None else config.get(nested)
    if not isinstance(dimensions, dict) or dimensions.get("auto_map"):
        raise unsupported()
    vocab, hidden = dimensions.get("vocab_size"), dimensions.get("hidden_size")
    if (
        type(vocab) is not int
        or type(hidden) is not int
        or not 0 < vocab <= MAX_SAFE_INTEGER
        or not 0 < hidden <= MAX_SAFE_INTEGER
    ):
        raise unsupported()
    tables = [t for t in source.tensors() if t.name == name]
    if len(tables) != 1:
        raise unsupported()
    table = tables[0]
    if (
        table.shape != (vocab, hidden)
        or table.rank != 2
        or table.numel != vocab * hidden
        or table.logical_dtype != "float32"
    ):
        raise unsupported()
    # Admission owns numeric availability; iter_rows owns physical decoding.
    # Quantization elsewhere in the checkpoint does not disable a native table.
    source.check_unchanged()
    return table


@dataclass(frozen=True)
class InputEmbeddings:
    source: ModelSource
    table: TensorDescriptor
    token_ids: tuple[int, ...]
    spec: ArtifactSpec

    def metadata(self) -> dict[str, object]:
        return dict(
            kind="input_embeddings",
            token_ids=list(self.token_ids),
            shape=[len(self.token_ids), self.table.shape[1]],
            dtype="float32",
            byte_order="little",
            layout="c",
            byte_length=self.spec.expected_bytes,
        )

    def reader(self) -> _TensorReader:
        return _TensorReader(
            self.source,
            self.table.id,
            iterator=self.source.iter_rows(
                self.table.id, self.token_ids, chunk_elements=CHUNK_ELEMENTS
            ),
        )


def resolve_embeddings(source: ModelSource, token_ids: tuple[int, ...]) -> InputEmbeddings:
    """Blocking preflight: validate the whole request before allocating a consumer."""
    table = resolve_input_table(source)
    if any(type(token) is not int or not 0 <= token < table.shape[0] for token in token_ids):
        raise ModelError("validation_error", "Invalid input token IDs.")
    shape = (safe_integer(len(token_ids)), table.shape[1])
    length = safe_integer(shape[0] * shape[1] * 4)
    # The runtime uses this descriptor for direct delivery only; it is never
    # submitted to the artifact store or a singleflight producer.
    spec = ArtifactSpec(
        model_fingerprint=source.fingerprint,
        source=table.id,
        operation="input_embeddings",
        parameters={
            "token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()
            ).hexdigest()
        },
        dtype="float32",
        layout="c",
        shape=shape,
        expected_bytes=length,
        producer="safetensors-input-rows-v1",
    )
    result = InputEmbeddings(source, table, token_ids, spec)
    try:
        LMEXWriter().metadata(result.metadata())
    except ValueError as exc:
        raise ModelError(
            "unsupported_size", "Input embedding metadata exceeds stream limits."
        ) from exc
    source.check_unchanged()
    return result


async def subscribe_embeddings(
    sessions: SessionRegistry, session_id: UUID, token_ids: tuple[int, ...]
) -> tuple[InputEmbeddings, Consumer]:
    session = await sessions.get(session_id)
    result = await sessions.work.run(resolve_embeddings, session.source, token_ids)
    sessions.require(session_id)
    consumer = sessions.operations.subscribe_source(
        result.spec, result.reader, session_id=session_id
    )
    consumer.read_guard = session.source.check_unchanged
    return result, consumer
