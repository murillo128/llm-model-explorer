"""Ephemeral comparison delivery; META is computed under the consumer lifetime."""

import array
import ctypes
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from ..artifacts import ArtifactSpec
from ..lmex import MAX_CONTROL_BYTES, LMEXWriter
from ..model_files import ModelError
from ..operations import MAX_READ_BYTES, Cancellation, Consumer, SourceReader
from ..sessions import SessionRegistry
from ..stream_metadata import NeuronComparisonOptions, NeuronLimits
from ..tokenization import load_tokenizer
from .activations import ActivationReader
from .admission import NeuronInput, Unavailable, admit
from .comparison import compare_neurons
from .runtime import NeuronLease, NeuronRuntime


@dataclass(frozen=True)
class Comparisons:
    selection: NeuronInput
    options: NeuronComparisonOptions
    model_root: Path
    identity: dict[str, Any]

    def compute(self, lease: NeuronLease) -> SourceReader:
        source = self.selection.model.source
        tokenizer = None
        if self.options.max_new_tokens:
            with source.local_directory() as directory:
                tokenizer = load_tokenizer(self.model_root, directory)
        result = compare_neurons(lease, self.selection, self.options, tokenizer)
        shape = list(result.probabilities.shape)
        length = 4 * result.probabilities.numel()  # At most 4 * 80 float32 values.
        metadata = dict(
            self.identity,
            candidate_token_ids=result.candidate_token_ids,
            top_token_ids=result.top_token_ids,
            continuations=result.continuations,
            sections=[dict(name="probabilities", shape=shape, offset=0, byte_length=length)],
            byte_length=length,
        )
        try:
            _, header = LMEXWriter().metadata(metadata)
        except ValueError:
            raise ModelError("unsupported_size", "Neuron metadata exceeds stream limits.") from None
        raw = ctypes.string_at(result.probabilities.data_ptr(), length)
        if sys.byteorder != "little":
            words = array.array("I", raw)
            words.byteswap()
            raw = words.tobytes()
        # As with embedding analysis, the direct reader owns an internal header
        # followed by DATA. This prefix is drained before LMEX META is emitted;
        # it is never a new wire frame or a persisted artifact.
        return ActivationReader(source, struct.pack("<I", len(header)) + bytes(header) + raw)

    async def metadata(self, consumer: Consumer) -> object:
        async def read_exact(length: int) -> bytes:
            result = bytearray()
            while len(result) < length:
                raw = await consumer.read(min(MAX_READ_BYTES, length - len(result)))
                if not raw:
                    raise ModelError("internal_error", "Incomplete comparison metadata.", 500)
                result.extend(raw)
            return bytes(result)

        length = struct.unpack("<I", await read_exact(4))[0]
        if not 0 < length <= MAX_CONTROL_BYTES:
            raise ModelError("unsupported_size", "Neuron metadata exceeds stream limits.")
        return json.loads(await read_exact(length))


async def subscribe_comparisons(
    sessions: SessionRegistry,
    runtime: NeuronRuntime,
    model_root: Path,
    session_id: UUID,
    model_revision: str,
    module_id: str,
    neuron_indices: tuple[int, ...],
    token_ids: tuple[int, ...],
    options: NeuronComparisonOptions,
) -> tuple[Comparisons, Consumer]:
    session = await sessions.get(session_id)

    def resolve() -> Comparisons:
        try:
            model = admit(session.source, model_root, runtime.device)
        except Unavailable:
            raise ModelError(
                "unsupported_representation", "Neuron execution is unavailable for this model."
            ) from None
        selection = NeuronInput(model, model_revision, module_id, neuron_indices, token_ids)
        selection.validate()
        try:
            options.check_limits(NeuronLimits.model_validate(model.limits), len(token_ids))
        except ValueError:
            raise ModelError("unsupported_size", "Comparison exceeds execution limits.") from None
        identity = dict(
            kind="neuron_comparison",
            session_id=str(session_id),
            model_id=session.source.model_id,
            model_revision=model_revision,
            execution=model.execution,
            limits=model.limits,
            module=next(m for m in model.modules if m["module_id"] == module_id),
            token_ids=list(token_ids),
            neuron_indices=list(neuron_indices),
            dtype="float32",
            byte_order="little",
            layout="c",
            **options.model_dump(),
        )
        if len(json.dumps(identity).encode("utf-8")) > MAX_CONTROL_BYTES:
            raise ModelError("unsupported_size", "Neuron metadata exceeds stream limits.")
        session.source.check_unchanged()
        return Comparisons(selection, options, model_root, identity)

    result = await sessions.work.run(resolve)
    sessions.require(session_id)

    async def prepare(cancellation: Cancellation) -> SourceReader:
        return await runtime.run(result.selection, cancellation, result.compute)

    # Direct-source bookkeeping only: output-dependent geometry lives in META,
    # never a cache entry or singleflight key. No prompt/condition persistence.
    spec = ArtifactSpec(
        model_fingerprint=session.source.fingerprint,
        source=module_id,
        operation="neuron_comparison",
        parameters={},
        dtype="float32",
        layout="c",
        shape=(0,),
        expected_bytes=0,
        producer=result.selection.model.execution["adapter_version"],
    )
    consumer = sessions.operations.subscribe_source(spec, prepare=prepare, session_id=session_id)
    consumer.read_guard = session.source.check_unchanged
    return result, consumer
