"""Request-owned activation delivery through the existing operation lifecycle."""

import array
import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from ..artifacts import ArtifactSpec
from ..lmex import LMEXWriter
from ..model_files import ModelError
from ..operations import Cancellation, Consumer, SourceReader
from ..sessions import SessionRegistry
from ..tensor_source import ModelSource
from .admission import NeuronInput, Unavailable, admit
from .capture import capture_activations
from .runtime import NeuronLease, NeuronRuntime


class ActivationReader:
    def __init__(self, source: ModelSource, data: bytes) -> None:
        self.source = source
        self.data = data
        self.offset = 0

    def read_available(self, max_bytes: int) -> bytes:
        self.source.check_unchanged()
        result = self.data[self.offset : self.offset + max_bytes]
        self.offset += len(result)
        self.source.check_unchanged()  # Including the final read before COMPLETE.
        return result

    def close(self) -> None:
        self.data = b""


@dataclass(frozen=True)
class Activations:
    selection: NeuronInput
    metadata: dict[str, Any]
    spec: ArtifactSpec

    def compute(self, lease: NeuronLease) -> SourceReader:
        values = capture_activations(lease, self.selection)
        raw = ctypes.string_at(values.data_ptr(), values.numel() * 4)
        if sys.byteorder != "little":
            words = array.array("I", raw)
            words.byteswap()
            raw = words.tobytes()
        return ActivationReader(self.selection.model.source, raw)


async def subscribe_activations(
    sessions: SessionRegistry,
    runtime: NeuronRuntime,
    model_root: Path,
    session_id: UUID,
    model_revision: str,
    module_id: str,
    neuron_indices: tuple[int, ...],
    token_ids: tuple[int, ...],
) -> tuple[Activations, Consumer]:
    session = await sessions.get(session_id)

    def resolve() -> Activations:
        try:
            model = admit(session.source, model_root, runtime.device)
        except Unavailable:
            raise ModelError(
                "unsupported_representation", "Neuron execution is unavailable for this model."
            ) from None
        selection = NeuronInput(model, model_revision, module_id, neuron_indices, token_ids)
        selection.validate()
        shape = (len(token_ids), len(neuron_indices))
        length = 4 * shape[0] * shape[1]  # Bounded by preflight to 512 * 8 F32 scalars.
        metadata = dict(
            kind="neuron_activations",
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
            sections=[dict(name="activations", shape=list(shape), offset=0, byte_length=length)],
            byte_length=length,
        )
        try:
            LMEXWriter().metadata(metadata)
        except ValueError:
            raise ModelError("unsupported_size", "Neuron metadata exceeds stream limits.") from None
        # Direct delivery only: this descriptor never enters cache/singleflight.
        spec = ArtifactSpec(
            model_fingerprint=session.source.fingerprint,
            source=module_id,
            operation="neuron_activations",
            parameters={},
            dtype="float32",
            layout="c",
            shape=shape,
            expected_bytes=length,
            producer=model.execution["adapter_version"],
        )
        session.source.check_unchanged()
        return Activations(selection, metadata, spec)

    result = await sessions.work.run(resolve)
    sessions.require(session_id)

    async def prepare(cancellation: Cancellation) -> SourceReader:
        return await runtime.run(result.selection, cancellation, result.compute)

    consumer = sessions.operations.subscribe_source(
        result.spec, prepare=prepare, session_id=session_id
    )
    consumer.read_guard = session.source.check_unchanged
    return result, consumer
