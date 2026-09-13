"""Complete logical tensors shared by HTTP and numerical artifact producers."""

import array
import ctypes
import errno
import math
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import UUID

from .artifacts import ArtifactSpec
from .model_files import ModelError
from .operations import MAX_READ_BYTES, Consumer, OperationRuntime, ProducerContext
from .sessions import SessionRegistry
from .tensor_source import ModelSource, TensorDescriptor, safe_integer

CHUNK_ELEMENTS = MAX_READ_BYTES // 4


class _TensorReader:
    """One bounded native source block, never a tensor-sized Python allocation."""

    def __init__(self, source: ModelSource, tensor_id: str) -> None:
        self.source = source
        self.iterator = source.iter_tensor(tensor_id, chunk_elements=CHUNK_ELEMENTS)
        self.pending = b""

    def read_available(self, max_bytes: int) -> bytes:
        self.source.check_unchanged()
        if not self.pending:
            values = next(self.iterator, None)
            if values is None:
                return b""
            # Copy raw native float32 memory without a NumPy dependency or a
            # Python numeric loop. F32 bit patterns (including NaNs) stay intact.
            raw = ctypes.string_at(values.data_ptr(), values.numel() * 4)
            if sys.byteorder != "little":
                words = array.array("I", raw)
                words.byteswap()
                raw = words.tobytes()
            self.pending = raw
        data, self.pending = self.pending[:max_bytes], self.pending[max_bytes:]
        self.source.check_unchanged()
        return data

    def close(self) -> None:
        self.iterator.close()
        self.pending = b""


@dataclass(frozen=True)
class LogicalTensor:
    """Validated identity, descriptor and reusable producer; no payload in RAM."""

    source: ModelSource
    descriptor: TensorDescriptor
    spec: ArtifactSpec

    @property
    def direct(self) -> bool:
        return self.descriptor.storage_dtype == "F32"

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "tensor",
            "tensor_id": self.descriptor.id,
            "name": self.descriptor.name,
            "shape": list(self.descriptor.shape),
            "dtype": "float32",
            "byte_order": "little",
            "layout": "c",
            "byte_length": self.spec.expected_bytes,
        }

    def reader(self) -> _TensorReader:
        return _TensorReader(self.source, self.descriptor.id)

    async def produce(self, context: ProducerContext) -> None:
        reader = self.reader()
        try:
            while data := await context.io(reader.read_available, MAX_READ_BYTES):
                await context.append(data)
            # Exhausting the source iterator includes its final snapshot check.
            await context.io(self.source.check_unchanged)
        except OSError as exc:
            if exc.errno in {errno.ENOSPC, errno.EDQUOT, errno.ENOMEM}:
                raise ModelError(
                    "resource_exhausted", "Insufficient resources to materialize tensor.", 503
                ) from exc
            raise
        finally:
            # Cleanup must run even when the producer cancellation token is set.
            await context.runtime.work.run(reader.close)


class LogicalTensorService:
    def __init__(self, runtime: OperationRuntime) -> None:
        self.runtime = runtime

    def resolve(self, source: ModelSource, tensor_id: str) -> LogicalTensor:
        """Blocking preflight, also available without a session or HTTP request."""
        descriptor = next((item for item in source.tensors() if item.id == tensor_id), None)
        if descriptor is None:
            raise ModelError("tensor_not_found", "Unknown tensor.", 404)
        if (
            descriptor.storage_format != "safetensors"
            or descriptor.storage_dtype not in {"F32", "F16", "BF16"}
            or descriptor.logical_dtype != "float32"
        ):
            raise ModelError("unsupported_representation", "Unsupported tensor representation.")
        shape = tuple(safe_integer(dimension) for dimension in descriptor.shape)
        numel = safe_integer(math.prod(shape))
        if descriptor.rank != len(shape) or descriptor.numel != numel:
            raise ModelError("validation_error", "Inconsistent tensor dimensions.")
        spec = ArtifactSpec(
            model_fingerprint=source.fingerprint,
            source=descriptor.id,
            operation="logical_tensor",
            parameters={"storage_dtype": descriptor.storage_dtype, "byte_order": "little"},
            dtype="float32",
            layout="c",
            shape=shape,
            expected_bytes=safe_integer(numel * 4),
            producer="safetensors-native-float32-v1",
        )
        return LogicalTensor(source, descriptor, spec)

    async def subscribe(
        self, sessions: SessionRegistry, session_id: UUID, tensor_id: str
    ) -> tuple[LogicalTensor, Consumer]:
        session = await sessions.get(session_id)
        tensor = await self.runtime.work.run(self.resolve, session.source, tensor_id)
        sessions.require(session_id)
        if tensor.direct:
            consumer = self.runtime.subscribe_source(
                tensor.spec, tensor.reader, session_id=session_id
            )
        else:
            consumer = await sessions.subscribe(session_id, tensor.spec, tensor.produce)
        consumer.read_guard = session.source.check_unchanged
        return tensor, consumer

    @asynccontextmanager
    async def dependency(
        self, context: ProducerContext, tensor: LogicalTensor
    ) -> AsyncIterator[Consumer]:
        """Bounded logical bytes with parent cancellation and shared conversion.

        Numerical producers acquire this before device compute, read to EOF for
        validated completion, and release it on exit. No HTTP loopback is needed.
        """
        await context.io(tensor.source.check_unchanged)
        if tensor.direct:
            consumer = self.runtime.subscribe_source(
                tensor.spec, tensor.reader, parent=context.cancellation
            )
            try:
                yield consumer
            finally:
                await consumer.aclose()
        else:
            async with context.dependency(tensor.spec, tensor.produce) as consumer:
                consumer.read_guard = tensor.source.check_unchanged
                yield consumer
