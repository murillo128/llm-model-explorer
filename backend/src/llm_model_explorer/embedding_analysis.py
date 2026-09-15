"""Ephemeral scientific analysis of exactly the requested input-embedding rows."""

import math
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import torch

from .artifacts import ArtifactSpec
from .embeddings import InputEmbeddings, resolve_embeddings
from .lmex import LMEXWriter
from .model_files import ModelError
from .operations import MAX_READ_BYTES, Cancellation, Consumer, OperationRuntime, SourceReader
from .sessions import SessionRegistry
from .tensor_analysis import (
    CPU_ALLOCATION_FAILURE,
    DOMAIN,
    STATISTICS,
    _copy_block,
    _count_bytes,
    calculate,
    distribution_fields,
    statistics_fields,
)
from .tensor_source import ModelSource, safe_integer

Kind = Literal["input_embeddings_statistics", "input_embeddings_distributions"]


class _AnalysisReader:
    """One request-owned result; counts are copied only in bounded delivery chunks."""

    def __init__(self, source: ModelSource, header: bytes, counts: torch.Tensor | None) -> None:
        self.source = source
        self.header = header
        self.counts = counts
        self.offset = 0

    def read_available(self, max_bytes: int) -> bytes:
        self.source.check_unchanged()
        if self.header:
            raw, self.header = self.header[:max_bytes], self.header[max_bytes:]
        elif self.counts is not None and self.offset < self.counts.numel():
            length = min(max_bytes // 4, self.counts.numel() - self.offset)
            if not length:
                raise ValueError("count reads require at least four bytes")
            raw = _count_bytes(self.counts[self.offset : self.offset + length])
            self.offset += length
        else:
            raw = b""
        self.source.check_unchanged()
        return raw

    def close(self) -> None:
        self.header = b""
        self.counts = None


@dataclass(frozen=True)
class EmbeddingAnalysis:
    embeddings: InputEmbeddings
    kind: Kind

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.embeddings.token_ids), self.embeddings.table.shape[1]

    def metadata_fields(self, header: bytes) -> dict[str, object]:
        rows, columns = self.shape
        if self.kind == "input_embeddings_statistics":
            fields = dict(shape=[rows, columns], **statistics_fields(header))
        else:
            fields = distribution_fields(header, rows, columns)
        return dict(kind=self.kind, token_ids=list(self.embeddings.token_ids), **fields)

    def preflight(self) -> None:
        """Bound result sizes and ordered control identity before allocation."""
        rows, columns = self.shape
        count = safe_integer(rows * columns)
        # Buffers use F32 values plus existing native F64/int64 analysis workspaces.
        safe_integer(count * 8)
        if self.kind == "input_embeddings_statistics":
            header = STATISTICS.pack(count, 0, count, *([math.nan] * 9))
        else:
            safe_integer((rows + columns) * 100 * 8)
            header = DOMAIN.pack(math.nan, math.nan)
        self.checked_metadata(header)

    def checked_metadata(self, header: bytes) -> dict[str, object]:
        metadata = self.metadata_fields(header)
        try:
            LMEXWriter().metadata(metadata)
        except ValueError as exc:
            raise ModelError(
                "unsupported_size", "Input embedding analysis exceeds stream limits."
            ) from exc
        return metadata

    async def prepare(self, runtime: OperationRuntime, cancellation: Cancellation) -> SourceReader:
        # Called only by Consumer's shielded read task. Session/HTTP cancellation
        # drains each blocking call before closing readers or releasing buffers.
        cancellation.check()
        try:
            reader = await runtime.work.run(self.embeddings.reader)
            try:
                cancellation.check()
                values = await runtime.work.run(
                    lambda: torch.empty(
                        self.embeddings.spec.expected_bytes // 4, dtype=torch.float32
                    )
                )
                offset = 0
                while True:
                    cancellation.check()
                    raw = await runtime.work.run(reader.read_available, MAX_READ_BYTES)
                    cancellation.check()
                    if not raw:
                        break
                    if len(raw) % 4 or offset + len(raw) // 4 > values.numel():
                        raise ValueError("invalid input embedding dependency length")
                    await runtime.work.run(_copy_block, values, offset, raw)
                    offset += len(raw) // 4
                if offset != values.numel():
                    raise ValueError("incomplete input embedding dependency")
            finally:
                await runtime.work.run(reader.close)
            # No device slot is held while gathering the dependency.
            header, counts = await runtime.scheduler.run(
                runtime.device,
                cancellation,
                calculate,
                values,
                "tensor_statistics"
                if self.kind == "input_embeddings_statistics"
                else "tensor_distributions",
                self.shape,
                runtime.device,
                cancellation,
            )
            del values
            await runtime.work.run(self.embeddings.source.check_unchanged)
            cancellation.check()
            return _AnalysisReader(self.embeddings.source, header, counts)
        except RuntimeError as exc:
            if not isinstance(exc, torch.OutOfMemoryError) and not CPU_ALLOCATION_FAILURE.search(
                str(exc)
            ):
                raise
            raise ModelError(
                "resource_exhausted", "Insufficient memory for input embedding analysis.", 503
            ) from exc

    async def metadata(self, consumer: Consumer) -> object:
        size = STATISTICS.size if self.kind == "input_embeddings_statistics" else DOMAIN.size
        header = bytearray()
        while len(header) < size:
            raw = await consumer.read(size - len(header))
            if not raw:
                raise ValueError("incomplete input embedding analysis header")
            header.extend(raw)
        # Leave the consumer active through META so cancellation remains observable.
        return self.checked_metadata(bytes(header))


async def subscribe_embedding_analysis(
    sessions: SessionRegistry,
    session_id: UUID,
    token_ids: tuple[int, ...],
    kind: Kind,
) -> tuple[EmbeddingAnalysis, Consumer]:
    session = await sessions.get(session_id)

    def resolve() -> EmbeddingAnalysis:
        analysis = EmbeddingAnalysis(resolve_embeddings(session.source, token_ids), kind)
        analysis.preflight()
        session.source.check_unchanged()
        return analysis

    analysis = await sessions.work.run(resolve)
    sessions.require(session_id)

    async def prepare(cancellation: Cancellation) -> SourceReader:
        return await analysis.prepare(sessions.operations, cancellation)

    # Direct sources never touch the artifact store, including successful results.
    spec = ArtifactSpec(
        model_fingerprint=session.source.fingerprint,
        source=analysis.embeddings.table.id,
        operation=kind,
        parameters={"input_rows_key": analysis.embeddings.spec.key},
        dtype="statistics-f64"
        if kind == "input_embeddings_statistics"
        else "domain-f64-counts-u32",
        layout="c",
        shape=analysis.shape,
        expected_bytes=STATISTICS.size
        if kind == "input_embeddings_statistics"
        else safe_integer(DOMAIN.size + sum(analysis.shape) * 400),
        producer="torch-exact-input-rows-analysis-v1",
    )
    consumer = sessions.operations.subscribe_source(spec, prepare=prepare, session_id=session_id)
    consumer.read_guard = session.source.check_unchanged
    return analysis, consumer
