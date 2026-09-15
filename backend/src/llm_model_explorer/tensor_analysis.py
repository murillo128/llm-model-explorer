"""Exact scientific artifacts over one logical tensor, independent of HTTP delivery."""

import array
import ctypes
import math
import re
import struct
import sys
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import torch

from .artifacts import ArtifactSpec
from .materialization import LogicalTensor, LogicalTensorService
from .model_files import ModelError
from .operations import MAX_READ_BYTES, Cancellation, Consumer, ProducerContext
from .sessions import SessionRegistry
from .tensor_source import safe_integer

Kind = Literal["tensor_statistics", "tensor_distributions"]
PERCENTILES = ("p01", "p05", "p50", "p95", "p99")
FRACTIONS = (0.01, 0.05, 0.50, 0.95, 0.99)
STATISTICS = struct.Struct("<3Q9d")
DOMAIN = struct.Struct("<2d")
BIN_COUNT = 100
UINT32_MAX = 2**32 - 1
WORK_ELEMENTS = MAX_READ_BYTES // 4
CPU_ALLOCATION_FAILURE = re.compile(r"DefaultCPUAllocator: can't allocate memory:.*Error code 12\b")


def _finite_mean(values: torch.Tensor, cancellation: Cancellation) -> float:
    """Exact F32 exponent buckets preserve small residuals between huge values.

    Each native block sums at most 2**16 signed 24-bit significands into int64.
    Only 277 exponent totals cross to Python, never individual tensor weights.
    Python integers combine blocks without overflow; fsum combines exact binary
    limbs before the final float64 division. Plain F64 sum/var_mean can lose the
    1 in [max_f32, 1, -max_f32], violating the near-zero mean tolerance.
    """
    totals = [0] * 277  # frexp exponents -148 (smallest subnormal) through 128.
    for start in range(0, values.numel(), WORK_ELEMENTS):
        cancellation.check()
        mantissas, exponents = torch.frexp(values[start : start + WORK_ELEMENTS])
        significands = (mantissas.to(torch.float64) * 2**24).to(torch.int64)
        sums = torch.zeros(277, dtype=torch.int64, device=values.device)
        sums.scatter_add_(0, (exponents + 148).to(torch.int64), significands)
        for index, subtotal in enumerate(sums.tolist()):
            totals[index] += subtotal
    terms: list[float] = []
    for index, total in enumerate(totals):
        high, low = divmod(total, 2**32)
        terms.extend((math.ldexp(high, index - 140), math.ldexp(low, index - 172)))
    return math.fsum(terms) / values.numel()


def statistics(values: torch.Tensor, cancellation: Cancellation) -> bytes:
    """Native exact sort avoids torch.quantile's 2**24 input-size boundary."""
    count = values.numel()
    finite = values[torch.isfinite(values)]
    n = finite.numel()
    if not n:
        return STATISTICS.pack(count, 0, count, *([math.nan] * 9))
    cancellation.check()
    # Wide Welford population variance; no F32 sums or squared differences.
    wide = finite.to(torch.float64)
    variance = torch.var(wide, correction=0)
    del wide
    mean = _finite_mean(finite, cancellation)
    cancellation.check()
    ordered = torch.sort(finite).values
    del finite
    cancellation.check()
    positions = torch.tensor(FRACTIONS, dtype=torch.float64, device=values.device) * (n - 1)
    lower = positions.floor().to(torch.int64)
    upper = positions.ceil().to(torch.int64)
    fraction = positions - lower
    left, right = ordered[lower].to(torch.float64), ordered[upper].to(torch.float64)
    # Equivalent linear interpolation; preserve equal endpoints exactly so a
    # constant tensor cannot acquire out-of-range percentiles through rounding.
    quantiles = left + fraction * (right - left)
    return STATISTICS.pack(
        count,
        n,
        count - n,
        ordered[0].item(),
        ordered[-1].item(),
        mean,
        variance.sqrt().item(),
        *quantiles.tolist(),
    )


def checked_uint32(counts: torch.Tensor) -> torch.Tensor:
    if counts.numel() and (counts.min().item() < 0 or counts.max().item() > UINT32_MAX):
        raise ModelError("unsupported_size", "A distribution count exceeds uint32.")
    return counts.to(dtype=torch.uint32, device="cpu")


def distributions(
    values: torch.Tensor, rows: int, columns: int, cancellation: Cancellation
) -> tuple[bytes, torch.Tensor]:
    """Bounded native binning blocks; int64 accumulators cannot wrap uint32."""
    counts = torch.zeros(BIN_COUNT * (rows + columns), dtype=torch.int64, device=values.device)
    lo, hi = math.inf, -math.inf
    for start in range(0, values.numel(), WORK_ELEMENTS):
        cancellation.check()
        block = values[start : start + WORK_ELEMENTS]
        finite = block[torch.isfinite(block)]
        if finite.numel():
            lo = min(lo, finite.min().item())
            hi = max(hi, finite.max().item())
    if lo == math.inf:
        return DOMAIN.pack(math.nan, math.nan), checked_uint32(counts)
    for start in range(0, values.numel(), WORK_ELEMENTS):
        cancellation.check()
        block = values[start : start + WORK_ELEMENTS]
        mask = torch.isfinite(block)
        offsets = torch.arange(start, start + block.numel(), device=values.device)[mask]
        finite = block[mask].to(torch.float64)
        bins = (
            torch.full_like(offsets, 50)
            if lo == hi
            else ((finite - lo) / (hi - lo) * BIN_COUNT).floor().clamp_(0, 99).to(torch.int64)
        )
        ones = torch.ones_like(bins)
        counts.scatter_add_(0, offsets.div(columns, rounding_mode="floor") * BIN_COUNT + bins, ones)
        counts.scatter_add_(0, rows * BIN_COUNT + bins * columns + offsets % columns, ones)
    cancellation.check()
    return DOMAIN.pack(lo, hi), checked_uint32(counts)


def _copy_block(destination: torch.Tensor, offset: int, raw: bytes) -> None:
    words = bytearray(raw)
    if sys.byteorder != "little":
        swapped = array.array("I", words)
        swapped.byteswap()
        words = bytearray(swapped.tobytes())
    block = torch.frombuffer(words, dtype=torch.float32)
    destination[offset : offset + block.numel()].copy_(block)


def _count_bytes(counts: torch.Tensor) -> bytes:
    raw = ctypes.string_at(counts.data_ptr(), counts.numel() * 4)
    if sys.byteorder != "little":
        words = array.array("I", raw)
        words.byteswap()
        raw = words.tobytes()
    return raw


def statistics_fields(raw: bytes) -> dict[str, object]:
    count, finite, nonfinite, *scalars = STATISTICS.unpack(raw)
    fields = [None if finite == 0 else value for value in scalars]
    return dict(
        count=count,
        finite_count=finite,
        non_finite_count=nonfinite,
        minimum=fields[0],
        maximum=fields[1],
        mean=fields[2],
        stddev=fields[3],
        percentiles=dict(zip(PERCENTILES, fields[4:], strict=True)),
        byte_length=0,
    )


def distribution_fields(raw: bytes, rows: int, columns: int) -> dict[str, object]:
    lo, hi = DOMAIN.unpack(raw)
    row_bytes, column_bytes = rows * BIN_COUNT * 4, columns * BIN_COUNT * 4
    return dict(
        rows=rows,
        columns=columns,
        bin_count=BIN_COUNT,
        binning="linear-full-range",
        domain_minimum=None if math.isnan(lo) else lo,
        domain_maximum=None if math.isnan(hi) else hi,
        dtype="uint32",
        byte_order="little",
        sections=[
            dict(name="row_counts", shape=[rows, BIN_COUNT], offset=0, byte_length=row_bytes),
            dict(
                name="column_counts",
                shape=[BIN_COUNT, columns],
                offset=row_bytes,
                byte_length=column_bytes,
            ),
        ],
        byte_length=row_bytes + column_bytes,
    )


def calculate(
    values: torch.Tensor,
    kind: Kind,
    shape: tuple[int, ...],
    device: str,
    cancellation: Cancellation,
) -> tuple[bytes, torch.Tensor | None]:
    try:
        values = values.to(device)
        cancellation.check()
        if kind == "tensor_statistics":
            return statistics(values, cancellation), None
        rows, columns = shape
        return distributions(values, rows, columns, cancellation)
    finally:
        # The queue slot owns all kernels, including cancellation/error paths.
        if device != "cpu":
            torch.cuda.synchronize(device)


@dataclass(frozen=True)
class TensorAnalysis:
    tensor: LogicalTensor
    tensors: LogicalTensorService
    kind: Kind
    spec: ArtifactSpec

    def _calculate(
        self, values: torch.Tensor, cancellation: Cancellation
    ) -> tuple[bytes, torch.Tensor | None]:
        return calculate(
            values,
            self.kind,
            self.tensor.descriptor.shape,
            self.tensors.runtime.device,
            cancellation,
        )

    async def produce(self, context: ProducerContext) -> None:
        try:
            await self._produce(context)
        except RuntimeError as exc:
            # The pinned native CPU allocator reports ENOMEM as RuntimeError,
            # unlike CUDA's typed OOM. Preserve every unrelated runtime failure.
            if not isinstance(exc, torch.OutOfMemoryError) and not CPU_ALLOCATION_FAILURE.search(
                str(exc)
            ):
                raise
            raise ModelError(
                "resource_exhausted", "Insufficient memory for tensor analysis.", 503
            ) from exc

    async def _produce(self, context: ProducerContext) -> None:
        # Dependencies finish before the device queue is acquired. Only the
        # requested tensor is resident; no chunk lists or whole-model loading.
        values = await context.io(
            lambda: torch.empty(self.tensor.descriptor.numel, dtype=torch.float32)
        )
        offset = 0
        async with self.tensors.dependency(context, self.tensor) as consumer:
            while raw := await consumer.read():
                await context.io(_copy_block, values, offset, raw)
                offset += len(raw) // 4
        if offset != values.numel():
            raise ValueError("incomplete logical tensor dependency")
        header, counts = await context.compute(self._calculate, values, context.cancellation)
        del values
        await context.io(self.tensor.source.check_unchanged)
        await context.append(header)
        if counts is not None:
            for start in range(0, counts.numel(), WORK_ELEMENTS):
                block = counts[start : start + WORK_ELEMENTS]
                await context.append(await context.io(_count_bytes, block))
        await context.io(self.tensor.source.check_unchanged)

    async def metadata(self, consumer: Consumer) -> object:
        size = STATISTICS.size if self.kind == "tensor_statistics" else DOMAIN.size
        raw = bytearray()
        while len(raw) < size:
            block = await consumer.read(size - len(raw))
            if not block:
                raise ValueError("incomplete analysis header")
            raw.extend(block)
        tensor_id = self.tensor.descriptor.id
        if self.kind == "tensor_statistics":
            # Wait for publication before the metadata-only result succeeds.
            if await consumer.read():
                raise ValueError("unexpected statistics payload")
            fields = statistics_fields(bytes(raw))
        else:
            fields = distribution_fields(bytes(raw), *self.tensor.descriptor.shape)
        return dict(kind=self.kind, tensor_id=tensor_id, **fields)


async def subscribe_analysis(
    tensors: LogicalTensorService,
    sessions: SessionRegistry,
    session_id: UUID,
    tensor_id: str,
    kind: Kind,
) -> tuple[TensorAnalysis, Consumer]:
    session = await sessions.get(session_id)
    tensor = await tensors.runtime.work.run(tensors.resolve, session.source, tensor_id)
    if kind == "tensor_distributions" and tensor.descriptor.rank != 2:
        raise ModelError("unsupported_rank", "Distributions require a rank-2 tensor.")
    expected = STATISTICS.size
    if kind == "tensor_distributions":
        expected = safe_integer(DOMAIN.size + safe_integer(sum(tensor.descriptor.shape) * 400))
    spec = ArtifactSpec(
        model_fingerprint=session.source.fingerprint,
        source=tensor_id,
        operation=kind,
        parameters={"logical_tensor_key": tensor.spec.key},
        dtype="statistics-f64" if kind == "tensor_statistics" else "domain-f64-counts-u32",
        layout="c",
        shape=tensor.descriptor.shape,
        expected_bytes=expected,
        producer="torch-exact-analysis-v1",
    )
    analysis = TensorAnalysis(tensor, tensors, kind, spec)
    consumer = await sessions.subscribe(session_id, spec, analysis.produce)
    consumer.read_guard = session.source.check_unchanged
    return analysis, consumer
