"""Bounded minimal-copy LMEX writer. No transport or numerical computation."""

import json
import struct
from enum import IntEnum

from .stream_metadata import METADATA, StreamError, StreamProgress

MAX_CONTROL_BYTES = 1024 * 1024
DEFAULT_DATA_BYTES = 256 * 1024


class FrameType(IntEnum):
    META_JSON = 1
    DATA = 2
    PROGRESS_JSON = 3
    COMPLETE = 4
    ERROR_JSON = 5
    CANCELLED = 6


# Separate header and payload avoids concatenating/copying numeric buffers.
Frame = tuple[bytes, bytes | memoryview]


def _json(value: object) -> bytes:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(payload) > MAX_CONTROL_BYTES:
        raise ValueError("control frame exceeds 1 MiB")
    return payload


def _frame(kind: FrameType, payload: bytes | memoryview = b"") -> Frame:
    return struct.pack("<4sBBHI", b"LMEX", kind, 0, 0, len(payload)), payload


class LMEXWriter:
    def __init__(self, max_data_bytes: int = DEFAULT_DATA_BYTES) -> None:
        if (
            type(max_data_bytes) is not int
            or not 0 < max_data_bytes <= DEFAULT_DATA_BYTES
            or max_data_bytes % 4
        ):
            raise ValueError("data bound must be a positive multiple of four up to 256 KiB")
        self.max_data_bytes = max_data_bytes
        self.expected: int | None = None
        self.sent = 0
        self.statistics = False
        self.terminal = False

    def _active(self, *, metadata: bool = False) -> None:
        if self.terminal:
            raise ValueError("stream already terminal")
        if metadata and self.expected is None:
            raise ValueError("metadata required")

    def metadata(self, value: object) -> Frame:
        self._active()
        if self.expected is not None:
            raise ValueError("duplicate metadata")
        # Serialize the original object so integer numeric scalars retain their
        # canonical fixture representation. Validation never coerces the input.
        parsed = METADATA.validate_python(value)
        payload = _json(value)
        self.expected = parsed.byte_length
        self.statistics = parsed.kind in ("tensor_statistics", "input_embeddings_statistics")
        return _frame(FrameType.META_JSON, payload)

    def data(self, value: bytes | memoryview) -> Frame:
        self._active(metadata=True)
        payload = memoryview(value).cast("B")
        assert self.expected is not None
        if self.statistics or len(payload) % 4 or len(payload) > self.max_data_bytes:
            raise ValueError("invalid DATA frame")
        if self.sent + len(payload) > self.expected:
            raise ValueError("DATA exceeds declared byte length")
        self.sent += len(payload)
        return _frame(FrameType.DATA, payload)

    def progress(self, value: object) -> Frame:
        self._active(metadata=True)
        StreamProgress.model_validate(value)
        return _frame(FrameType.PROGRESS_JSON, _json(value))

    def complete(self) -> Frame:
        self._active(metadata=True)
        if self.sent != self.expected:
            raise ValueError("incomplete DATA length")
        self.terminal = True
        return _frame(FrameType.COMPLETE)

    def error(self, value: object) -> Frame:
        self._active()
        StreamError.model_validate(value)
        payload = _json(value)
        self.terminal = True
        return _frame(FrameType.ERROR_JSON, payload)

    def cancelled(self) -> Frame:
        self._active()
        self.terminal = True
        return _frame(FrameType.CANCELLED)
