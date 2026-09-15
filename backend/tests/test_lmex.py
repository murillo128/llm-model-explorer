"""Independent shared contract oracles exercise the server writer, not a decoder."""

import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from llm_model_explorer.lmex import MAX_CONTROL_BYTES, LMEXWriter
from llm_model_explorer.stream_metadata import METADATA, StreamError, StreamProgress

FIXTURES: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "api/fixtures/conformance.json").read_text()
)
EMBEDDINGS: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "api/fixtures/embeddings.json").read_text()
)

ANALYSIS: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "api/fixtures/embedding-analysis.json").read_text()
)


def metadata(length: int = 8) -> dict[str, object]:
    return dict(
        kind="tensor",
        tensor_id="t",
        name="層.😀",
        shape=[length // 4],
        dtype="float32",
        byte_order="little",
        layout="c",
        byte_length=length,
    )


@pytest.mark.parametrize(
    "case",
    [
        c
        for c in FIXTURES["wire_cases"] + EMBEDDINGS["wire_cases"] + ANALYSIS["wire_cases"]
        if c["expected"]["outcome"] != "reject"
    ],
    ids=lambda c: c["name"],
)
def test_shared_wire_bytes(case: dict[str, Any]) -> None:
    golden = bytes.fromhex(case["wire_hex"])
    writer = LMEXWriter()
    encoded = bytearray()
    offset = 0
    methods = {1: writer.metadata, 3: writer.progress, 5: writer.error}
    while offset < len(golden):
        _, kind, _, _, length = struct.unpack_from("<4sBBHI", golden, offset)
        payload = golden[offset + 12 : offset + 12 + length]
        offset += 12 + length
        if kind in methods:
            frame = methods[kind](json.loads(payload))
        elif kind == 2:
            frame = writer.data(payload)
        elif kind == 4:
            frame = writer.complete()
        else:
            frame = writer.cancelled()
        encoded.extend(frame[0])
        encoded.extend(frame[1])
    assert encoded == golden
    assert writer.terminal


@pytest.mark.parametrize(
    "case",
    [
        c
        for c in FIXTURES["schema_cases"] + EMBEDDINGS["schema_cases"] + ANALYSIS["schema_cases"]
        if c["schema"] in ("StreamMetadata", "InputEmbeddingsMetadata", "StreamProgress", "Error")
    ],
    ids=lambda c: c["name"],
)
def test_shared_control_schemas(case: dict[str, Any]) -> None:
    validators: dict[str, Callable[[Any], object]] = {
        "StreamMetadata": METADATA.validate_python,
        "InputEmbeddingsMetadata": METADATA.validate_python,
        "StreamProgress": StreamProgress.model_validate,
        "Error": StreamError.model_validate,
    }
    validate = validators[case["schema"]]
    if case["valid"]:
        validate(case["value"])
    else:
        with pytest.raises(ValueError):
            validate(case["value"])


def test_non_ascii_and_minimal_copy() -> None:
    writer = LMEXWriter(4)
    assert "層.😀".encode() in writer.metadata(metadata())[1]
    payload = bytearray(struct.pack("<2I", 0x01020304, 0xFFFFFFFF))
    _, view = writer.data(memoryview(payload)[:4])
    assert isinstance(view, memoryview) and view.obj is payload
    writer.data(memoryview(payload)[4:])
    assert writer.complete()[0] == bytes.fromhex("4c4d45580400000000000000")


@pytest.mark.parametrize("action", ["data", "progress", "complete"])
def test_metadata_required(action: str) -> None:
    writer = LMEXWriter()
    with pytest.raises(ValueError):
        if action == "data":
            writer.data(b"abcd")
        elif action == "progress":
            writer.progress({"completed": 0, "unit": "bytes"})
        else:
            writer.complete()


@pytest.mark.parametrize("terminal", ["complete", "error", "cancelled"])
def test_no_bytes_after_terminal(terminal: str) -> None:
    writer = LMEXWriter()
    writer.metadata(metadata(0))
    if terminal == "error":
        writer.error({"code": "internal_error", "message": "Failed"})
    else:
        getattr(writer, terminal)()
    emitters: list[Callable[[], object]] = [
        lambda: writer.metadata(metadata()),
        lambda: writer.data(b""),
        lambda: writer.progress({"completed": 0, "unit": "bytes"}),
        writer.complete,
        writer.cancelled,
        lambda: writer.error({"code": "internal_error", "message": "Failed"}),
    ]
    for emit in emitters:
        with pytest.raises(ValueError):
            emit()


def test_lengths_alignment_duplicate_and_limits() -> None:
    writer = LMEXWriter(4)
    writer.metadata(metadata(4))
    for emit in [
        lambda: writer.metadata(metadata()),
        lambda: writer.data(b"a"),
        lambda: writer.data(b"abcdefgh"),
        writer.complete,
    ]:
        with pytest.raises(ValueError):
            emit()
    writer.data(b"abcd")
    with pytest.raises(ValueError):
        writer.data(b"efgh")
    for bound in [0, 3, 262148, True]:
        with pytest.raises(ValueError):
            LMEXWriter(bound)
    with pytest.raises(ValueError):
        LMEXWriter().metadata(dict(metadata(), name="x" * MAX_CONTROL_BYTES))
    for value in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError):
            LMEXWriter().error(
                {"code": "internal_error", "message": "Failed", "details": {"nested": [value]}}
            )
    stats = next(c["value"] for c in FIXTURES["schema_cases"] if c["name"] == "statistics-empty")
    writer = LMEXWriter()
    writer.metadata(stats)
    with pytest.raises(ValueError):
        writer.data(b"")
    writer.complete()
