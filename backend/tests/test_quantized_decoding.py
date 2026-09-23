"""Decoder checkpoint: independent numeric references before logical integration."""

import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO, cast
from unittest.mock import Mock

import pytest
import torch
from quantized_oracles import (
    PackedFixture,
    compressed_tensors_fixture,
    e2m1,
    e4m3,
    gptq_fixture,
    nvfp4_fixture,
)

from llm_model_explorer import quantized_decoding
from llm_model_explorer.model_files import FileSnapshot, ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.tensor_source import DEFAULT_CHUNK_ELEMENTS, ModelSource


@pytest.fixture(params=["gptq-int4", "nvfp4", "compressed-tensors-w4a16-int4"])
def packed(request: pytest.FixtureRequest) -> PackedFixture:
    if request.param == "gptq-int4":
        return gptq_fixture()
    if request.param == "nvfp4":
        return nvfp4_fixture()
    return compressed_tensors_fixture()


def pin(root: Path, fixture: PackedFixture, *, split: bool = False) -> ModelSource:
    directory = fixture.write(root, split=split)
    return ModelCatalogue(root).pin(f"{directory.name}@{fixture.encoding}")


def decode(source: ModelSource, packed: PackedFixture, start: int, count: int) -> torch.Tensor:
    return quantized_decoding.decode_range(
        source._snapshot, source.physical_tensors(), packed.encoding, start, count
    )


def raw(values: torch.Tensor) -> bytes:
    assert values.dtype == torch.float32
    assert values.device.type == "cpu"
    assert values.ndim == 1
    assert values.is_contiguous()
    return bytes(values.view(torch.uint8).tolist())


@pytest.mark.parametrize("split", [False, True])
def test_complete_matrix_matches_independent_scalar_bytes(
    tmp_path: Path, packed: PackedFixture, split: bool
) -> None:
    source = pin(tmp_path, packed, split=split)
    expected = packed.expected()
    actual = decode(source, packed, 0, len(expected) // 4)
    assert raw(actual) == expected
    assert actual.untyped_storage().nbytes() == len(expected)


def test_unaligned_ranges_and_chunks_preserve_logical_order(
    tmp_path: Path, packed: PackedFixture
) -> None:
    source = pin(tmp_path, packed, split=True)
    expected = packed.expected()
    total = len(expected) // 4
    # Every nibble offset, group/row crossings, the final scalar and empty end.
    ranges = [(offset, 19) for offset in range(16)]
    ranges += [(packed.shape[1] - 3, 11), (127, 131), (total - 1, 1), (total, 0)]
    if packed.encoding == "compressed-tensors-w4a16-int4":
        ranges += [(31, 3), (63, 5), (packed.shape[1] - 1, 2)]
    for start, count in ranges:
        if start + count <= total:
            assert (
                raw(decode(source, packed, start, count))
                == expected[start * 4 : (start + count) * 4]
            )
    chunks = [
        decode(source, packed, start, min(131, total - start)) for start in range(0, total, 131)
    ]
    assert raw(torch.cat(chunks)) == expected


def test_nvfp4_all_nibbles_and_signed_zero_are_explicit() -> None:
    expected = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    assert b"".join(struct.pack("<f", e2m1(code)) for code in range(16)) == struct.pack(
        "<16f", *expected
    )
    packed = nvfp4_fixture()
    weights = packed.storage[0].data
    assert {value & 15 for value in weights} == set(range(16))
    assert {value >> 4 for value in weights} == set(range(16))
    assert e4m3(1) == 2**-9
    assert e4m3(7) == 7 * 2**-9
    assert e4m3(8) == 2**-6
    assert e4m3(0x7E) == 448


def test_gptq_fixture_exercises_stored_group_mapping_and_zero_endpoints() -> None:
    fixture = gptq_fixture()
    data = {entry.name.rsplit(".", 1)[1]: entry.data for entry in fixture.storage}
    indices = struct.unpack("<256i", data["g_idx"])
    assert set(indices) == {0, 1}
    assert any(group != column // 128 for column, group in enumerate(indices))
    assert any(indices[index] == indices[index + 1] for index in range(len(indices) - 1))
    zeros = struct.unpack("<4I", data["qzeros"])
    assert {word >> (4 * nibble) & 15 for word in zeros for nibble in range(8)} == set(range(16))
    weights = struct.unpack("<512I", data["qweight"])
    for nibble in range(8):
        assert {word >> (4 * nibble) & 15 for word in weights} == set(range(16))


def test_activation_scale_does_not_change_nvfp4_weight(tmp_path: Path) -> None:
    first, second = nvfp4_fixture(input_scale=31.0), nvfp4_fixture(input_scale=0.000031)
    first_source = pin(tmp_path / "a", first)
    second_source = pin(tmp_path / "b", second)
    assert raw(decode(first_source, first, 0, 256)) == raw(decode(second_source, second, 0, 256))


@pytest.mark.parametrize("bad_group", [-1, 2, 2**31 - 1])
def test_invalid_gptq_group_is_rejected_before_any_values(tmp_path: Path, bad_group: int) -> None:
    fixture = gptq_fixture()
    entry = fixture.storage[-1]
    modified = bytearray(entry.data)
    struct.pack_into("<i", modified, 5 * 4, bad_group)
    fixture = replace(
        fixture, storage=fixture.storage[:-1] + (replace(entry, data=bytes(modified)),)
    )
    source = pin(tmp_path, fixture)
    with pytest.raises(ModelError) as raised:
        decode(source, fixture, 3, 8)
    assert raised.value.code in {"validation_error", "unsupported_representation"}
    assert str(tmp_path) not in str(raised.value)


@pytest.mark.parametrize("start,count", [(-1, 1), (0, -1), (2**53, 1), (0, 2**53), (2**53 - 1, 1)])
def test_decoder_rejects_unsafe_ranges(
    tmp_path: Path, packed: PackedFixture, start: int, count: int
) -> None:
    source = pin(tmp_path, packed)
    with pytest.raises((ModelError, ValueError)):
        decode(source, packed, start, count)


def test_decoder_rejects_outside_shape_and_oversized_chunk(
    tmp_path: Path, packed: PackedFixture
) -> None:
    source = pin(tmp_path, packed)
    total = packed.shape[0] * packed.shape[1]
    with pytest.raises(ModelError):
        decode(source, packed, total - 1, 2)
    with pytest.raises(ValueError):
        decode(source, packed, 0, DEFAULT_CHUNK_ELEMENTS + 1)


def test_small_range_copies_only_bounded_storage_and_owns_output(
    tmp_path: Path, packed: PackedFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = pin(tmp_path, packed, split=True)
    files = {path: path.read_bytes() for path in source._snapshot.directory.glob("*.safetensors")}
    reads: list[int] = []
    selected_counts: list[int] = []
    original_open, original_select = FileSnapshot.open, torch.index_select

    @contextmanager
    def observed_open(snapshot: FileSnapshot, name: str) -> Iterator[BinaryIO]:
        with original_open(snapshot, name) as stream:
            proxy = Mock(wraps=stream)

            def read(size: int) -> bytes:
                assert 0 < size <= 5 * 4
                reads.append(size)
                return stream.read(size)

            proxy.read.side_effect = read
            yield cast(BinaryIO, proxy)

    def observed_select(
        values: torch.Tensor, dimension: int, indices: torch.Tensor
    ) -> torch.Tensor:
        selected_counts.append(indices.numel())
        # Physical reads may gather four bytes per requested logical scalar.
        assert indices.numel() <= 5 * 4
        return original_select(values, dimension, indices)

    monkeypatch.setattr(FileSnapshot, "open", observed_open)
    monkeypatch.setattr(torch, "index_select", observed_select)
    values = decode(source, packed, 3, 5)
    assert raw(values) == packed.expected()[12:32]
    assert selected_counts
    assert reads and sum(reads) <= 4 * 5 * 4
    assert values.untyped_storage().nbytes() == 5 * 4
    values.fill_(987.0)
    assert {path: path.read_bytes() for path in files} == files
    assert raw(decode(source, packed, 3, 5)) == packed.expected()[12:32]


@pytest.mark.parametrize("stage", ["before_read", "after_read", "before_gather"])
def test_concurrent_truncation_invalidates_owned_read(
    tmp_path: Path, packed: PackedFixture, stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = pin(tmp_path, packed, split=True)
    original_open, original_select = FileSnapshot.open, torch.index_select
    mutated = False

    def truncate(path: Path) -> None:
        nonlocal mutated
        if not mutated:
            with path.open("r+b") as writer:
                writer.truncate(0)
            mutated = True

    @contextmanager
    def racing_open(snapshot: FileSnapshot, name: str) -> Iterator[BinaryIO]:
        with original_open(snapshot, name) as stream:
            proxy = Mock(wraps=stream)

            def read(size: int) -> bytes:
                if stage == "before_read":
                    truncate(snapshot.directory / name)
                data = stream.read(size)
                if stage == "after_read":
                    truncate(snapshot.directory / name)
                return data

            proxy.read.side_effect = read
            yield cast(BinaryIO, proxy)

    def racing_select(values: torch.Tensor, dimension: int, indices: torch.Tensor) -> torch.Tensor:
        if stage == "before_gather":
            # Gather operates on owned bytes, so truncation cannot cause SIGBUS.
            for item in source._physical:
                selected_suffix = (
                    ".g_idx"
                    if packed.encoding == "gptq-int4"
                    else ".weight_packed"
                    if packed.encoding == "compressed-tensors-w4a16-int4"
                    else ".weight"
                )
                if item.name.endswith(selected_suffix):
                    truncate(source._snapshot.directory / item.file)
                    break
        return original_select(values, dimension, indices)

    monkeypatch.setattr(FileSnapshot, "open", racing_open)
    monkeypatch.setattr(torch, "index_select", racing_select)
    with pytest.raises(ModelError) as raised:
        decode(source, packed, 3, 5)
    assert mutated and raised.value.code == "model_content_changed"


def test_decoder_detects_source_mutation(tmp_path: Path, packed: PackedFixture) -> None:
    source = pin(tmp_path, packed, split=True)
    location = source.physical_tensors()[0]
    path = source._snapshot.directory / location.file
    with path.open("r+b") as stream:
        stream.seek(location.offset)
        original = stream.read(1)
        stream.seek(location.offset)
        stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(ModelError) as raised:
        quantized_decoding.decode_range(source._snapshot, source._physical, packed.encoding, 0, 1)
    assert raised.value.code == "model_content_changed"
