"""Admission and shared logical access for the reviewed BNB NF4 layout."""

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest
from quantized_oracles import PackedFixture, Stored, bnb_nf4_fixture
from test_models import mutate_last_byte

from llm_model_explorer import bnb_nf4
from llm_model_explorer.embeddings import resolve_embeddings
from llm_model_explorer.model_files import FileSnapshot, ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.tensor_source import PhysicalTensor


def model_id(fixture: PackedFixture) -> str:
    return f"numeric@{fixture.encoding}"


def test_bnb_nf4_records_complete_group_and_stable_logical_descriptor(tmp_path: Path) -> None:
    fixture = bnb_nf4_fixture()
    directory = fixture.write(tmp_path, split=True)
    source = ModelCatalogue(tmp_path).pin(model_id(fixture))

    (tensor,) = source.tensors()
    assert tensor.id == hashlib.sha256(fixture.name.encode()).hexdigest()
    assert tensor.name == fixture.name
    assert tensor.shape == fixture.shape
    assert tensor.rank == 2 and tensor.numel == fixture.shape[0] * fixture.shape[1]
    assert tensor.storage_dtype == "U8" and tensor.storage_format == "bnb-nf4-dq"
    assert source.inventory()["coverage"] == "complete"
    assert len(source.physical_tensors()) == len(fixture.storage) == 6
    assert len(list(directory.glob("*.safetensors"))) == 6


@pytest.mark.parametrize("defect", ["missing", "state", "shape", "codebook"])
def test_invalid_bnb_groups_are_not_admitted(tmp_path: Path, defect: str) -> None:
    fixture = bnb_nf4_fixture()
    storage = list(fixture.storage)
    if defect == "missing":
        storage = [item for item in storage if not item.name.endswith("nested_quant_map")]
    elif defect == "state":
        item = next(item for item in storage if item.name.endswith("bitsandbytes__nf4"))
        state = json.loads(item.data)
        state["nested_blocksize"] = 128
        data = json.dumps(state).encode()
        storage[storage.index(item)] = replace(item, shape=(len(data),), data=data)
    elif defect == "shape":
        item = next(item for item in storage if item.name.endswith("bitsandbytes__nf4"))
        state = json.loads(item.data)
        state["shape"][1] += 1
        data = json.dumps(state).encode()
        storage[storage.index(item)] = replace(item, shape=(len(data),), data=data)
    else:
        item = next(item for item in storage if item.name.endswith("quant_map"))
        data = struct.pack("<16f", 0.0, *struct.unpack("<15f", item.data[4:]))
        storage[storage.index(item)] = replace(item, data=data)
    broken = replace(fixture, storage=tuple(storage))
    broken.write(tmp_path)
    assert ModelCatalogue(tmp_path).discover() == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("quant_method", "other"),
        ("load_in_4bit", False),
        ("bnb_4bit_quant_type", "fp4"),
        ("bnb_4bit_use_double_quant", False),
        ("bnb_4bit_quant_storage", "uint4"),
        ("unreviewed_option", True),
    ],
)
def test_unreviewed_bnb_configuration_is_not_admitted(
    tmp_path: Path, field: str, value: object
) -> None:
    fixture = bnb_nf4_fixture()
    directory = fixture.write(tmp_path)
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"][field] = value
    path.write_text(json.dumps(config))
    assert ModelCatalogue(tmp_path).discover() == ()


def test_orphan_bnb_record_remains_visible_without_becoming_numeric(tmp_path: Path) -> None:
    fixture = bnb_nf4_fixture()
    orphan = Stored("model.layers.0.self_attn.q_proj.weight.extra", "U8", (1,), b"\x00")
    fixture = replace(fixture, storage=(*fixture.storage, orphan))
    fixture.write(tmp_path, split=True)
    source = ModelCatalogue(tmp_path).pin(model_id(fixture))

    assert [tensor.name for tensor in source.tensors()] == [fixture.name]
    inventory = source.inventory()
    assert inventory["coverage"] == "partial"
    assert "weight.extra" in str(inventory["diagnostics"])


def test_metadata_companion_mutation_fails_source_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = bnb_nf4_fixture()
    directory = fixture.write(tmp_path, split=True)
    original = bnb_nf4.read_physical

    def mutate_after_state_read(
        snapshot: FileSnapshot, tensor: PhysicalTensor, limit: int
    ) -> bytes:
        data = original(snapshot, tensor, limit)
        if tensor.name.endswith("bitsandbytes__nf4"):
            mutate_last_byte(directory / tensor.file)
        return data

    monkeypatch.setattr(bnb_nf4, "read_physical", mutate_after_state_read)
    with pytest.raises(ModelError) as raised:
        ModelCatalogue(tmp_path)._inspect_base(directory)
    assert raised.value.code == "model_content_changed"


def test_nf4_input_embedding_rows_use_shared_decoder(tmp_path: Path) -> None:
    fixture = bnb_nf4_fixture(prefix="model.embed_tokens")
    directory = fixture.write(tmp_path, split=True)
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config.update(vocab_size=fixture.shape[0], hidden_size=fixture.shape[1])
    config_path.write_text(json.dumps(config))
    source = ModelCatalogue(tmp_path).pin(model_id(fixture))
    requested = (2, 0, 2)

    result = resolve_embeddings(source, requested)
    reader = result.reader()
    blocks: list[bytes] = []
    while block := reader.read_available(4096):
        blocks.append(block)
    reader.close()
    expected = fixture.expected()
    width = fixture.shape[1] * 4
    assert b"".join(blocks) == b"".join(
        expected[row * width : (row + 1) * width] for row in requested
    )
