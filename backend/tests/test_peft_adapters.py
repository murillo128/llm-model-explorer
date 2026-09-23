"""Small local PEFT fixtures for discovery, composition, values and integrity."""

import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_models import mutate_last_byte, write_weights
from test_quantized_models import config_for, packed_group, write_storage
from test_tensor_data import frames

from llm_model_explorer import tokenization
from llm_model_explorer.app import create_app
from llm_model_explorer.model_files import FileSnapshot, ModelError
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings
from llm_model_explorer.tokenization import TokenizerService

MODULE = "model.layers.0.self_attn.q_proj"
V_MODULE = MODULE.removesuffix("q_proj") + "v_proj"
UPSTREAM = "org/base"
ADAPTER = "org/adapter"


def make_bases(root: Path, *, adapter_tokens: bool = False) -> tuple[Path, Path]:
    native = root / "base-native"
    native.mkdir()
    native_config = copy.deepcopy(config_for("JunHowie"))
    native_config.pop("quantization_config")
    native_config["_name_or_path"] = UPSTREAM
    native_config["num_hidden_layers"] = 1
    native_config["hidden_size"] = 128
    native_config["vocab_size"] = 16
    (native / "config.json").write_text(json.dumps(native_config))
    write_weights(
        native / "model.safetensors",
        [
            (name + ".weight", "F32", [8, 128], [float(i) / 32 for i in range(8 * 128)])
            for name in (MODULE, V_MODULE)
        ]
        + [
            (
                "model.embed_tokens.weight",
                "F32",
                [16, 128],
                [float(i) / 128 for i in range(16 * 128)],
            )
        ],
    )
    (native / "tokenizer.json").write_text('{"model":"base"}')

    quantized = root / "base-gptq"
    quantized.mkdir()
    quantized_config = copy.deepcopy(config_for("JunHowie"))
    quantized_config["_name_or_path"] = UPSTREAM
    quantized_config["num_hidden_layers"] = 1
    quantized_config["hidden_size"] = 128
    quantized_config["vocab_size"] = 16
    (quantized / "config.json").write_text(json.dumps(quantized_config))
    q_group = packed_group("JunHowie")
    v_group = [(name.replace(MODULE, V_MODULE), dtype, shape) for name, dtype, shape in q_group]
    write_storage(quantized / "model.safetensors", [*q_group, *v_group])
    if adapter_tokens:
        adapter = root / "adapter"
        adapter.mkdir(exist_ok=True)
        (adapter / "tokenizer.json").write_text('{"model":"adapter"}')
    return native, quantized


def make_adapter(
    root: Path,
    *,
    config_updates: dict[str, object] | None = None,
    factors: list[tuple[str, str, list[int], list[float]]] | None = None,
    indexed: bool = False,
) -> Path:
    directory = root / "adapter"
    directory.mkdir(exist_ok=True)
    config: dict[str, object] = {
        "_name_or_path": ADAPTER,
        "base_model_name_or_path": UPSTREAM,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "fan_in_fan_out": False,
        "use_dora": False,
        "use_rslora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
    }
    config.update(config_updates or {})
    (directory / "adapter_config.json").write_text(json.dumps(config))
    if factors is None:
        factors = [
            (
                "base_model.model." + MODULE + ".lora_A.default.weight",
                "F16",
                [2, 128],
                [float(i - 64) / 16 for i in range(256)],
            ),
            (
                "base_model.model." + MODULE + ".lora_B.default.weight",
                "BF16",
                [8, 2],
                [float(i - 8) / 8 for i in range(16)],
            ),
            (
                "base_model.model." + V_MODULE + ".lora_A.default.weight",
                "F16",
                [2, 128],
                [float(i - 32) / 16 for i in range(256)],
            ),
            (
                "base_model.model." + V_MODULE + ".lora_B.default.weight",
                "BF16",
                [8, 2],
                [float(i - 4) / 8 for i in range(16)],
            ),
        ]
    if indexed:
        payload = directory / "weights/adapter-part.safetensors"
        write_weights(payload, factors)
        mapping = {name: "weights/adapter-part.safetensors" for name, *_ in factors}
        (directory / "adapter_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )
    else:
        write_weights(directory / "adapter_model.safetensors", factors)
    return directory


def composite_id(base_variant: str = "native") -> str:
    base = UPSTREAM if base_variant == "native" else f"{UPSTREAM}@gptq-int4"
    return f"{base}+peft-lora:{ADAPTER}"


def as_float32(dtype: str, values: list[float]) -> bytes:
    result = bytearray()
    for value in values:
        if dtype == "F32":
            converted = value
        elif dtype == "F16":
            converted = struct.unpack("<e", struct.pack("<e", value))[0]
        else:
            bits = struct.pack("<f", value)[2:]
            converted = struct.unpack("<f", b"\0\0" + bits)[0]
        result.extend(struct.pack("<f", converted))
    return bytes(result)


def tensor_payload(client: TestClient, session_id: str, tensor_id: str) -> bytes:
    response = client.get(f"/sessions/{session_id}/tensors/{tensor_id}/data")
    assert response.status_code == 200, response.text
    return b"".join(block for kind, block in frames(response.content) if kind == 2)


def test_models_and_sessions_expose_native_and_quantized_compositions(settings: Settings) -> None:
    make_bases(settings.model_root)
    make_adapter(settings.model_root, indexed=True)
    catalogue = ModelCatalogue(settings.model_root)
    models = {model.id: model for model in catalogue.list_models()}
    assert set(models) == {
        UPSTREAM,
        f"{UPSTREAM}@gptq-int4",
        composite_id("native"),
        composite_id("gptq"),
    }
    for model_id in (composite_id("native"), composite_id("gptq")):
        model = models[model_id]
        assert model.model_type == "qwen3"
        assert model.architectures == ("Qwen3ForCausalLM",)
        assert "LoRA" in model.display_name and "org/adapter" in model.display_name
        assert str(settings.model_root) not in model.model_dump_json()

    with TestClient(create_app(settings)) as client:
        public_models = client.get("/models").json()["models"]
        assert {model["id"] for model in public_models} == set(models)
        for model_id in (composite_id("native"), composite_id("gptq")):
            base_id = UPSTREAM if model_id == composite_id("native") else f"{UPSTREAM}@gptq-int4"
            expected_base = catalogue.pin(base_id).tensors()
            response = client.post("/sessions", json={"model_id": model_id})
            assert response.status_code == 201, response.text
            session_id = response.json()["id"]
            assert response.json()["model_id"] == model_id
            inventory = client.get(f"/sessions/{session_id}/tensors").json()
            tensors = inventory["tensors"]
            assert tensors[: len(expected_base)] == [
                tensor.model_dump(mode="json") for tensor in expected_base
            ]
            assert inventory["coverage"] == "complete" and inventory["diagnostics"] == [], (
                model_id,
                inventory["diagnostics"],
            )
            adapter_tensors = tensors[len(expected_base) :]
            assert len(adapter_tensors) == 4
            assert {tuple(tensor["shape"]) for tensor in adapter_tensors} == {
                (2, 128),
                (8, 2),
            }
            for tensor in adapter_tensors:
                assert tensor["id"] == hashlib.sha256(tensor["name"].encode()).hexdigest()
            a_tensor = next(t for t in adapter_tensors if ".q_proj.lora_A." in t["name"])
            b_tensor = next(t for t in adapter_tensors if ".q_proj.lora_B." in t["name"])
            assert tensor_payload(client, session_id, a_tensor["id"]) == as_float32(
                "F16", [float(i - 64) / 16 for i in range(256)]
            )
            assert tensor_payload(client, session_id, b_tensor["id"]) == as_float32(
                "BF16", [float(i - 8) / 8 for i in range(16)]
            )


def test_current_smollm_adapter_inert_peft_metadata_is_accepted(settings: Settings) -> None:
    make_bases(settings.model_root)
    make_adapter(
        settings.model_root,
        config_updates={
            "inference_mode": True,
            "lora_dropout": 0.05,
            "eva_config": None,
            "lora_ga_config": None,
            "use_bdlora": None,
            "qalora_group_size": 16,
            "megatron_core": "megatron.core",
        },
    )
    models = {model.id for model in ModelCatalogue(settings.model_root).list_models()}
    assert composite_id("native") in models
    assert composite_id("gptq") in models


@pytest.mark.parametrize(
    "config_updates,factors",
    [
        ({"peft_type": "IA3"}, None),
        ({"task_type": "FEATURE_EXTRACTION"}, None),
        ({"use_dora": True}, None),
        ({"use_bdlora": True}, None),
        ({"eva_config": {"enabled": True}}, None),
        ({"lora_ga_config": {"enabled": True}}, None),
        ({"megatron_core": "custom.core"}, None),
        ({"qalora_group_size": 0}, None),
        ({"rank_pattern": {"q_proj": 4}}, None),
        ({"target_modules": ["q_proj", "self_attn.q_proj"]}, None),
        ({"new_inference_option": True}, None),
        (
            {},
            [
                (
                    "base_model.model." + MODULE + ".lora_A.weight",
                    "F32",
                    [2, 127],
                    [1.0] * (2 * 127),
                ),
                (
                    "base_model.model." + MODULE + ".lora_B.weight",
                    "F32",
                    [8, 2],
                    [1.0] * 16,
                ),
            ],
        ),
        (
            {},
            [
                (
                    "base_model.model." + MODULE + ".lora_A.weight",
                    "F32",
                    [2, 128],
                    [1.0] * 256,
                ),
            ],
        ),
    ],
)
def test_invalid_or_unsupported_adapters_are_not_candidates(
    settings: Settings,
    config_updates: dict[str, object],
    factors: list[tuple[str, str, list[int], list[float]]] | None,
) -> None:
    make_bases(settings.model_root)
    make_adapter(settings.model_root, config_updates=config_updates, factors=factors)
    assert {model.id for model in ModelCatalogue(settings.model_root).list_models()} == {
        UPSTREAM,
        f"{UPSTREAM}@gptq-int4",
    }


@pytest.mark.parametrize("fault", ["malformed-config", "ambiguous-payload", "invalid-index"])
def test_malformed_or_ambiguous_adapter_files_are_not_candidates(
    settings: Settings, fault: str
) -> None:
    make_bases(settings.model_root)
    adapter = make_adapter(settings.model_root, indexed=fault == "invalid-index")
    if fault == "malformed-config":
        (adapter / "adapter_config.json").write_text("{")
    elif fault == "ambiguous-payload":
        shutil.copyfile(adapter / "adapter_model.safetensors", adapter / "unexpected.safetensors")
    else:
        index = adapter / "adapter_model.safetensors.index.json"
        document = json.loads(index.read_text())
        first = next(iter(document["weight_map"]))
        document["weight_map"][first] = "../outside.safetensors"
        index.write_text(json.dumps(document))
    assert {model.id for model in ModelCatalogue(settings.model_root).list_models()} == {
        UPSTREAM,
        f"{UPSTREAM}@gptq-int4",
    }


def test_duplicate_factor_alias_is_rejected(settings: Settings) -> None:
    make_bases(settings.model_root)
    factors = [
        ("base_model.model." + MODULE + ".lora_A.weight", "F32", [2, 128], [1.0] * 256),
        (MODULE + ".lora_A.weight", "F32", [2, 128], [2.0] * 256),
        (MODULE + ".lora_B.weight", "F32", [8, 2], [1.0] * 16),
    ]
    make_adapter(settings.model_root, factors=factors)
    assert {model.id for model in ModelCatalogue(settings.model_root).list_models()} == {
        UPSTREAM,
        f"{UPSTREAM}@gptq-int4",
    }


def test_embedding_weights_are_not_supported_lora_targets(settings: Settings) -> None:
    make_bases(settings.model_root)
    factors = [
        (
            "base_model.model.model.embed_tokens.lora_A.weight",
            "F32",
            [2, 128],
            [1.0] * (2 * 128),
        ),
        (
            "base_model.model.model.embed_tokens.lora_B.weight",
            "F32",
            [16, 2],
            [1.0] * (16 * 2),
        ),
    ]
    make_adapter(
        settings.model_root,
        config_updates={"target_modules": ["embed_tokens"]},
        factors=factors,
    )
    assert {model.id for model in ModelCatalogue(settings.model_root).list_models()} == {
        UPSTREAM,
        f"{UPSTREAM}@gptq-int4",
    }


def test_regex_targets_and_directory_basename_matching(settings: Settings) -> None:
    native, _ = make_bases(settings.model_root)
    native.rename(settings.model_root / "base")
    config_path = settings.model_root / "base" / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("_name_or_path")
    config_path.write_text(json.dumps(config))
    make_adapter(
        settings.model_root,
        config_updates={"target_modules": r"model\.layers\..*\.self_attn\.(?:q|v)_proj"},
    )
    identities = {model.id for model in ModelCatalogue(settings.model_root).list_models()}
    assert "base+peft-lora:org/adapter" in identities


def test_composite_fingerprint_is_path_independent(settings: Settings) -> None:
    native, _ = make_bases(settings.model_root)
    adapter = make_adapter(settings.model_root)
    catalogue = ModelCatalogue(settings.model_root)
    before = catalogue.pin(composite_id()).fingerprint
    native.rename(settings.model_root / "relocated-base")
    adapter.rename(settings.model_root / "relocated-adapter")
    after = ModelCatalogue(settings.model_root).pin(composite_id()).fingerprint
    assert after == before
    assert after != ModelCatalogue(settings.model_root).pin(UPSTREAM).fingerprint


def test_composite_pin_rechecks_both_snapshots_after_hashing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    native, _ = make_bases(settings.model_root)
    make_adapter(settings.model_root)
    entry = next(
        entry
        for entry in ModelCatalogue(settings.model_root).discover()
        if entry.summary.id == composite_id()
    )
    original_fingerprint = FileSnapshot.fingerprint
    changed_after_hash = False

    def fingerprint(snapshot: FileSnapshot) -> str:
        nonlocal changed_after_hash
        result = original_fingerprint(snapshot)
        if snapshot.directory == native and not changed_after_hash:
            changed_after_hash = True
            mutate_last_byte(native / "model.safetensors")
        return result

    monkeypatch.setattr(FileSnapshot, "fingerprint", fingerprint)
    with pytest.raises(ModelError) as error:
        entry.pin()
    assert error.value.code == "model_content_changed"


@pytest.mark.parametrize("side", ["base", "adapter"])
@pytest.mark.parametrize("change", ["modify", "add", "remove"])
def test_composite_sessions_fail_on_either_snapshot_change(
    settings: Settings, side: str, change: str
) -> None:
    native, _ = make_bases(settings.model_root)
    adapter = make_adapter(settings.model_root)
    base = native if side == "base" else adapter
    asset = base / ("model.safetensors" if side == "base" else "adapter_model.safetensors")
    with TestClient(create_app(settings)) as client:
        session = client.post("/sessions", json={"model_id": composite_id()})
        assert session.status_code == 201
        session_id = session.json()["id"]
        if change == "modify":
            mutate_last_byte(asset)
        elif change == "add":
            (base / "extra.txt").write_text("new source asset")
        else:
            asset.unlink()
        response = client.get(f"/sessions/{session_id}")
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "model_content_changed"
        assert str(settings.model_root) not in response.text


def test_composite_tokenizer_uses_base_assets(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    native, _ = make_bases(settings.model_root, adapter_tokens=True)
    make_adapter(settings.model_root)
    source = ModelCatalogue(settings.model_root).pin(composite_id())
    loaded_from: list[Path] = []

    class FakeTokenizer:
        is_fast = False
        all_special_ids: list[int] = []

        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
            return {"input_ids": [7]}

        def convert_ids_to_tokens(self, token_id: int) -> str:
            return "base-token"

        def decode(self, token_ids: list[int], **kwargs: Any) -> str:
            return "base-token"

    def load(root: Path, directory: Path) -> FakeTokenizer:
        loaded_from.append(directory)
        return FakeTokenizer()

    monkeypatch.setattr(tokenization, "load_tokenizer", load)
    result = TokenizerService(settings.model_root).tokenize(source, "hello")
    assert loaded_from == [native]
    assert result["tokens"][0]["token"] == "base-token"
