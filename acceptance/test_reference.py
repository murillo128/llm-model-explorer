"""Capability-gated reference smoke. No implicit downloads or repository weights."""

import json
import os
import struct
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from acceptance.fixtures import TEXT
from acceptance.reference import samples
from acceptance.test_network import Frames, Service


def test_local_smollm2_base(tmp_path):
    supplied = os.environ.get("LMEX_REFERENCE_MODEL_DIR")
    if not supplied:
        pytest.skip("LMEX_REFERENCE_MODEL_DIR not supplied; local SmolLM2-135M Base not tested")
    directory = Path(supplied).resolve(strict=True)
    expected = samples(directory)
    service = Service(
        tmp_path, device=os.environ.get("LMEX_REFERENCE_DEVICE", "cpu"), model_root=directory.parent
    )
    try:
        models = service.client.get("/models").json()["models"]
        config = json.loads((directory / "config.json").read_text())
        identity = config.get("_name_or_path", config.get("name_or_path", directory.name))
        model = next(m for m in models if m["id"].split("@")[0] in (identity, directory.name))
        session = service.client.post("/sessions", json={"model_id": model["id"]}).json()["id"]
        inventory = service.client.get(f"/sessions/{session}/tensors").json()["tensors"]
        assert sorted((t["name"], t["shape"]) for t in inventory) == sorted(
            (t["name"], t["shape"]) for t in expected["inventory"]
        )
        for tensor in expected["selected"]:
            descriptor = next(t for t in inventory if t["name"] == tensor["name"])
            offsets = {
                4 * (s["row"] * tensor["shape"][-1] + s["column"]): s["value"]
                for s in tensor["samples"]
            }
            with service.client.stream(
                "GET", f"/sessions/{session}/tensors/{descriptor['id']}/data"
            ) as response:
                frames = Frames(response)
                assert frames.next()[0] == 1
                received = 0
                while True:
                    kind, body = frames.next()
                    if kind == 4:
                        break
                    assert kind == 2
                    for offset, value in list(offsets.items()):
                        if received <= offset < received + len(body):
                            assert struct.unpack_from("<f", body, offset - received)[0] == value
                            del offsets[offset]
                    received += len(body)
                assert received == descriptor["numel"] * 4 and not offsets
                assert not frames.pending and list(frames.chunks) == []
        actual = service.client.post(f"/sessions/{session}/tokenize", json={"text": TEXT}).json()
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        assert [t["id"] for t in actual["tokens"]] == tokenizer(TEXT)["input_ids"]
        assert service.client.delete(f"/sessions/{session}").status_code == 204
    finally:
        service.stop()
        service.client.close()
