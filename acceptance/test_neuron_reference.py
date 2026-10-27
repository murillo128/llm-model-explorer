"""One full-checkpoint numerical bridge; tiny mathematical oracles stay in backend tests."""

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from acceptance.architecture_reference import header_storage, inventory, selections
from acceptance.test_network import Frames, Service


def test_full_smollm2_neuron_baseline_and_weight_integrity(tmp_path, record_property):
    selection = selections().get("smollm2")
    if selection is None:
        if os.environ.get("LMEX_REQUIRE_ARCHITECTURE_REFERENCES") == "1":
            pytest.fail("The SmolLM2 reference manifest is required")
        pytest.skip("Complete local SmolLM2 Base reference not supplied")
    directory = Path(selection["directory"]).resolve(strict=True)
    directory, model_id, identity = inventory("smollm2", selection)
    device = os.environ.get("LMEX_REFERENCE_DEVICE", "cpu")
    # Fixed public test prompt; no user content is persisted by this check.
    prompt = "The capital of France is"
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    ids = tokenizer(prompt)["input_ids"]
    shards = [str(path.relative_to(directory)) for path in directory.rglob("*.safetensors")]
    before_files = {}
    for name in shards:
        with (directory / name).open("rb") as stream:
            before_files[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    started = time.perf_counter()
    service = Service(
        tmp_path,
        model_root=directory.parent,
        device=device,
        startup_timeout=120,
        client_timeout=120,
        observe_neurons=True,
    )
    startup = time.perf_counter() - started
    try:
        models = service.client.get("/models").json()["models"]
        model = next(item for item in models if item["id"] == model_id)
        session = service.client.post("/sessions", json={"model_id": model["id"]}).json()
        base = f"/sessions/{session['id']}"
        capability = service.client.get(base + "/neuron-capabilities").json()
        assert capability["observation"]["status"] == "available"
        module = next(item for item in capability["modules"] if item["layer_index"] == 7)
        actual_tokens = service.client.post(base + "/tokenize", json={"text": prompt}).json()
        assert [token["id"] for token in actual_tokens["tokens"]] == ids
        request = {
            "model_revision": session["model_revision"],
            "module_id": module["module_id"],
            "neuron_indices": [427],
            "token_ids": ids,
        }

        def execute(endpoint, recipe):
            with service.client.stream("POST", base + endpoint, json=recipe) as response:
                frames = Frames(response)
                kind, body = frames.next()
                assert kind == 1
                metadata = json.loads(body)
                for key, value in request.items():
                    if key == "module_id":
                        assert metadata["module"]["module_id"] == value
                    else:
                        assert metadata[key] == value
                assert metadata["session_id"] == session["id"]
                assert metadata["model_id"] == model["id"]
                records = frames.rest()
                assert records[-1] == (4, b"")
                assert all(kind == 2 for kind, _ in records[:-1])
                data = b"".join(body for _, body in records[:-1])
                assert len(data) == metadata["byte_length"]
                values = np.frombuffer(data, dtype="<f4").copy()
                assert np.isfinite(values).all()
                return metadata, values

        assert service.client.get("/__test/neuron-state").json()["loaded"] is False
        started = time.perf_counter()
        metadata, values = execute("/neuron-activations", request)
        initial_execution = time.perf_counter() - started
        assert metadata["module"] == module
        assert values.shape == (len(ids),)
        service.idle()
        before = service.client.get("/__test/neuron-state").json()
        assert before["loaded"] and before["hooks"] == before["tasks"] == 0
        conditions = [
            {"id": "baseline", "mode": "baseline"},
            {"id": "zero", "mode": "zero", "scope": "all"},
            {"id": "scale", "mode": "scale", "scope": "all", "factor": 2},
        ]
        recipe = {
            **request,
            "conditions": conditions,
            "top_k": 5,
            "max_new_tokens": 2,
            "decoding": "greedy",
        }
        started = time.perf_counter()
        compared, probabilities = execute("/neuron-comparisons", recipe)
        comparison_seconds = time.perf_counter() - started
        assert compared["conditions"] == conditions
        assert compared["max_new_tokens"] == 2 and compared["decoding"] == "greedy"
        service.idle()
        assert service.client.get("/__test/neuron-state").json() == before

        # Independent loading and an uninstrumented ordinary HF forward: no
        # runtime loader, capture hook, comparison helper or product encoder.
        reference = (
            LlamaForCausalLM.from_pretrained(
                directory, local_files_only=True, torch_dtype=torch.float32
            )
            .to(device)
            .eval()
        )
        reference_digest = hashlib.sha256()
        for name, parameter in sorted(reference.named_parameters()):
            reference_digest.update(name.encode() + b"\0")
            reference_digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        assert reference_digest.hexdigest() == before["weights_sha256"]
        with torch.inference_mode():
            prefix = torch.tensor([ids], device=device)
            logits = reference(input_ids=prefix, use_cache=False).logits[0, -1].float()
            expected = torch.softmax(logits, dim=-1).cpu().numpy()
            generated = []
            for _ in range(2):
                token = int(logits.argmax())
                generated.append(token)
                if token in capability["limits"]["eos_token_ids"]:
                    break
                prefix = torch.cat((prefix, torch.tensor([[token]], device=device)), dim=1)
                logits = reference(input_ids=prefix, use_cache=False).logits[0, -1].float()
        union = compared["candidate_token_ids"]
        table = probabilities.reshape(len(conditions), len(union))
        # Float32 reduction roundoff between independently loaded forward paths.
        np.testing.assert_allclose(table[0], expected[union], rtol=1e-5, atol=1e-7)
        assert (
            compared["top_token_ids"][0]
            == sorted(range(len(expected)), key=lambda token: (-float(expected[token]), token))[:5]
        )
        assert compared["continuations"][0]["token_ids"] == generated
        assert compared["continuations"][0]["text"] == tokenizer.decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        del reference
        # After interventions the same resident instance gives a clean baseline.
        clean_meta, clean = execute(
            "/neuron-comparisons",
            {
                **recipe,
                "conditions": conditions[:1],
                "max_new_tokens": 0,
            },
        )
        np.testing.assert_allclose(
            clean, expected[clean_meta["candidate_token_ids"]], rtol=1e-5, atol=1e-7
        )
        service.idle()
        assert service.client.get("/__test/neuron-state").json() == before
        for name, digest in before_files.items():
            with (directory / name).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
        memory = {}
        status = Path(f"/proc/{service.process.pid}/status")
        if status.exists():
            memory = {
                line.split(":")[0]: line.split(":")[1].strip()
                for line in status.read_text().splitlines()
                if line.startswith(("VmRSS:", "VmHWM:"))
            }
        report = {
            "repository": selection["repository"],
            "revision": selection.get("revision"),
            "fingerprint": identity["fingerprint"],
            "execution": capability["execution"],
            "python": platform.python_version(),
            "platform": platform.system(),
            "storage_dtypes": sorted(
                {t["dtype"] for t in header_storage(directory, shards).values()}
            ),
            "startup_seconds": startup,
            "initial_execution_seconds": initial_execution,
            "comparison_seconds": comparison_seconds,
            "backend_memory": memory,
            "baseline_max_absolute_error": float(np.max(np.abs(table[0] - expected[union]))),
            "intervention_max_absolute_delta": float(np.max(np.abs(table[1:] - table[0]))),
            "resident_weights_sha256": before["weights_sha256"],
            "cuda_available": torch.cuda.is_available(),
        }
        record_property("neuron_reference", json.dumps(report))
        assert service.client.delete(base).status_code == 204
        service.idle()
    finally:
        service.stop()
        service.client.close()
