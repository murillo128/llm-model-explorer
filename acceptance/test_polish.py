"""Integrated packed numerics and architecture-aware embeddings over real TCP."""

import hashlib
import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open

from acceptance.polish_fixtures import generate, packed_table
from acceptance.test_network import Frames, Service, result


def embedding_result(service, prefix, ids):
    with service.client.stream("POST", prefix + "/embeddings", json={"token_ids": ids}) as response:
        frames = Frames(response).rest()
    assert frames[0][0] == 1 and frames[-1] == (4, b"")
    metadata = json.loads(frames[0][1])
    payload = b"".join(data for kind, data in frames if kind == 2)
    assert metadata["token_ids"] == ids
    assert metadata["byte_length"] == len(payload)
    assert "tensor_id" not in metadata and "name" not in metadata
    return metadata, payload


def assert_analysis(service, path, expected):
    """NumPy float64 statistics and scalar bin assignment, independent of PyTorch."""
    rows, columns = expected.shape
    finite = expected[np.isfinite(expected)].astype(np.float64)
    stats, payload = result(service, path + "/statistics")
    assert payload == b""
    assert (stats["count"], stats["finite_count"], stats["non_finite_count"]) == (
        rows * columns,
        finite.size,
        rows * columns - finite.size,
    )
    for key, value in {
        "minimum": finite.min(),
        "maximum": finite.max(),
        "mean": finite.mean(),
        "stddev": finite.std(),
    }.items():
        assert stats[key] == pytest.approx(value, rel=1e-6, abs=1e-7)
    for key, q in zip(
        ("p01", "p05", "p50", "p95", "p99"), (0.01, 0.05, 0.5, 0.95, 0.99), strict=True
    ):
        assert stats["percentiles"][key] == pytest.approx(
            np.quantile(finite, q), rel=1e-6, abs=1e-7
        )
    metadata, payload = result(service, path + "/distributions")
    low, high = float(finite.min()), float(finite.max())
    assert (metadata["domain_minimum"], metadata["domain_maximum"]) == (low, high)
    row_counts = np.zeros((rows, 100), dtype="<u4")
    column_counts = np.zeros((100, columns), dtype="<u4")
    for (row, column), scalar in np.ndenumerate(expected):
        if not math.isfinite(scalar):
            continue
        bin_index = (
            50 if low == high else min(99, math.floor((float(scalar) - low) / (high - low) * 100))
        )
        row_counts[row, bin_index] += 1
        column_counts[bin_index, column] += 1
    assert payload == row_counts.tobytes() + column_counts.tobytes()


@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
@pytest.mark.parametrize("unresolved", [False, True], ids=["complete", "partial"])
def test_packed_inventory_data_analysis_and_embedding_rows_over_tcp(tmp_path, family, unresolved):
    fixture = packed_table(tmp_path / "models", family, unresolved=unresolved)
    expected = fixture.expected()  # Scalar format oracle, including negative-zero bits.
    service = Service(tmp_path, model_root=tmp_path / "models")
    try:
        response = service.client.post("/sessions", json={"model_id": "numeric"})
        assert response.status_code == 201, response.text
        prefix = f"/sessions/{response.json()['id']}"
        inventory = service.client.get(prefix + "/tensors").json()
        assert inventory["coverage"] == ("partial" if unresolved else "complete")
        assert bool(inventory["diagnostics"]) == unresolved
        if unresolved:
            assert "unknown.packed" in str(inventory["diagnostics"])
        (descriptor,) = inventory["tensors"]
        assert descriptor["name"] == fixture.name
        assert descriptor["shape"] == list(fixture.shape)
        assert descriptor["logical_dtype"] == "float32"
        assert descriptor["storage_format"] == fixture.encoding
        assert descriptor["id"] == hashlib.sha256(fixture.name.encode()).hexdigest()
        path = f"{prefix}/tensors/{descriptor['id']}"
        for _ in range(2):  # Both cold production and complete cached artifacts.
            metadata, payload = result(service, path + "/data")
            assert metadata["shape"] == list(fixture.shape)
            assert payload == expected
            assert_analysis(
                service, path, np.frombuffer(expected, dtype="<f4").reshape(fixture.shape)
            )
        rows, columns = fixture.shape
        ids = [rows - 1, 0, 1, rows - 1, 1]
        metadata, payload = embedding_result(service, prefix, ids)
        assert metadata["shape"] == [len(ids), columns]
        assert payload == b"".join(expected[i * columns * 4 : (i + 1) * columns * 4] for i in ids)
        assert embedding_result(service, prefix, [])[0]["shape"] == [0, columns]
        if unresolved:
            unknown = hashlib.sha256(b"unknown.packed").hexdigest()
            for kind in ("data", "statistics", "distributions"):
                response = service.client.get(f"{prefix}/tensors/{unknown}/{kind}")
                assert response.status_code == 404 and response.json()["code"] == "tensor_not_found"
        assert str(tmp_path) not in json.dumps(inventory)
    finally:
        service.stop()
        service.client.close()


def test_text_families_and_non_text_capability_over_tcp(tmp_path):
    generate(tmp_path / "models")
    service = Service(tmp_path, model_root=tmp_path / "models")
    try:
        for family, rows, columns in (
            ("smollm2", 16, 12),
            ("qwen3-native", 16, 128),
            ("qwen35-native", 64, 32),
            ("qwen3", 16, 128),
            ("qwen35", 64, 32),
        ):
            session = service.client.post("/sessions", json={"model_id": family})
            assert session.status_code == 201, session.text
            prefix = f"/sessions/{session.json()['id']}"
            tokenized = service.client.post(
                prefix + "/tokenize", json={"text": "one two one", "add_special_tokens": True}
            )
            assert tokenized.status_code == 200, tokenized.text
            assert [token["id"] for token in tokenized.json()["tokens"]] == [1, 2, 3, 2]
            for ids in ([1, 2, 3, 2], [rows - 1, 0, rows - 1, 1], []):
                metadata, payload = embedding_result(service, prefix, ids)
                assert metadata["shape"] == [len(ids), columns]
                # Arithmetic oracle is independent of fixture writer and source reads.
                expected = b"".join(
                    struct.pack("<f", ((token * columns + column) % 29 - 14) / 8)
                    for token in ids
                    for column in range(columns)
                )
                assert payload == expected
            assert service.client.delete(prefix).status_code == 204
        session = service.client.post("/sessions", json={"model_id": "vjepa2"})
        assert session.status_code == 201
        prefix = f"/sessions/{session.json()['id']}"
        for ids in ([], [0]):
            response = service.client.post(prefix + "/embeddings", json={"token_ids": ids})
            assert (
                response.status_code == 422
                and response.json()["code"] == "unsupported_representation"
            )
        graph = service.client.get(prefix + "/architecture").json()
        assert graph["status"] == "available"
        assert graph["graph"]["scope"] == "visual_encoder_predictor"
        inventory = service.client.get(prefix + "/tensors").json()
        tensor = next(t for t in inventory["tensors"] if t["rank"] == 2)
        assert (
            result(service, f"{prefix}/tensors/{tensor['id']}/data")[0]["shape"] == tensor["shape"]
        )
    finally:
        service.stop()
        service.client.close()


@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
def test_local_quantized_checkpoint_input_rows(tmp_path, family):
    from acceptance.architecture_reference import inventory, selections

    entries = selections()
    if family not in entries:
        if os.environ.get("LMEX_REQUIRE_ARCHITECTURE_REFERENCES") == "1":
            pytest.fail(f"Required complete local {family} checkpoint is missing")
        pytest.skip(f"No configured local {family} checkpoint; fixture success is separate")
    directory, model_id, evidence = inventory(family, entries[family])
    name = (
        "model.embed_tokens.weight"
        if family == "qwen3"
        else "model.language_model.embed_tokens.weight"
    )
    expected = None
    for shard in directory.rglob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            if name not in weights.keys():
                continue
            view = weights.get_slice(name)
            vocabulary, hidden = view.get_shape()
            ids = [vocabulary - 1, 0, 1, vocabulary - 1, 1]
            expected = b"".join(view[i : i + 1].float().numpy().tobytes() for i in ids)
            break
    assert expected is not None
    service = Service(tmp_path, model_root=directory.parent, startup_timeout=180)
    try:
        response = service.client.post("/sessions", json={"model_id": model_id})
        assert response.status_code == 201, response.text
        prefix = f"/sessions/{response.json()['id']}"
        metadata, payload = embedding_result(service, prefix, ids)
        assert metadata["shape"] == [len(ids), hidden]
        assert payload == expected
        assert embedding_result(service, prefix, [])[0]["shape"] == [0, hidden]
        evidence.update(shape=[len(ids), hidden], token_ids=ids, exact_bytes=len(expected))
        if output := os.environ.get("LMEX_EVIDENCE_DIR"):
            Path(output).mkdir(parents=True, exist_ok=True)
            (Path(output) / f"input-rows-{family}.json").write_text(json.dumps(evidence, indent=2))
    finally:
        service.stop()
        service.client.close()
