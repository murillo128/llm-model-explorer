"""Optional local SmolLM2 Base lookup against independent safetensors slices.

Run: python scripts/smoke_input_embeddings.py /local/models/SmolLM2-135M
Never downloads assets or loads the full model/table. Missing assets report SKIP.
"""

import hashlib
import json
import struct
import sys
import tempfile
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from safetensors import safe_open

from llm_model_explorer.app import create_app
from llm_model_explorer.settings import Settings


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: smoke_input_embeddings.py /local/models/SmolLM2-135M")
    directory = Path(sys.argv[1]).resolve()
    weights = directory / "model.safetensors"
    if not weights.is_file() or not (directory / "config.json").is_file():
        print("SKIP: supplied local reference model is unavailable.")
        return
    config = json.loads((directory / "config.json").read_text())
    assert config["model_type"] == "llama"
    assert config["vocab_size"] == 49152 and config["hidden_size"] == 576
    ids = [49151, 0, 128, 1, 128]
    with safe_open(weights, framework="pt", device="cpu") as checkpoint:
        table = checkpoint.get_slice("model.embed_tokens.weight")
        expected = torch.cat([table[token : token + 1] for token in ids]).float().numpy()
        expected_bytes = expected.astype("<f4").tobytes()
    with tempfile.TemporaryDirectory(prefix="embedding-smoke-") as cache:
        settings = Settings(model_root=directory.parent, cache_dir=Path(cache))
        with TestClient(create_app(settings)) as client:
            catalogue = client.get("/models").json()["models"]
            candidates = [
                model
                for model in catalogue
                if model["id"] in {directory.name, "HuggingFaceTB/SmolLM2-135M"}
            ]
            assert len(candidates) == 1
            session = client.post("/sessions", json={"model_id": candidates[0]["id"]})
            assert session.status_code == 201
            response = client.post(
                f"/sessions/{session.json()['id']}/embeddings", json={"token_ids": ids}
            )
            assert response.status_code == 200
            offset, actual, kinds = 0, bytearray(), []
            while offset < len(response.content):
                magic, kind, flags, reserved, size = struct.unpack_from(
                    "<4sBBHI", response.content, offset
                )
                assert (magic, flags, reserved) == (b"LMEX", 0, 0)
                data = response.content[offset + 12 : offset + 12 + size]
                kinds.append(kind)
                if kind == 1:
                    meta = json.loads(data)
                    assert meta["token_ids"] == ids and meta["shape"] == [5, 576]
                elif kind == 2:
                    actual.extend(data)
                offset += 12 + size
            assert kinds[0] == 1 and kinds[-1] == 4 and 5 not in kinds
            assert actual == expected_bytes
        assert not list(Path(cache).iterdir())
    print(
        json.dumps(
            {
                "outcome": "PASS",
                "reference": "HuggingFaceTB/SmolLM2-135M Base",
                "token_ids": ids,
                "shape": [5, 576],
                "bytes": len(actual),
                "sha256": hashlib.sha256(actual).hexdigest(),
                "cache_entries": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
