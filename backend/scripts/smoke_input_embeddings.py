"""Optional local supported text-model lookup against independent safetensors slices.

Run: python scripts/smoke_input_embeddings.py /local/models/CHECKPOINT
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
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.settings import Settings


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: smoke_input_embeddings.py /local/models/CHECKPOINT")
    directory = Path(sys.argv[1]).resolve()
    if not (directory / "config.json").is_file():
        print("SKIP: supplied local reference model is unavailable.")
        return
    config = json.loads((directory / "config.json").read_text())
    # Independent oracle: never call the production input-table resolver.
    family = config["model_type"]
    assert family in {"llama", "qwen3", "qwen3_5"}
    key = (
        "model.language_model.embed_tokens.weight"
        if family == "qwen3_5"
        else "model.embed_tokens.weight"
    )
    dimensions = config["text_config"] if family == "qwen3_5" else config
    vocab, hidden = dimensions["vocab_size"], dimensions["hidden_size"]
    ids = [vocab - 1, 0, min(128, vocab - 1), min(1, vocab - 1), min(128, vocab - 1)]
    indexes = list(directory.glob("*.safetensors.index.json"))
    if indexes:
        assert len(indexes) == 1
        weights = directory / json.loads(indexes[0].read_text())["weight_map"][key]
    else:
        candidates = []
        for candidate in directory.glob("*.safetensors"):
            with safe_open(candidate, framework="pt", device="cpu") as checkpoint:
                if key in checkpoint.keys():
                    candidates.append(candidate)
        assert len(candidates) == 1
        weights = candidates[0]
    with safe_open(weights, framework="pt", device="cpu") as checkpoint:
        table = checkpoint.get_slice(key)
        assert table.get_shape() == [vocab, hidden]
        expected = torch.cat([table[token : token + 1] for token in ids]).float().numpy()
        expected_bytes = expected.astype("<f4").tobytes()
    entry = next(
        entry
        for entry in ModelCatalogue(directory.parent).discover()
        if entry._snapshot.directory == directory
    )
    with tempfile.TemporaryDirectory(prefix="embedding-smoke-") as cache:
        settings = Settings(model_root=directory.parent, cache_dir=Path(cache))
        with TestClient(create_app(settings)) as client:
            session = client.post("/sessions", json={"model_id": entry.summary.id})
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
                    assert meta["token_ids"] == ids and meta["shape"] == [5, hidden]
                elif kind == 2:
                    actual.extend(data)
                offset += 12 + size
            assert kinds[0] == 1 and kinds[-1] == 4 and 5 not in kinds
            assert actual == expected_bytes
        # Startup may publish structured architecture artifacts; row lookup must
        # never publish numeric artifacts or temporary partial entries.
        assert all(
            (item / "manifest.json").is_file()
            and json.loads((item / "manifest.json").read_text()).get("format") == 2
            for item in Path(cache).iterdir()
        )
    print(
        json.dumps(
            {
                "outcome": "PASS",
                "reference": entry.summary.id,
                "table_shape": [vocab, hidden],
                "token_ids": ids,
                "shape": [5, hidden],
                "bytes": len(actual),
                "sha256": hashlib.sha256(actual).hexdigest(),
                "numeric_cache_entries": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
