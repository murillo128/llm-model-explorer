"""Real Linux CPU allocator exhaustion, isolated from the pytest process."""

import json
import os
import resource
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from test_models import make_model, write_weights
from test_tensor_data import address, frames

from llm_model_explorer.app import create_app
from llm_model_explorer.settings import Settings


def main() -> None:
    root = Path(sys.argv[1])
    model_root = root / "models"
    model_root.mkdir()
    directory = make_model(model_root)
    # No large input fixture: the empty tensor still requires 800 MB of int64
    # histogram workspace under the accepted section geometry.
    write_weights(directory / "model.safetensors", [("weight", "F32", [0, 1_000_000], [])])
    settings = Settings(model_root=model_root, cache_dir=root / "cache")
    with TestClient(create_app(settings)) as client:
        url = address(client).removesuffix("data")
        assert frames(client.get(url + "statistics").content)[-1] == (4, b"")
        limits = resource.getrlimit(resource.RLIMIT_AS)
        virtual = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        constrained = virtual + 256 * 1024**2
        try:
            resource.setrlimit(resource.RLIMIT_AS, (constrained, limits[1]))
            try:
                torch.empty(200_000_000, dtype=torch.float32)
            except RuntimeError as exc:
                allocator_type, allocator_message = type(exc).__name__, str(exc)
            else:
                raise AssertionError("native allocation unexpectedly succeeded under RLIMIT_AS")
            response = client.get(url + "distributions")
        finally:
            resource.setrlimit(resource.RLIMIT_AS, limits)
        result = frames(response.content)
        print(
            json.dumps(
                dict(
                    allocator_type=allocator_type,
                    allocator_message=allocator_message,
                    status=response.status_code,
                    frame_types=[kind for kind, _ in result],
                    error=json.loads(result[-1][1]),
                    complete_artifacts=len(list(settings.cache_dir.glob("*/manifest.json"))),
                    temporary_artifacts=len(list(settings.cache_dir.glob(".tmp-*"))),
                )
            )
        )


if __name__ == "__main__":
    main()
