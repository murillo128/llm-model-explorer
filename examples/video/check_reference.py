"""Complete local reference acceptance over the repository's production TCP harness.

No download or upstream model runtime. Run from the repository with backend Python.
The independent numerical oracle is Safetensors slices and the pinned loader's
float32(q) * float32(weight_scale) rule, never the application's decoder.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from acceptance.test_network import Frames, Service  # noqa: E402
from api.architecture_conformance import validate_architecture  # noqa: E402


def check(package: Path, source_root: Path) -> dict[str, object]:
    package = package.resolve(strict=True)
    config = json.loads((package / "config.json").read_text())
    provenance = json.loads((package / "video-provenance.json").read_text())
    profile = config["video_inspection"]["profile"]
    prefix = "video.denoiser." if "a14b" not in profile else "video.high_noise."
    selected = [
        prefix
        + (
            "transformer_blocks.1.attn1.to_q.weight"
            if profile.startswith("ltx")
            else "blocks.1.self_attn.q.weight"
        ),
        "video.text_encoder.encoder.block.1.layer.0.SelfAttention.q.weight",
    ]
    if "a14b" in profile:
        selected.append("video.low_noise.blocks.1.self_attn.q.weight")
    report: dict[str, object] = {
        "profile": profile,
        "sources": provenance["sources"],
        "package_bytes": sum(p.stat().st_size for p in package.iterdir() if p.is_file()),
        "numerical_oracle": "Safetensors original slices; float32(q) * weight scale only",
    }
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="video-reference-") as temporary:
        service = Service(
            Path(temporary), model_root=package.parent, startup_timeout=300, client_timeout=180
        )
        try:
            models = service.client.get("/models").json()["models"]
            model = next(m for m in models if m["id"].startswith(config["_name_or_path"]))
            session = service.client.post("/sessions", json={"model_id": model["id"]}).json()
            address = f"/sessions/{session['id']}"
            inventory = service.client.get(address + "/tensors").json()
            response = service.client.get(address + "/architecture")
            response.raise_for_status()
            body = response.json()
            assert body["status"] == "available", body
            graph = body["graph"]
            assert graph["scope"] == "video_generation" and graph["coverage"] == "complete"
            validate_architecture(
                body,
                {
                    "session": session,
                    "inventory": inventory,
                    "tokenizer_available": model["tokenizer_available"],
                },
            )
            report.update(
                model_id=model["id"],
                graph_id=graph["graph_id"],
                scope=graph["scope"],
                graph_bytes=len(response.content),
                nodes=len(graph["nodes"]),
                parameters=len(graph["parameters"]),
                tensors=len(inventory["tensors"]),
            )
            tokenized = service.client.post(
                address + "/tokenize", json={"text": "A bird flies over the sea."}
            )
            tokenized.raise_for_status()
            tokens = tokenized.json()["tokens"]
            assert tokens
            report["token_ids"] = [t["id"] for t in tokens]
            measurements = []
            for name in selected:
                binding = next(t for t in inventory["tensors"] if t["name"] == name)
                original = provenance["tensors"][name]
                matches = list(source_root.rglob(Path(original["artifact"]).name))
                assert len(matches) == 1, original["artifact"]
                expected = hashlib.sha256()
                samples = []
                with safe_open(matches[0], framework="pt", device="cpu") as upstream:
                    weight = upstream.get_slice(original["original_name"])
                    scale_name = original["original_name"].removesuffix(".weight") + ".scale_weight"
                    scale = (
                        upstream.get_tensor(scale_name).float()
                        if scale_name in upstream.keys()
                        else None
                    )
                    for row in range(0, binding["shape"][0], 64):
                        chunk = weight[row : row + 64].float()
                        if scale is not None:
                            chunk = chunk * scale
                        if row == 0:
                            samples = chunk[0, :3].tolist()
                        expected.update(chunk.contiguous().numpy().tobytes())
                actual = hashlib.sha256()
                count = 0
                with service.client.stream(
                    "GET", address + f"/tensors/{binding['id']}/data"
                ) as stream:
                    stream.raise_for_status()
                    frames = Frames(stream)
                    kind, metadata = frames.next()
                    assert kind == 1 and json.loads(metadata)["shape"] == binding["shape"]
                    terminal = None
                    for kind, data in frames.rest():
                        if kind == 2:
                            actual.update(data)
                            count += len(data)
                        else:
                            terminal = kind
                    assert terminal == 4 and count == 4 * binding["numel"]
                assert actual.digest() == expected.digest(), name
                measurements.append(
                    {
                        "name": name,
                        "shape": binding["shape"],
                        "logical_bytes": count,
                        "sha256_float32": actual.hexdigest(),
                        "first_three": samples,
                        "storage_format": binding.get("storage_format", "safetensors"),
                    }
                )
            report["weights"] = measurements
            assert service.process is not None
            status = Path(f"/proc/{service.process.pid}/status").read_text().splitlines()
            report["backend_peak_rss_kib"] = int(
                next(line for line in status if line.startswith("VmHWM:")).split()[1]
            )
        finally:
            service.stop()
            service.client.close()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    args.output.write_text(json.dumps(check(args.package, args.source_root), indent=2) + "\n")
