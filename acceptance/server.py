"""Real Uvicorn service with loopback-only test delay/fault/observation controls.

Never shipped or imported by the application. Calculations, storage, sessions,
HTTP framing and tokenization remain the production implementations.
"""

import argparse
import asyncio
import json
import tempfile
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch
from weakref import WeakSet

import torch
import uvicorn
from fastapi import APIRouter
from llm_model_explorer.app import create_app
from llm_model_explorer.operations import Consumer, ProducerContext
from llm_model_explorer.services import open_services
from llm_model_explorer.settings import Settings
from llm_model_explorer.tensor_analysis import TensorAnalysis
from llm_model_explorer.tensor_source import ModelSource

from acceptance.fixtures import generate


def application(root: Path, origin: str, device: str = "cpu"):
    control = {"kind": "", "mode": "", "entered": False, "released": False}
    counts = Counter()
    timings = {}
    release = asyncio.Event()
    append = ProducerContext.append
    calculate = TensorAnalysis._calculate
    read = Consumer.read
    iter_rows = ModelSource.iter_rows
    iter_tensor = ModelSource.iter_tensor
    started = WeakSet()
    source_reads = {"row_elements": 0, "max_row_block": 0, "full_tensors": []}

    async def observed_read(consumer, *args, **kwargs):
        kind = json.loads(consumer._flight.spec.canonical)["operation"]
        if kind in {
            "input_embeddings",
            "input_embeddings_statistics",
            "input_embeddings_distributions",
        }:
            if control["kind"] == kind and control["mode"] == "pre-meta-error":
                raise RuntimeError("Injected embedding analysis failure")
            if consumer in started and control["kind"] == kind:
                await barrier(consumer)
            started.add(consumer)
        return await read(consumer, *args, **kwargs)

    def observed_rows(source, *args, **kwargs):
        with_iterator = iter_rows(source, *args, **kwargs)
        try:
            for block in with_iterator:
                source_reads["row_elements"] += block.numel()
                source_reads["max_row_block"] = max(source_reads["max_row_block"], block.numel())
                yield block
        finally:
            with_iterator.close()

    def observed_tensor(source, tensor_id, **kwargs):
        source_reads["full_tensors"].append(tensor_id)
        yield from iter_tensor(source, tensor_id, **kwargs)

    async def barrier(context):
        control["entered"] = True
        await context.cancellation.wait(release)

    async def observed_append(context, data):
        spec = json.loads(context._flight.spec.canonical)
        kind = spec["operation"]
        available = context._flight.writer.available_bytes
        first = available == 0
        checkpoint = available == (16 if kind == "tensor_distributions" else 0)
        await append(context, data)
        if first:
            timings[f"{kind}:first_append"] = time.monotonic()
        if checkpoint and kind == control["kind"]:
            await barrier(context)
            if control["mode"] == "midstream-error":
                raise RuntimeError("Injected producer failure")

    def observed_calculate(analysis, *args):
        if analysis.kind == control["kind"] and control["mode"] == "pre-meta-error":
            raise RuntimeError("Injected pre-META computation failure")
        return calculate(analysis, *args)

    @asynccontextmanager
    async def lifespan(settings):
        async with open_services(settings) as services:
            store = services.artifacts
            begin = store.begin_write

            def begin_write(spec):
                counts[spec.key] += 1
                return begin(spec)

            with (
                patch.object(ProducerContext, "append", observed_append),
                patch.object(Consumer, "read", observed_read),
                patch.object(ModelSource, "iter_rows", observed_rows),
                patch.object(ModelSource, "iter_tensor", observed_tensor),
                patch.object(TensorAnalysis, "_calculate", observed_calculate),
                patch.object(store, "begin_write", begin_write),
            ):
                yield services

    router = APIRouter()

    @router.post("/__test/arm")
    async def arm(body: dict):
        release.clear()
        control.update(
            kind=body["kind"],
            mode=body.get("mode", "barrier"),
            entered=False,
            released=False,
        )
        return control

    @router.post("/__test/release")
    async def unblock():
        control["released"] = True
        release.set()
        return control

    @router.get("/__test/state")
    async def state():
        runtime = app.state.services.operation_delivery
        consumers = runtime._consumers
        artifacts = {}
        for manifest in sorted((root / "cache").glob("*/manifest.json")):
            metadata = json.loads(manifest.read_text())
            if metadata["format"] != 1:
                continue  # This observer measures numerical production, not startup graphs.
            payload = manifest.parent / "payload.bin"
            artifacts[manifest.parent.name] = {
                "manifest": metadata,
                "inode": payload.stat().st_ino,
                "mtime_ns": payload.stat().st_mtime_ns,
            }
        return {
            "control": control,
            "productions": dict(counts),
            "timings": timings,
            "operations": len(runtime._operations),
            "consumers": len(consumers),
            "readers": sum(
                c._reader is not None or c._source_reader is not None for c in consumers
            ),
            "flights": len(runtime._flights),
            "tasks": len(runtime._tasks),
            "artifacts": artifacts,
            "temporary": len(list((root / "cache").glob(".tmp-*"))),
            "device": runtime.device,
            "source_reads": source_reads,
        }

    settings = Settings(
        model_root=root / "models",
        cache_dir=root / "cache",
        cors_origins=(origin,),
        device=device,
    )
    app = create_app(settings, routers=[router], service_lifespan=lifespan)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--origin", default="http://127.0.0.1:4175")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--polish", action="store_true")
    args = parser.parse_args()
    # Bound fixture kernels on shared CI hosts; no performance claim about this setting.
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="lmex-acceptance-") as temporary:
        root = args.root or Path(temporary)
        if not (root / "models").exists():
            if args.polish:
                from acceptance.polish_fixtures import generate as generate_polish

                generate_polish(root / "models")
            else:
                generate(root / "models", extended=True)
        uvicorn.run(
            application(root, args.origin, args.device),
            host="127.0.0.1",
            port=args.port,
            log_level="warning",
        )


if __name__ == "__main__":
    main()
