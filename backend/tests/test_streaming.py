"""Real TCP streaming barriers plus deterministic transport failure/backpressure."""

import asyncio
import hashlib
import json
import socket
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
import uvicorn
from fastapi import APIRouter, FastAPI
from starlette.types import Message
from test_lmex import metadata
from test_operations import run, runtime, spec

from llm_model_explorer.artifacts import ArtifactReader, ArtifactWriter
from llm_model_explorer.dependencies import get_operation_delivery
from llm_model_explorer.model_files import ModelError
from llm_model_explorer.operations import Consumer, OperationRuntime, ProducerContext
from llm_model_explorer.session_routes import LifecycleRoute
from llm_model_explorer.session_routes import router as lifecycle_router
from llm_model_explorer.streaming import LMEXResponse, _SendFailed


class Frames:
    def __init__(self, response: httpx.Response) -> None:
        self.chunks = response.aiter_bytes()
        self.buffer = bytearray()

    async def take(self, count: int) -> bytes:
        while len(self.buffer) < count:
            self.buffer.extend(await anext(self.chunks))
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    async def next(self) -> tuple[int, bytes]:
        magic, kind, flags, reserved, length = struct.unpack("<4sBBHI", await self.take(12))
        assert (magic, flags, reserved) == (b"LMEX", 0, 0)
        return kind, await self.take(length)

    async def end(self) -> None:
        assert not self.buffer
        with pytest.raises(StopAsyncIteration):
            await anext(self.chunks)


@asynccontextmanager
async def server(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    instance = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="off"))
    task = asyncio.create_task(instance.serve(sockets=[sock]))
    try:
        while not instance.started:
            if task.done():
                task.result()
            await asyncio.sleep(0.01)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
            yield client
    finally:
        instance.should_exit = True
        await task
        sock.close()


def fixture_app(service: OperationRuntime) -> tuple[FastAPI, APIRouter]:
    app = FastAPI()
    app.dependency_overrides[get_operation_delivery] = lambda: service
    app.include_router(lifecycle_router)
    router = APIRouter(route_class=LifecycleRoute)

    @router.get("/fixture/metadata")
    async def small() -> dict[str, bool]:
        return {"responsive": True}

    return app, router


async def immediate() -> object:
    return metadata()


async def forgotten(service: OperationRuntime) -> None:
    async with asyncio.timeout(3):
        while service._operations or service._consumers or service._flights:
            await asyncio.sleep(0.01)


def test_socket_progress_two_consumers_delete_and_cache_hit(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            app, router = fixture_app(service)
            finish, meta_ready = asyncio.Event(), asyncio.Event()
            calls = 0
            consumers: list[Consumer] = []

            async def produce(ctx: ProducerContext) -> None:
                nonlocal calls
                calls += 1
                await ctx.append(b"abcd")
                await ctx.cancellation.wait(finish)
                await ctx.append(b"efgh")

            async def delayed_meta() -> object:
                await meta_ready.wait()
                return metadata()

            @router.get("/fixture/stream")
            async def stream() -> LMEXResponse:
                consumer = await service.subscribe(uuid4(), spec(), produce)
                consumers.append(consumer)
                return LMEXResponse(consumer, delayed_meta, max_data_bytes=4)

            app.include_router(router)
            async with server(app) as client:
                async with client.stream("GET", "/fixture/stream") as a:
                    assert a.headers["content-type"] == LMEXResponse.media_type
                    assert a.headers["cache-control"] == "no-store"
                    assert "content-length" not in a.headers
                    assert a.headers["x-operation-id"] == str(consumers[0].operation_id)
                    assert not meta_ready.is_set() and not finish.is_set()
                    meta_ready.set()
                    fa = Frames(a)
                    assert (await fa.next())[0] == 1
                    assert await fa.next() == (2, b"abcd")
                    assert not finish.is_set()  # Actual socket bytes precede producer completion.
                    assert service.store.lookup(spec()) is None
                    async with client.stream("GET", "/fixture/stream") as b:
                        fb = Frames(b)
                        assert (await fb.next())[0] == 1
                        assert await fb.next() == (2, b"abcd")
                        assert a.headers["x-operation-id"] != b.headers["x-operation-id"]
                        cancelled = await client.delete(
                            f"/operations/{a.headers['x-operation-id']}"
                        )
                        assert cancelled.status_code == 204
                        assert await fa.next() == (6, b"")
                        await fa.end()
                        assert not consumers[1]._flight.cancellation.requested.is_set()
                        assert (await client.get("/fixture/metadata")).json() == {
                            "responsive": True
                        }
                        finish.set()
                        assert await fb.next() == (2, b"efgh")
                        assert await fb.next() == (4, b"")
                        await fb.end()
                await forgotten(service)
                async with client.stream("GET", "/fixture/stream") as cached:
                    frames = Frames(cached)
                    assert (await frames.next())[0] == 1
                    assert await frames.next() == (2, b"abcd")
                    assert await frames.next() == (2, b"efgh")
                    assert await frames.next() == (4, b"")
                    await frames.end()
                await forgotten(service)
                assert calls == 1
                assert all(c._reader is None and c._closed for c in consumers)

    run(scenario())


@pytest.mark.parametrize("mode", ["early-error", "early-cancel", "mid-error", "disconnect"])
def test_socket_terminal_paths(tmp_path: Path, mode: str) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            app, router = fixture_app(service)
            finish = asyncio.Event()
            consumers: list[Consumer] = []
            metadata_closed = asyncio.Event()

            async def produce(ctx: ProducerContext) -> None:
                await ctx.append(b"abcd")
                await ctx.cancellation.wait(finish)
                raise OSError("/private/model/path must not appear on wire")

            async def get_meta() -> object:
                try:
                    if mode == "early-error":
                        raise OSError("/private/metadata")
                    if mode == "early-cancel":
                        await asyncio.Event().wait()
                    return metadata()
                finally:
                    metadata_closed.set()

            @router.get("/fixture/stream")
            async def stream() -> LMEXResponse:
                c = await service.subscribe(uuid4(), spec(), produce)
                consumers.append(c)
                return LMEXResponse(c, get_meta, max_data_bytes=4)

            @router.get("/fixture/invalid")
            async def invalid() -> LMEXResponse:
                raise ModelError("unsupported_rank", "Distributions require rank two.")

            app.include_router(router)
            async with server(app) as client:
                invalid_response = await client.get("/fixture/invalid")
                assert invalid_response.status_code == 422
                assert invalid_response.json()["code"] == "unsupported_rank"
                assert "x-operation-id" not in invalid_response.headers
                assert not service._operations
                async with client.stream("GET", "/fixture/stream") as response:
                    frames = Frames(response)
                    if mode in ("mid-error", "disconnect"):
                        assert (await frames.next())[0] == 1
                        assert await frames.next() == (2, b"abcd")
                    if mode == "early-cancel":
                        assert (
                            await client.delete(f"/operations/{response.headers['x-operation-id']}")
                        ).status_code == 204
                        assert await frames.next() == (6, b"")
                    elif mode != "disconnect":
                        finish.set()
                        kind, payload = await frames.next()
                        assert kind == 5
                        assert json.loads(payload)["code"] == "internal_error"
                        assert b"/private" not in payload
                    if mode != "disconnect":
                        await frames.end()
                await forgotten(service)
                assert consumers[0]._reader is None and consumers[0]._closed
                assert metadata_closed.is_set()
                assert service.store.lookup(spec()) is None

    run(scenario())


@pytest.mark.parametrize("failure", ["read", "publication", "short", "overrun", "unaligned"])
def test_data_and_publication_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            if failure in ("read", "publication"):

                def fail(*args: object) -> Any:
                    raise OSError("/private/failure")

                monkeypatch.setattr(
                    ArtifactReader if failure == "read" else ArtifactWriter,
                    "read_available" if failure == "read" else "commit",
                    fail,
                )
            raw = {"short": b"abcd", "overrun": b"abcdefghijkl", "unaligned": b"abc"}.get(
                failure, b"abcdefgh"
            )

            async def produce(ctx: ProducerContext) -> None:
                await ctx.append(raw)

            c = await service.subscribe(uuid4(), spec(length=len(raw)), produce)
            chunks: list[bytes] = []

            async def send(message: Message) -> None:
                if message["type"] == "http.response.body":
                    chunks.append(bytes(message["body"]))

            async def receive() -> Message:
                await asyncio.Event().wait()
                return {"type": "http.disconnect"}

            await LMEXResponse(c, immediate, max_data_bytes=4)({}, receive, send)
            body = b"".join(chunks)
            offset, types = 0, []
            while offset < len(body):
                _, kind, _, _, size = struct.unpack_from("<4sBBHI", body, offset)
                types.append(kind)
                offset += 12 + size
            assert types[0] == 1 and types[-1] == 5 and 4 not in types
            assert c._reader is None and c._closed
            await forgotten(service)

    run(scenario())


@pytest.mark.parametrize("fail_send", [True, False])
def test_bounded_delivery_backpressure_and_send_cleanup(tmp_path: Path, fail_send: bool) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            length = 1024 * 1024
            prefix, release = asyncio.Event(), asyncio.Event()
            calls = 0

            async def produce(ctx: ProducerContext) -> None:
                nonlocal calls
                calls += 1
                # Deliberately split values across producer boundaries.
                for size in (1, 2, 4093):
                    await ctx.append(b"a" * size)
                prefix.set()
                await ctx.cancellation.wait(release)
                for _ in range((length - 4096) // 4096):
                    await ctx.append(b"a" * 4096)

            c = await service.subscribe(uuid4(), spec(length=length), produce)
            survivor = await service.subscribe(uuid4(), spec(length=length), produce)

            async def meta() -> object:
                return metadata(length)

            blocked = asyncio.Event()
            body_sizes: list[int] = []
            digest = hashlib.sha256()

            async def send(message: Message) -> None:
                if message["type"] == "http.response.body" and isinstance(
                    message["body"], memoryview
                ):
                    body_sizes.append(len(message["body"]))
                    digest.update(message["body"])
                    blocked.set()
                    if fail_send:
                        raise OSError("socket closed")
                    await release.wait()

            async def receive() -> Message:
                await asyncio.Event().wait()
                return {"type": "http.disconnect"}

            response = LMEXResponse(c, meta, max_data_bytes=4096)
            assert not hasattr(response, "body")
            task = asyncio.create_task(response({}, receive, send))
            await prefix.wait()
            await blocked.wait()
            app, router = fixture_app(service)
            app.include_router(router)
            async with server(app) as client:
                assert (await client.get("/fixture/metadata")).status_code == 200
            assert not release.is_set()
            if fail_send:
                with pytest.raises(_SendFailed):
                    await task
                assert c._reader is None and c._closed
                assert not survivor._flight.cancellation.requested.is_set()
            release.set()
            if not fail_send:
                await task
            total = 0
            try:
                while chunk := await survivor.read():
                    total += len(chunk)
            finally:
                await survivor.aclose()
            assert total == length and calls == 1
            assert max(body_sizes) <= 4096
            if not fail_send:
                assert digest.digest() == hashlib.sha256(b"a" * length).digest()
            await forgotten(service)

    run(scenario())


@pytest.mark.parametrize("fail_at", [0, 1, 2, 3, 4, 5, 6])
def test_send_failure_at_every_response_boundary(tmp_path: Path, fail_at: int) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:

            async def produce(ctx: ProducerContext) -> None:
                await ctx.append(b"abcdefgh")

            c = await service.subscribe(uuid4(), spec(), produce)
            sent = 0

            async def send(message: Message) -> None:
                nonlocal sent
                index = sent
                sent += 1
                if index == fail_at:
                    raise OSError("transport failure")

            async def receive() -> Message:
                await asyncio.Event().wait()
                return {"type": "http.disconnect"}

            with pytest.raises((OSError, _SendFailed)):
                await LMEXResponse(c, immediate)({}, receive, send)
            # A failed header/payload/terminal/send-end is never followed by
            # another frame, since the peer may have received a partial write.
            assert sent == fail_at + 1
            await forgotten(service)
            assert c._reader is None and c._closed

    run(scenario())


def test_task_cancellation_closes_pending_metadata(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            entered, closed = asyncio.Event(), asyncio.Event()

            async def produce(ctx: ProducerContext) -> None:
                await ctx.cancellation.wait(asyncio.Event())

            async def meta() -> object:
                entered.set()
                try:
                    await asyncio.Event().wait()
                    return metadata()
                finally:
                    closed.set()

            async def send(message: Message) -> None:
                pass

            async def receive() -> Message:
                await asyncio.Event().wait()
                return {"type": "http.disconnect"}

            c = await service.subscribe(uuid4(), spec(), produce)
            task = asyncio.create_task(LMEXResponse(c, meta)({}, receive, send))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await forgotten(service)
            assert closed.is_set() and c._closed

    run(scenario())
