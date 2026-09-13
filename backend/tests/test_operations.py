"""Deterministic lifecycle barriers and real disk spools; no CUDA hardware needed."""

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest

from llm_model_explorer.artifacts import ArtifactSpec, ArtifactStore
from llm_model_explorer.execution import BlockingWork
from llm_model_explorer.operations import (
    MAX_READ_BYTES,
    Cancellation,
    Consumer,
    DeviceScheduler,
    OperationCancelled,
    OperationRuntime,
    ProducerContext,
)


def spec(key: str = "tensor", length: int = 8) -> ArtifactSpec:
    return ArtifactSpec(
        model_fingerprint="snapshot",
        source="weight",
        operation=key,
        parameters={},
        dtype="float32",
        layout="C",
        shape=(length // 4,),
        expected_bytes=length,
        producer="test",
    )


@asynccontextmanager
async def runtime(tmp_path: Path) -> AsyncIterator[OperationRuntime]:
    work = BlockingWork()
    service = OperationRuntime(
        ArtifactStore(tmp_path / "cache", model_root=tmp_path / "models"), work
    )
    try:
        yield service
    finally:
        await service.aclose()
        assert not service._operations
        assert not service._flights
        assert not service._consumers
        assert not list(service.store.root.glob(".tmp-*"))
        await work.aclose()


async def drain(consumer: Consumer) -> bytes:
    chunks = []
    try:
        while data := await consumer.read(4):
            chunks.append(data)
        return b"".join(chunks)
    finally:
        await consumer.aclose()


def run(test: Awaitable[None]) -> None:
    async def bounded() -> None:
        async with asyncio.timeout(15):
            await test

    asyncio.run(bounded())


def test_singleflight_late_join_distinct_keys_and_cache_hit(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            prefix, finish = asyncio.Event(), asyncio.Event()
            calls = 0

            async def produce(ctx: ProducerContext) -> None:
                nonlocal calls
                calls += 1
                await ctx.append(b"abcd")
                prefix.set()
                await ctx.cancellation.wait(finish)
                await ctx.append(b"efgh")

            a, b = await asyncio.gather(
                service.subscribe(uuid4(), spec(), produce),
                service.subscribe(uuid4(), spec(), produce),
            )
            await prefix.wait()
            assert await a.read(4) == b"abcd"
            c = await service.subscribe(uuid4(), spec(), produce)
            assert await c.read(4) == b"abcd"
            assert len({a.operation_id, b.operation_id, c.operation_id}) == 3
            other = await service.subscribe(uuid4(), spec("other"), produce)
            finish.set()
            assert await drain(a) == b"efgh"
            assert await drain(b) == b"abcdefgh"
            assert await drain(c) == b"efgh"
            assert await drain(other) == b"abcdefgh"
            assert calls == 2
            hit = await service.subscribe(uuid4(), spec(), produce)
            assert hit.operation_id not in {a.operation_id, b.operation_id, c.operation_id}
            assert await drain(hit) == b"abcdefgh"
            assert calls == 2

    run(scenario())


def test_cancel_one_then_all_and_retry_same_key(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            started, finish = asyncio.Event(), asyncio.Event()

            async def produce(ctx: ProducerContext) -> None:
                await ctx.append(b"abcd")
                started.set()
                await ctx.cancellation.wait(finish)
                await ctx.append(b"efgh")

            a = await service.subscribe(uuid4(), spec(), produce)
            b = await service.subscribe(uuid4(), spec(), produce)
            await started.wait()
            assert a.operation_id is not None
            await service.cancel(a.operation_id)
            assert not b._flight.cancellation.requested.is_set()
            with pytest.raises(OperationCancelled):
                await a.read()
            assert await b.read(4) == b"abcd"
            flight = b._flight
            await b.aclose()
            await flight.finished.wait()
            assert flight.error is not None
            assert service.store.lookup(spec()) is None
            await service.cancel(a.operation_id)  # Removed IDs remain idempotent.
            finish.set()
            retry = await service.subscribe(uuid4(), spec(), produce)
            assert await drain(retry) == b"abcdefgh"

    run(scenario())


def test_before_launch_and_session_cancellation(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            calls = 0

            async def produce(ctx: ProducerContext) -> None:
                nonlocal calls
                calls += 1
                await ctx.append(b"abcdefgh")

            session = uuid4()
            a = await service.subscribe(session, spec(), produce)
            b = await service.subscribe(session, spec(), produce)
            await service.cancel_session(session)
            await a._flight.finished.wait()
            assert calls == 0
            assert b.terminal == "cancelled"
            c = await service.subscribe(session, spec(), produce)
            d = await service.subscribe(uuid4(), spec(), produce)
            await service.cancel_session(session)
            assert c.terminal == "cancelled"
            assert await drain(d) == b"abcdefgh"
            assert calls == 1

    run(scenario())


def test_cancel_blocked_io_defers_writer_cleanup(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            entered, release = Event(), Event()

            def blocked() -> bytes:
                entered.set()
                assert release.wait(5)
                return b"abcdefgh"

            async def produce(ctx: ProducerContext) -> None:
                await ctx.append(await ctx.io(blocked))

            a = await service.subscribe(uuid4(), spec(), produce)
            assert await asyncio.to_thread(entered.wait, 5)
            await a.aclose()
            assert list(service.store.root.glob(".tmp-*"))
            assert not a._flight.finished.is_set()
            release.set()
            await a._flight.finished.wait()
            assert service.store.lookup(spec()) is None
            assert not list(service.store.root.glob(".tmp-*"))

    run(scenario())


def test_slow_reader_bounded_replay_does_not_stall_fast_reader(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            chunk = b"x" * MAX_READ_BYTES
            large = spec(length=16 * len(chunk))

            async def produce(ctx: ProducerContext) -> None:
                for _ in range(16):
                    await ctx.append(chunk)

            slow = await service.subscribe(uuid4(), large, produce)
            fast = await service.subscribe(uuid4(), large, produce)
            assert len(await slow.read()) == len(chunk)
            total = 0
            while data := await fast.read():
                assert len(data) <= MAX_READ_BYTES
                total += len(data)
            assert total == large.expected_bytes
            assert slow._reader is not None
            reader = slow._reader
            assert reader._payload.tell() == len(chunk)
            assert not any(isinstance(v, (bytes, bytearray, list)) for v in vars(slow).values())
            assert len(await slow.read()) == len(chunk)
            await slow.aclose()
            await fast.aclose()
            assert reader._payload.closed
            hit = service.store.lookup(large)
            assert hit is not None
            hit.close()

    run(scenario())


def test_failure_disconnect_and_terminal_cleanup(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            prefix = asyncio.Event()
            forever = asyncio.Event()

            async def produce(ctx: ProducerContext) -> None:
                await ctx.append(b"abcd")
                prefix.set()
                await ctx.cancellation.wait(forever)

            a = await service.subscribe(uuid4(), spec(), produce)
            await prefix.wait()
            assert await a.read(4) == b"abcd"
            reader = a._reader
            read = asyncio.create_task(a.read())
            await asyncio.sleep(0)
            read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await read
            await a._flight.finished.wait()
            assert reader is not None and reader._payload.closed
            assert a.operation_id not in service._operations
            assert a.terminal == "cancelled"

            async def fail(ctx: ProducerContext) -> None:
                await ctx.append(b"abcd")
                raise ValueError("producer failure")

            b = await service.subscribe(uuid4(), spec(), fail)
            with pytest.raises(ValueError, match="producer failure"):
                await drain(b)
            assert b.terminal == "error"
            assert service.store.lookup(spec()) is None

    run(scenario())


def test_cancel_after_commit_preserves_artifact_and_terminal(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:

            async def produce(ctx: ProducerContext) -> None:
                await ctx.append(b"abcdefgh")

            a = await service.subscribe(uuid4(), spec(), produce)
            await a._flight.finished.wait()
            assert a.operation_id is not None
            await service.cancel(a.operation_id)
            assert a.terminal == "cancelled"
            b = await service.subscribe(uuid4(), spec(), produce)
            assert await drain(b) == b"abcdefgh"
            assert b.operation_id is not None
            await service.cancel(b.operation_id)
            assert b.terminal == "complete"

    run(scenario())


def test_device_queue_cancellation_and_separate_devices(tmp_path: Path) -> None:
    async def scenario() -> None:
        work = BlockingWork()
        scheduler = DeviceScheduler(work)
        entered, release = Event(), Event()
        launched: list[str] = []

        def kernel(name: str) -> None:
            launched.append(name)
            if name == "active":
                entered.set()
                assert release.wait(5)

        try:
            first_token, queued_token = Cancellation(), Cancellation()
            first = asyncio.create_task(scheduler.run("cuda", first_token, kernel, "active"))
            assert await asyncio.to_thread(entered.wait, 5)
            queued = asyncio.create_task(scheduler.run("cuda:0", queued_token, kernel, "queued"))
            await asyncio.sleep(0)
            queued_token.cancel()
            with pytest.raises(OperationCancelled):
                await queued
            await scheduler.run("cuda:1", Cancellation(), kernel, "other-device")
            await scheduler.run("cpu", Cancellation(), kernel, "cpu")
            first_token.cancel()
            assert not first.done()  # Active kernel is not preempted.
            release.set()
            with pytest.raises(OperationCancelled):
                await first
            await scheduler.run("cuda:0", Cancellation(), kernel, "next")
            assert launched == ["active", "other-device", "cpu", "next"]
        finally:
            release.set()
            await work.aclose()

    run(scenario())


def test_nested_dependency_releases_interest_before_gpu_slot(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            service.device = "cuda:0"  # Fake blocking callbacks, no CUDA API invocation.
            calls = []

            async def materialize(ctx: ProducerContext) -> None:
                data = await ctx.compute(lambda: b"abcdefgh")
                calls.append("materialize")
                await ctx.append(data)

            async def statistics(ctx: ProducerContext) -> None:
                async with ctx.dependency(spec(), materialize) as dependency:
                    assert dependency.operation_id is None
                    data = await drain(dependency)
                await ctx.append(await ctx.compute(lambda: data[:4]))
                calls.append("statistics")

            result = await service.subscribe(uuid4(), spec("statistics", 4), statistics)
            assert await drain(result) == b"abcd"
            assert calls == ["materialize", "statistics"]

    run(scenario())


def test_nested_dependency_cancellation_propagates(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            started = asyncio.Event()

            async def materialize(ctx: ProducerContext) -> None:
                started.set()
                await ctx.cancellation.wait(asyncio.Event())

            async def statistics(ctx: ProducerContext) -> None:
                async with ctx.dependency(spec(), materialize) as dependency:
                    await drain(dependency)

            consumer = await service.subscribe(uuid4(), spec("statistics"), statistics)
            await started.wait()
            await consumer.aclose()
            await consumer._flight.finished.wait()

    run(scenario())


def test_cancel_during_commit_preserves_completed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_model_explorer.artifacts import ArtifactWriter

    entered, release = Event(), Event()
    original = ArtifactWriter.commit

    def commit(writer: ArtifactWriter) -> None:
        entered.set()
        assert release.wait(5)
        original(writer)

    monkeypatch.setattr(ArtifactWriter, "commit", commit)

    async def scenario() -> None:
        async with runtime(tmp_path) as service:

            async def produce(ctx: ProducerContext) -> None:
                await ctx.append(b"abcdefgh")

            consumer = await service.subscribe(uuid4(), spec(), produce)
            assert await asyncio.to_thread(entered.wait, 5)
            await consumer.aclose()
            release.set()
            await consumer._flight.finished.wait()
            assert consumer._flight.committed
            hit = service.store.lookup(spec())
            assert hit is not None
            with hit:
                assert hit.read_available(8) == b"abcdefgh"

    run(scenario())


def test_disconnect_during_blocking_read_retains_handle_until_read_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_model_explorer.artifacts import ArtifactReader

    entered, release = Event(), Event()
    original = ArtifactReader.read_available
    readers: list[ArtifactReader] = []

    def read(reader: ArtifactReader, bound: int) -> bytes:
        readers.append(reader)
        entered.set()
        assert release.wait(5)
        assert not reader._payload.closed
        return original(reader, bound)

    monkeypatch.setattr(ArtifactReader, "read_available", read)

    async def scenario() -> None:
        async with runtime(tmp_path) as service:

            async def produce(ctx: ProducerContext) -> None:
                await ctx.append(b"abcdefgh")

            consumer = await service.subscribe(uuid4(), spec(), produce)
            await consumer._flight.finished.wait()
            task = asyncio.create_task(consumer.read())
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()  # Repeated disconnect cancellation cannot close a live read.
            await asyncio.sleep(0)
            assert not task.done()
            assert not readers[0]._payload.closed
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert readers[0]._payload.closed
            assert consumer.operation_id not in service._operations

    run(scenario())


def test_external_task_cancel_keeps_gpu_slot_until_kernel_settles(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            entered, release, next_entered = Event(), Event(), Event()

            def kernel() -> None:
                entered.set()
                assert release.wait(5)

            first = asyncio.create_task(service.scheduler.run("cuda:0", Cancellation(), kernel))
            assert await asyncio.to_thread(entered.wait, 5)
            first.cancel()
            second = asyncio.create_task(
                service.scheduler.run("cuda:0", Cancellation(), next_entered.set)
            )
            await asyncio.sleep(0)
            assert not next_entered.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            await second
            assert next_entered.is_set()

    run(scenario())


@asynccontextmanager
async def saturated_executor(work: BlockingWork) -> AsyncIterator[None]:
    """Occupy every real worker until the test explicitly leaves this context."""
    loop = asyncio.get_running_loop()
    release = Event()
    entered = [asyncio.Event() for _ in range(work._executor._max_workers)]

    def occupy(ready: asyncio.Event) -> None:
        loop.call_soon_threadsafe(ready.set)
        assert release.wait(10)

    workers = [asyncio.create_task(work.run(occupy, ready)) for ready in entered]
    try:
        await asyncio.gather(*(ready.wait() for ready in entered))
        yield
    finally:
        release.set()
        await asyncio.gather(*workers)


async def wait_for_executor_submission(work: BlockingWork) -> None:
    # All workers are held at a barrier, so the queued callback cannot disappear.
    # Yield until submission, without assuming how many event-loop turns it takes.
    while work._executor._work_queue.empty():
        await asyncio.sleep(0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cancel_compute_queued_in_saturated_executor(tmp_path: Path, device: str) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            cancellation = Cancellation()
            launched = Event()
            async with saturated_executor(service.work):
                queued = asyncio.create_task(
                    service.scheduler.run(device, cancellation, launched.set)
                )
                await wait_for_executor_submission(service.work)
                cancellation.cancel()
            with pytest.raises(OperationCancelled):
                await queued
            assert not launched.is_set()
            # The skipped callback also releases its slot for subsequent live work.
            assert await service.scheduler.run(device, Cancellation(), lambda: 42) == 42

    run(scenario())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("cancel_last", [True, False])
def test_consumer_cancellation_of_compute_in_saturated_executor(
    tmp_path: Path, device: str, cancel_last: bool
) -> None:
    async def scenario() -> None:
        async with runtime(tmp_path) as service:
            service.device = device
            ready, compute = asyncio.Event(), asyncio.Event()
            launched = Event()

            def kernel() -> bytes:
                launched.set()
                return b"abcdefgh"

            async def produce(ctx: ProducerContext) -> None:
                ready.set()  # Lookup/writer creation have completed before saturation.
                await ctx.cancellation.wait(compute)
                await ctx.append(await ctx.compute(kernel))

            first = await service.subscribe(uuid4(), spec(), produce)
            second = await service.subscribe(uuid4(), spec(), produce)
            await ready.wait()
            async with saturated_executor(service.work):
                compute.set()
                await wait_for_executor_submission(service.work)
                assert first.operation_id is not None
                await service.cancel(first.operation_id)
                assert not first._flight.cancellation.requested.is_set()
                if cancel_last:
                    assert second.operation_id is not None
                    await service.cancel(second.operation_id)
                    assert second._flight.cancellation.requested.is_set()
            await second._flight.finished.wait()
            if cancel_last:
                assert not launched.is_set()
                assert isinstance(second._flight.error, OperationCancelled)
                assert service.store.lookup(spec()) is None
                assert first.terminal == second.terminal == "cancelled"
            else:
                assert launched.is_set()
                assert await drain(second) == b"abcdefgh"
                assert first.terminal == "cancelled"
                assert second.terminal == "complete"
            assert not service._operations
            assert not service._flights
            assert not list(service.store.root.glob(".tmp-*"))
            assert await service.scheduler.run(device, Cancellation(), lambda: 42) == 42

    run(scenario())
