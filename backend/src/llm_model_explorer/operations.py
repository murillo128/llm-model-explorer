"""App-local singleflight production and independently owned bounded file delivery.

Registry mutations run on one event loop and never suspend. Blocking work belongs
in ProducerContext.io/compute; producer coroutines only coordinate safe boundaries.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Event
from typing import ParamSpec, Protocol, TypeVar
from uuid import UUID, uuid4

from .artifacts import ArtifactReader, ArtifactSpec, ArtifactStore, ArtifactWriter
from .execution import BlockingWork

P = ParamSpec("P")
T = TypeVar("T")
MAX_READ_BYTES = 256 * 1024


async def _drain[T](task: asyncio.Task[T]) -> None:
    """Let owned work settle even under repeated HTTP task cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not task.cancelled():
        task.exception()  # Retrieve failure without replacing the caller's cancellation.


async def _complete[T](task: asyncio.Task[T]) -> T:
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _drain(task)
        raise


class OperationCancelled(Exception):
    """Distinct consumer/producer cancellation; adapters emit CANCELLED, not ERROR."""


class Cancellation:
    def __init__(self) -> None:
        self.requested = Event()  # Also safe to inspect inside blocking callbacks.
        self._wake = asyncio.Event()

    def cancel(self) -> None:
        self.requested.set()
        self._wake.set()

    def check(self) -> None:
        if self.requested.is_set():
            raise OperationCancelled()

    async def wait(self, event: asyncio.Event) -> None:
        self.check()
        tasks = [asyncio.create_task(event.wait()), asyncio.create_task(self._wake.wait())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            self.check()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class DeviceScheduler:
    """FIFO asyncio locks per canonical device; slot covers only a blocking kernel.

    CUDA callbacks must synchronize their launched work before returning. No slot
    is held across producer dependencies, file delivery, or other async work.
    """

    def __init__(self, work: BlockingWork) -> None:
        self.work = work
        self._devices: dict[str, asyncio.Lock] = {}

    async def run(
        self,
        device: str,
        cancellation: Cancellation,
        function: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        task = asyncio.create_task(self._run(device, cancellation, function, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancellation.cancel()
            await _drain(task)
            raise

    async def _run(
        self,
        device: str,
        cancellation: Cancellation,
        function: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        cancellation.check()

        def invoke() -> T:
            # A device slot does not imply an available executor worker. Check
            # again at launch, after any wait in the shared thread-pool queue.
            cancellation.check()
            return function(*args, **kwargs)

        if device == "cpu":
            result = await self.work.run(invoke)
        else:
            device = "cuda:0" if device == "cuda" else device
            lock = self._devices.setdefault(device, asyncio.Lock())
            # Cancellation wakes queued jobs immediately, without cancelling kernels.
            acquire = asyncio.create_task(lock.acquire())
            cancelled = asyncio.create_task(cancellation._wake.wait())
            acquired = False
            try:
                await asyncio.wait([acquire, cancelled], return_when=asyncio.FIRST_COMPLETED)
                if acquire.done():
                    acquired = acquire.result()
                cancellation.check()
                result = await self.work.run(invoke)
            finally:
                if not acquire.done():
                    acquire.cancel()
                cancelled.cancel()
                await asyncio.gather(acquire, cancelled, return_exceptions=True)
                if acquired:
                    lock.release()
        cancellation.check()
        return result


Producer = Callable[["ProducerContext"], Awaitable[None]]


class SourceReader(Protocol):
    """Blocking bounded reader; empty bytes means validated successful EOF."""

    def read_available(self, max_bytes: int) -> bytes: ...

    def close(self) -> None: ...


@dataclass(eq=False)
class _Flight:
    spec: ArtifactSpec
    producer: Producer
    cancellation: Cancellation = field(default_factory=Cancellation)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    file_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    interests: set[Consumer] = field(default_factory=set)
    writer: ArtifactWriter | None = None
    committed: bool = False
    error: Exception | None = None
    task: asyncio.Task[None] | None = None

    def notify(self) -> None:
        self.changed.set()
        self.changed = asyncio.Event()


class ProducerContext:
    def __init__(self, runtime: OperationRuntime, flight: _Flight) -> None:
        self.runtime = runtime
        self._flight = flight
        self.cancellation = flight.cancellation

    async def io(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        self.cancellation.check()
        result = await self.runtime.work.run(function, *args, **kwargs)
        self.cancellation.check()
        return result

    async def compute(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        return await self.runtime.scheduler.run(
            self.runtime.device, self.cancellation, function, *args, **kwargs
        )

    async def append(self, data: bytes) -> None:
        self.cancellation.check()
        writer = self._flight.writer
        assert writer is not None
        await self.runtime.work.run(writer.append, data)
        self._flight.notify()
        self.cancellation.check()

    @asynccontextmanager
    async def dependency(self, spec: ArtifactSpec, producer: Producer) -> AsyncIterator[Consumer]:
        """Acquire an internal interest before compute; always release it on exit.

        Dependency graphs must be acyclic. This is a materialization/statistics
        seam, not a general workflow engine. Internal interests have no public ID.
        """
        if spec.key == self._flight.spec.key:
            raise ValueError("an artifact cannot depend on itself")
        consumer = await self.runtime._subscribe(spec, producer, None, self.cancellation)
        try:
            yield consumer
        finally:
            await consumer.aclose()


class Consumer:
    def __init__(
        self,
        runtime: OperationRuntime,
        flight: _Flight,
        session_id: UUID | None,
        parent: Cancellation | None,
    ) -> None:
        self.runtime = runtime
        self._flight = flight
        self.session_id = session_id
        self.operation_id = uuid4() if session_id is not None else None
        self.cancellation = Cancellation()
        self._parent = parent
        self._reader: ArtifactReader | None = None
        self._source_factory: Callable[[], SourceReader] | None = None
        self._source_prepare: Callable[[Cancellation], Awaitable[SourceReader]] | None = None
        self._source_reader: SourceReader | None = None
        self.read_guard: Callable[[], None] | None = None
        self._read_lock = asyncio.Lock()
        self._closed = False
        self.terminal: str | None = None
        self._watcher: asyncio.Task[None] | None = None
        if parent is not None:
            self._watcher = asyncio.create_task(self._watch_parent())

    async def _watch_parent(self) -> None:
        assert self._parent is not None
        await self._parent._wake.wait()
        self.cancel()

    def cancel(self) -> None:
        if self.terminal is None:
            self.terminal = "cancelled"
            self.cancellation.cancel()
            self.runtime._release(self)

    async def _open_reader(self) -> None:
        flight = self._flight
        async with flight.file_lock:
            self.cancellation.check()
            if flight.committed:
                self._reader = await self.runtime.work.run(self.runtime.store.lookup, flight.spec)
                if self._reader is None:
                    raise OSError("committed artifact is unavailable")
            elif flight.writer is not None and not flight.finished.is_set():
                self._reader = await self.runtime.work.run(flight.writer.open_reader)

    async def _read(self, max_bytes: int) -> bytes:
        async with self._read_lock:
            try:
                while True:
                    self.cancellation.check()
                    if self.terminal == "complete":
                        return b""
                    if self._closed:
                        raise RuntimeError("consumer is closed")
                    if self.read_guard is not None:
                        await self.runtime.work.run(self.read_guard)
                        self.cancellation.check()
                    if self._source_factory is not None or self._source_prepare is not None:
                        if self._source_reader is None:
                            if self._source_prepare is not None:
                                self._source_reader = await self._source_prepare(self.cancellation)
                            else:
                                assert self._source_factory is not None
                                self._source_reader = await self.runtime.work.run(
                                    self._source_factory
                                )
                            self.cancellation.check()
                        data = await self.runtime.work.run(
                            self._source_reader.read_available, max_bytes
                        )
                        self.cancellation.check()
                        if data:
                            return data
                        self.terminal = "complete"
                        self.runtime._release(self)
                        await self._close_reader()
                        return b""
                    flight = self._flight
                    changed = flight.changed
                    finished = flight.finished.is_set()
                    if self._reader is None:
                        await self._open_reader()
                    if self._reader is not None:
                        data = await self.runtime.work.run(self._reader.read_available, max_bytes)
                        self.cancellation.check()
                        if self.read_guard is not None:
                            await self.runtime.work.run(self.read_guard)
                            self.cancellation.check()
                        if data:
                            return data
                    if finished:
                        if flight.error is not None:
                            raise flight.error
                        self.terminal = "complete"
                        self.runtime._release(self)
                        await self._close_reader()
                        return b""
                    await self.cancellation.wait(changed)
            except OperationCancelled:
                self.cancel()
                await self._close_reader()
                raise
            except Exception:
                if self.terminal is None:
                    self.terminal = "error"
                self.runtime._release(self)
                await self._close_reader()
                raise

    async def read(self, max_bytes: int = MAX_READ_BYTES) -> bytes:
        """One bounded chunk; empty bytes means successful EOF, exceptions are terminal.

        One caller at a time. On disconnect, wait for any blocking read to settle
        before closing its handle. No producer is ever cancelled via Task.cancel().
        """
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_READ_BYTES:
            raise ValueError(f"read bound must be between 1 and {MAX_READ_BYTES}")
        task = asyncio.create_task(self._read(max_bytes))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.cancel()
            await _drain(task)
            await _drain(asyncio.create_task(self.aclose()))
            raise

    async def _close_reader(self) -> None:
        if self._source_reader is not None:
            await self.runtime.work.run(self._source_reader.close)
            self._source_reader = None
        if self._reader is not None:
            await self.runtime.work.run(self._reader.close)
            self._reader = None

    async def aclose(self) -> None:
        self.cancel()
        await _complete(asyncio.create_task(self._dispose()))

    async def _dispose(self) -> None:
        async with self._read_lock:
            await self._close_reader()
            self._closed = True
        if self._watcher is not None:
            self._watcher.cancel()
            await asyncio.gather(self._watcher, return_exceptions=True)
            self._watcher = None
        self.runtime._forget(self)


class OperationRuntime:
    def __init__(self, store: ArtifactStore, work: BlockingWork, device: str = "cpu") -> None:
        self.store = store
        self.work = work
        self.device = device
        self.scheduler = DeviceScheduler(work)
        self._flights: dict[str, _Flight] = {}
        self._operations: dict[UUID, Consumer] = {}
        self._consumers: set[Consumer] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    async def subscribe(self, session_id: UUID, spec: ArtifactSpec, producer: Producer) -> Consumer:
        return await self._subscribe(spec, producer, session_id, None)

    def subscribe_source(
        self,
        spec: ArtifactSpec,
        factory: Callable[[], SourceReader] | None = None,
        *,
        prepare: Callable[[Cancellation], Awaitable[SourceReader]] | None = None,
        session_id: UUID | None = None,
        parent: Cancellation | None = None,
    ) -> Consumer:
        """Track direct immutable-source delivery without producing a cache copy.

        Each interest owns its file cursor. Cancellation, session teardown and
        internal parent ownership use the same consumer lifecycle as artifacts.
        Optional async preparation runs inside the shielded read lifetime, so
        cancellation settles owned computation before its reader is disposed.
        """
        if self._closed:
            raise RuntimeError("operation runtime is closed")
        if parent is not None:
            parent.check()
        if (factory is None) == (prepare is None):
            raise ValueError("provide exactly one direct source factory or preparer")

        async def unused(context: ProducerContext) -> None:
            raise AssertionError("direct sources have no producer")

        flight = _Flight(spec, unused)
        flight.finished.set()
        consumer = Consumer(self, flight, session_id, parent)
        consumer._source_factory = factory
        consumer._source_prepare = prepare
        flight.interests.add(consumer)
        self._consumers.add(consumer)
        if consumer.operation_id is not None:
            self._operations[consumer.operation_id] = consumer
        return consumer

    async def _subscribe(
        self,
        spec: ArtifactSpec,
        producer: Producer,
        session_id: UUID | None,
        parent: Cancellation | None,
    ) -> Consumer:
        while True:
            if self._closed:
                raise RuntimeError("operation runtime is closed")
            if parent is not None:
                parent.check()
            flight = self._flights.get(spec.key)
            if flight is not None and flight.cancellation.requested.is_set():
                if parent is None:
                    await flight.finished.wait()
                else:
                    await parent.wait(flight.finished)
                continue
            # No awaits from lookup through insertion: atomic on the owning loop.
            if flight is None:
                flight = _Flight(spec, producer)
                self._flights[spec.key] = flight
                flight.task = asyncio.create_task(self._produce(flight))
                self._tasks.add(flight.task)
                flight.task.add_done_callback(self._tasks.discard)
            consumer = Consumer(self, flight, session_id, parent)
            flight.interests.add(consumer)
            self._consumers.add(consumer)
            if consumer.operation_id is not None:
                self._operations[consumer.operation_id] = consumer
            return consumer

    async def _produce(self, flight: _Flight) -> None:
        try:
            flight.cancellation.check()
            async with flight.file_lock:
                cached = await self.work.run(self.store.lookup, flight.spec)
                if cached is not None:
                    await self.work.run(cached.close)
                    flight.committed = True
                else:
                    flight.cancellation.check()
                    flight.writer = await self.work.run(self.store.begin_write, flight.spec)
            flight.notify()
            if not flight.committed:
                flight.cancellation.check()
                await flight.producer(ProducerContext(self, flight))
                async with flight.file_lock:
                    flight.cancellation.check()
                    assert flight.writer is not None
                    await self.work.run(flight.writer.commit)
                    flight.committed = True
        except asyncio.CancelledError:
            flight.error = OperationCancelled()
        except Exception as exc:
            flight.error = exc
        finally:
            async with flight.file_lock:
                if flight.writer is not None:
                    await self.work.run(flight.writer.abort)
                flight.finished.set()
                flight.notify()
            if not flight.interests and self._flights.get(flight.spec.key) is flight:
                del self._flights[flight.spec.key]

    def _release(self, consumer: Consumer) -> None:
        flight = consumer._flight
        flight.interests.discard(consumer)
        if not flight.interests:
            if flight.finished.is_set():
                if self._flights.get(flight.spec.key) is flight:
                    del self._flights[flight.spec.key]
            else:
                flight.cancellation.cancel()
                flight.notify()

    def _forget(self, consumer: Consumer) -> None:
        self._release(consumer)
        self._consumers.discard(consumer)
        if consumer.operation_id is not None:
            self._operations.pop(consumer.operation_id, None)

    async def cancel(self, operation_id: UUID) -> None:
        consumer = self._operations.get(operation_id)
        if consumer is not None:
            consumer.cancel()
            await consumer.aclose()

    async def cancel_session(self, session_id: UUID) -> None:
        consumers = [c for c in self._consumers if c.session_id == session_id]
        for consumer in consumers:
            consumer.cancel()
        await asyncio.gather(*(c.aclose() for c in consumers))

    async def aclose(self) -> None:
        self._closed = True
        consumers = list(self._consumers)
        for consumer in consumers:
            consumer.cancel()
        await asyncio.gather(*(c.aclose() for c in consumers))
        await asyncio.gather(*self._tasks)
