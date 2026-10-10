"""Subscription-owned, shared metadata observation; no model analysis or replay log."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import TypeAdapter

from .execution import BlockingWork
from .model_event_records import ModelObservationError, ModelState
from .model_files import ModelError
from .models import ModelCatalogue

logger = logging.getLogger(__name__)
MAX_EVENT_BYTES = 64 * 1024
MAX_SUBSCRIBERS = 256
MAX_SEQUENCE = 2**53 - 1
_STATE: TypeAdapter[ModelState] = TypeAdapter(ModelState)
TICK_SECONDS = 1.0
HEARTBEAT_SECONDS = 15.0


@dataclass(eq=False)
class ModelSubscription:
    model_id: str
    queue: asyncio.Queue[bytes | None] = field(default_factory=lambda: asyncio.Queue(maxsize=1))
    state: tuple[str, str | None] | None = None

    def replace(self, frame: bytes | None) -> None:
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(frame)


class ModelObserver:
    def __init__(self, catalogue: ModelCatalogue, work: BlockingWork) -> None:
        self.catalogue = catalogue
        self.work = work
        self.epoch = str(uuid4())
        self._sequence = 0
        self._subscribers: set[ModelSubscription] = set()
        self._lifetime = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._closed = False

    async def subscribe(self, model_id: str) -> ModelSubscription:
        # JSON escaping preserves every admitted logical ID (including spaces),
        # while leaving ample room for bounded fields and SSE framing.
        if (
            len(model_id) > MAX_EVENT_BYTES
            or len(json.dumps(model_id, ensure_ascii=True)) > MAX_EVENT_BYTES - 1024
        ):
            raise ModelError("unsupported_size", "Model identity exceeds the event size limit.")
        async with self._lifetime:
            if self._closed or len(self._subscribers) >= MAX_SUBSCRIBERS:
                raise ModelError("resource_exhausted", "Model observation capacity exhausted.", 503)
            subscription = ModelSubscription(model_id)
            # Registration precedes observation. New subscribers never receive a
            # cached initial state without checking the filesystem again.
            self._subscribers.add(subscription)
            if self._task is None:
                self._stop = asyncio.Event()
                self._task = asyncio.create_task(self._run())
            return subscription

    async def unsubscribe(self, subscription: ModelSubscription) -> None:
        async with self._lifetime:
            self._subscribers.discard(subscription)
            if not self._subscribers:
                await self._settle()

    async def _settle(self) -> None:
        self._stop.set()
        if self._task is not None:
            # Never cancel a worker future: the lock and its filesystem work must
            # settle before another observer starts or the pool is disposed.
            cancelled = False
            while not self._task.done():
                try:
                    await asyncio.shield(self._task)
                except asyncio.CancelledError:
                    cancelled = True
            self._task.result()
            self._task = None
            if cancelled:
                raise asyncio.CancelledError()

    def request_stop(self) -> None:
        """End streams before an HTTP server waits for its active requests to drain."""
        self._closed = True
        self._stop.set()
        for subscription in self._subscribers:
            subscription.replace(None)

    async def aclose(self) -> None:
        self.request_stop()
        async with self._lifetime:
            try:
                await self._settle()
            finally:
                self._subscribers.clear()

    async def _wait_tick(self) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), TICK_SECONDS)
        except TimeoutError:
            pass

    def _publish(self, subscription: ModelSubscription, status: str, revision: str | None) -> None:
        if self._closed:
            return
        state = (status, revision)
        if subscription.state == state:
            return
        if self._sequence == MAX_SEQUENCE:
            # Practically unreachable, but never emit an unsafe integer.
            self.epoch, self._sequence = str(uuid4()), 0
        self._sequence += 1
        data: dict[str, object] = {
            "epoch": self.epoch,
            "sequence": self._sequence,
            "model_id": subscription.model_id,
        }
        event = "model-state"
        if status == "error":
            event = "observation-error"
            data["code"] = "observation_failed"
        else:
            data.update(status=status, model_revision=revision)
        record = (
            ModelObservationError.model_validate(data)
            if status == "error"
            else _STATE.validate_python(data)
        )
        frame = (
            f"id: {self.epoch}:{self._sequence}\nevent: {event}\n"
            f"data: {json.dumps(record.document(), ensure_ascii=True, separators=(',', ':'))}\n\n"
        ).encode()
        if len(frame) > MAX_EVENT_BYTES:
            raise ValueError("Model event exceeds the supported size")
        subscription.state = state
        subscription.replace(frame)

    async def _run(self) -> None:
        previous = None
        failure_generation = None
        retry_ticks = 0
        delay_ticks = 1
        while not self._stop.is_set():
            captured = False
            try:
                generation = await self.work.run(self.catalogue.metadata_generation)
                captured = True
                if generation != previous:
                    previous = generation
                    retry_ticks, delay_ticks = 0, 1
                elif retry_ticks:
                    retry_ticks -= 1
                else:
                    try:
                        revisions = await self.work.run(self.catalogue.observe, generation)
                    except Exception:
                        failure_generation = generation
                        retry_ticks = delay_ticks
                        delay_ticks = min(delay_ticks * 2, 32)
                        raise
                    if not self._stop.is_set():
                        for subscription in self._subscribers:
                            revision = revisions.get(subscription.model_id)
                            self._publish(
                                subscription,
                                "present" if revision is not None else "unavailable",
                                revision,
                            )
                    failure_generation = None
                    delay_ticks = 1
                # A subscriber joining during backoff also gets the safe error.
                if generation == failure_generation:
                    for subscription in self._subscribers:
                        self._publish(subscription, "error", None)
            except Exception:
                if not captured:
                    previous = None
                logger.warning("Unable to observe the local model catalogue", exc_info=True)
                for subscription in self._subscribers:
                    self._publish(subscription, "error", None)
            await self._wait_tick()
