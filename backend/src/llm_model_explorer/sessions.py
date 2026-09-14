"""In-memory, immutable model bindings; all registry access uses the app event loop."""

import json
from dataclasses import dataclass
from uuid import UUID, uuid4

from .artifacts import ArtifactSpec
from .execution import BlockingWork
from .model_files import ModelError
from .models import ModelCatalogue
from .operations import Consumer, OperationRuntime, Producer
from .tensor_source import ModelSource, TensorDescriptor


@dataclass(frozen=True)
class Session:
    id: UUID
    source: ModelSource

    def descriptor(self) -> dict[str, str]:
        return {"id": str(self.id), "model_id": self.source.model_id}


class SessionRegistry:
    def __init__(
        self, catalogue: ModelCatalogue, work: BlockingWork, operations: OperationRuntime
    ) -> None:
        self.catalogue = catalogue
        self.work = work
        self.operations = operations
        self._sessions: dict[UUID, Session] = {}

    async def create(self, model_id: str) -> Session:
        try:
            source = await self.work.run(self.catalogue.pin, model_id)
        except ModelError as exc:
            if exc.code == "model_content_changed":
                raise ModelError(
                    "validation_error", "Model changed during session creation.", 422
                ) from exc
            raise
        session = Session(uuid4(), source)
        self._sessions[session.id] = session
        return session

    def require(self, session_id: UUID) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise ModelError("session_not_found", "Unknown session.", 404)
        return session

    async def get(self, session_id: UUID) -> Session:
        session = self.require(session_id)
        await self.work.run(session.source.check_unchanged)
        return self.require(session_id)

    async def tensors(self, session_id: UUID) -> tuple[TensorDescriptor, ...]:
        session = self.require(session_id)
        tensors = await self.work.run(session.source.tensors)
        self.require(session_id)
        return tensors

    async def inventory(self, session_id: UUID) -> dict[str, object]:
        session = self.require(session_id)
        inventory = await self.work.run(session.source.inventory)
        self.require(session_id)
        return inventory

    async def tensor(self, session_id: UUID, tensor_id: str) -> TensorDescriptor:
        for tensor in await self.tensors(session_id):
            if tensor.id == tensor_id:
                return tensor
        raise ModelError("tensor_not_found", "Unknown tensor.", 404)

    async def delete(self, session_id: UUID) -> None:
        self.require(session_id)
        del self._sessions[session_id]
        await self.operations.cancel_session(session_id)

    async def subscribe(self, session_id: UUID, spec: ArtifactSpec, producer: Producer) -> Consumer:
        session = await self.get(session_id)
        # Artifact keys must belong to the exact model-content snapshot.
        if json.loads(spec.canonical)["model_fingerprint"] != session.source.fingerprint:
            raise ValueError("artifact does not match the session model snapshot")
        consumer = await self.operations.subscribe(session_id, spec, producer)
        if session_id not in self._sessions:
            await consumer.aclose()
            self.require(session_id)
        return consumer

    async def aclose(self) -> None:
        self._sessions.clear()
        await self.operations.aclose()
