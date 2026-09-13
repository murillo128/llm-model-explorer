"""Concrete application composition slots for downstream domain modules."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .artifacts import ArtifactStore
from .execution import BlockingWork
from .models import ModelCatalogue
from .operations import OperationRuntime
from .sessions import SessionRegistry
from .settings import Settings


@dataclass
class Services:
    blocking_work: BlockingWork
    # Replace object with the concrete domain type when that module is implemented.
    # None means unconfigured; it never stands in for successful product data.
    catalogue: ModelCatalogue | None = None
    sessions: SessionRegistry | None = None
    artifacts: ArtifactStore | None = None
    operation_delivery: OperationRuntime | None = None


@asynccontextmanager
async def open_services(settings: Settings) -> AsyncIterator[Services]:
    """Own resources for one application lifespan, with no model access."""
    work = BlockingWork()
    sessions = None
    try:
        artifacts = await work.run(
            ArtifactStore, settings.cache_dir, model_root=settings.model_root
        )
        catalogue = ModelCatalogue(settings.model_root)
        operations = OperationRuntime(artifacts, work, settings.device)
        sessions = SessionRegistry(catalogue, work, operations)
        yield Services(
            blocking_work=work,
            catalogue=catalogue,
            artifacts=artifacts,
            sessions=sessions,
            operation_delivery=operations,
        )
    finally:
        try:
            if sessions is not None:
                await sessions.aclose()
        finally:
            await work.aclose()
