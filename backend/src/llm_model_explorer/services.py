"""Concrete application composition slots for downstream domain modules."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .execution import BlockingWork
from .settings import Settings


@dataclass
class Services:
    blocking_work: BlockingWork
    # Replace object with the concrete domain type when that module is implemented.
    # None means unconfigured; it never stands in for successful product data.
    catalogue: object | None = None
    sessions: object | None = None
    artifacts: object | None = None
    operation_delivery: object | None = None


@asynccontextmanager
async def open_services(settings: Settings) -> AsyncIterator[Services]:
    """Own resources for one application lifespan, with no model access."""
    work = BlockingWork()
    try:
        yield Services(blocking_work=work)
    finally:
        await work.aclose()
