"""Prepared architecture HTTP adapter; no analysis or numerical operations on GET."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response

from .architecture_service import ArchitectureService
from .dependencies import get_architectures
from .session_routes import LifecycleRoute, Sessions

router = APIRouter(route_class=LifecycleRoute)


@router.get("/sessions/{session_id}/architecture", operation_id="getArchitecture")
async def get_architecture(
    session_id: UUID,
    sessions: Sessions,
    architectures: Annotated[ArchitectureService, Depends(get_architectures)],
) -> Response:
    session = sessions.require(session_id)
    body = await architectures.work.run(architectures.lookup, session.source)
    sessions.require(session_id)
    return Response(body, media_type="application/json", headers={"Cache-Control": "no-store"})
