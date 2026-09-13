"""Complete logical tensor delivery through the common operation/LMEX adapter."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from .dependencies import get_logical_tensors
from .materialization import LogicalTensorService
from .session_routes import LifecycleRoute, Sessions
from .streaming import LMEXResponse

router = APIRouter(route_class=LifecycleRoute)
Tensors = Annotated[LogicalTensorService, Depends(get_logical_tensors)]


@router.get("/sessions/{session_id}/tensors/{tensor_id}/data")
async def tensor_data(
    session_id: UUID, tensor_id: str, sessions: Sessions, tensors: Tensors
) -> LMEXResponse:
    tensor, consumer = await tensors.subscribe(sessions, session_id, tensor_id)

    async def metadata() -> object:
        return tensor.metadata()

    return LMEXResponse(consumer, metadata)
