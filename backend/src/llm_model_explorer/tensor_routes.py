"""Complete logical tensor delivery through the common operation/LMEX adapter."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from .dependencies import get_logical_tensors
from .materialization import LogicalTensorService
from .session_routes import LifecycleRoute, Sessions
from .streaming import LMEXResponse
from .tensor_analysis import subscribe_analysis

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


@router.get("/sessions/{session_id}/tensors/{tensor_id}/statistics")
async def tensor_statistics(
    session_id: UUID, tensor_id: str, sessions: Sessions, tensors: Tensors
) -> LMEXResponse:
    analysis, consumer = await subscribe_analysis(
        tensors, sessions, session_id, tensor_id, "tensor_statistics"
    )

    async def metadata() -> object:
        return await analysis.metadata(consumer)

    return LMEXResponse(consumer, metadata)


@router.get("/sessions/{session_id}/tensors/{tensor_id}/distributions")
async def tensor_distributions(
    session_id: UUID, tensor_id: str, sessions: Sessions, tensors: Tensors
) -> LMEXResponse:
    analysis, consumer = await subscribe_analysis(
        tensors, sessions, session_id, tensor_id, "tensor_distributions"
    )

    async def metadata() -> object:
        return await analysis.metadata(consumer)

    return LMEXResponse(consumer, metadata)
