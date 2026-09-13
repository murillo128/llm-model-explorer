"""Typed input embedding lookup through the common LMEX response adapter."""

from uuid import UUID

from fastapi import APIRouter

from .embeddings import subscribe_embeddings
from .session_routes import LifecycleRoute, Sessions
from .stream_metadata import Control, Size
from .streaming import LMEXResponse

router = APIRouter(route_class=LifecycleRoute)


class InputEmbeddingsRequest(Control):
    token_ids: list[Size]


@router.post("/sessions/{session_id}/embeddings")
async def input_embeddings(
    session_id: UUID, body: InputEmbeddingsRequest, sessions: Sessions
) -> LMEXResponse:
    result, consumer = await subscribe_embeddings(sessions, session_id, tuple(body.token_ids))

    async def metadata() -> object:
        await sessions.work.run(result.source.check_unchanged)
        return result.metadata()

    return LMEXResponse(consumer, metadata)
