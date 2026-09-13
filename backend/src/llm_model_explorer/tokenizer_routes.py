"""JSON tokenization adapter; ordinary edits do not allocate long operations."""

import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .dependencies import get_blocking_work, get_sessions, get_tokenizers
from .execution import BlockingWork
from .session_routes import LifecycleRoute
from .sessions import SessionRegistry
from .tokenization import TokenizerService

router = APIRouter(route_class=LifecycleRoute)


class TokenizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str
    add_special_tokens: bool = True


@router.post("/sessions/{session_id}/tokenize")
async def tokenize(
    session_id: UUID,
    body: TokenizeRequest,
    request: Request,
    sessions: Annotated[SessionRegistry, Depends(get_sessions)],
    tokenizers: Annotated[TokenizerService, Depends(get_tokenizers)],
    work: Annotated[BlockingWork, Depends(get_blocking_work)],
) -> JSONResponse:
    session = sessions.require(session_id)

    # The body has already been consumed by FastAPI. Watch the remaining ASGI
    # channel so a disconnected editor releases interest in queued/running work.
    async def disconnected() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    task = asyncio.create_task(
        work.run(tokenizers.tokenize, session.source, body.text, body.add_special_tokens)
    )
    disconnect = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((task, disconnect), return_when=asyncio.FIRST_COMPLETED)
        if disconnect in done:
            raise asyncio.CancelledError
        result = task.result()
    finally:
        task.cancel()
        disconnect.cancel()
        await asyncio.gather(task, disconnect, return_exceptions=True)
    sessions.require(session_id)
    return JSONResponse(result, headers={"Cache-Control": "no-store"})
