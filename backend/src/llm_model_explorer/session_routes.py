"""HTTP lifecycle adapters; numerical producers and binary encoding live elsewhere."""

import logging
from collections.abc import Callable, Coroutine
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from .dependencies import get_operation_delivery, get_sessions
from .model_files import ModelError
from .operations import OperationRuntime
from .sessions import SessionRegistry

logger = logging.getLogger(__name__)


class LifecycleRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                malformed = any(error["type"] == "json_invalid" for error in exc.errors())
                code, status = ("malformed_json", 400) if malformed else ("validation_error", 422)
                message = (
                    "Malformed JSON." if malformed else "Invalid request fields or identifier."
                )
            except ModelError as exc:
                code, status, message = exc.code, exc.status, str(exc)
            except MemoryError:
                code, status, message = "resource_exhausted", 503, "Insufficient runtime memory."
            except Exception:
                logger.exception("Session/operation request failed")
                code, status, message = "internal_error", 500, "Unable to complete request."
            return JSONResponse(
                {"code": code, "message": message},
                status_code=status,
                headers={"Cache-Control": "no-store"},
            )

        return handle


router = APIRouter(route_class=LifecycleRoute)
Sessions = Annotated[SessionRegistry, Depends(get_sessions)]
Operations = Annotated[OperationRuntime, Depends(get_operation_delivery)]


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model_id: str = Field(min_length=1)


@router.post("/sessions")
async def create_session(body: CreateSession, sessions: Sessions) -> JSONResponse:
    session = await sessions.create(body.model_id)
    return JSONResponse(
        session.descriptor(), status_code=201, headers={"Cache-Control": "no-store"}
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: UUID, sessions: Sessions) -> JSONResponse:
    session = await sessions.get(session_id)
    return JSONResponse(session.descriptor(), headers={"Cache-Control": "no-store"})


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: UUID, sessions: Sessions) -> Response:
    await sessions.delete(session_id)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


@router.get("/sessions/{session_id}/tensors")
async def list_tensors(session_id: UUID, sessions: Sessions) -> JSONResponse:
    tensors = await sessions.tensors(session_id)
    return JSONResponse(
        {"tensors": [tensor.model_dump(mode="json") for tensor in tensors]},
        headers={"Cache-Control": "no-store"},
    )


@router.delete("/operations/{operation_id}")
async def cancel_operation(operation_id: UUID, operations: Operations) -> Response:
    await operations.cancel(operation_id)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
