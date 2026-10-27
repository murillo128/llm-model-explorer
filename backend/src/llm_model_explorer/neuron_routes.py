"""Cheap execution admission. Numerical operations are separate explicit capabilities."""

import json
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .dependencies import get_services, get_sessions, get_settings
from .model_files import ModelError
from .neuron_execution import Unavailable, admit
from .neuron_execution.activations import subscribe_activations
from .session_routes import LifecycleRoute
from .stream_metadata import Control, ModelRevision, NeuronId, Size
from .streaming import LMEXResponse

router = APIRouter(route_class=LifecycleRoute)


class NeuronActivationsRequest(Control):
    model_revision: ModelRevision
    module_id: NeuronId
    neuron_indices: list[Size]
    token_ids: list[Size]


@router.post("/sessions/{session_id}/neuron-activations")
async def neuron_activations(
    session_id: UUID, body: NeuronActivationsRequest, request: Request
) -> LMEXResponse:
    runtime = get_services(request).neurons
    if runtime is None:
        raise ModelError("unsupported_representation", "Neuron runtime is not configured.")
    sessions = get_sessions(request)
    result, consumer = await subscribe_activations(
        sessions,
        runtime,
        get_settings(request).model_root,
        session_id,
        body.model_revision,
        body.module_id,
        tuple(body.neuron_indices),
        tuple(body.token_ids),
    )

    async def metadata() -> object:
        await sessions.work.run(result.selection.model.source.check_unchanged)
        return result.metadata

    return LMEXResponse(consumer, metadata)


@router.get("/sessions/{session_id}/neuron-capabilities")
async def neuron_capabilities(session_id: UUID, request: Request) -> JSONResponse:
    sessions = get_sessions(request)
    settings = get_settings(request)
    session = await sessions.get(session_id)

    def inspect() -> dict[str, object]:
        result: dict[str, object] = dict(
            session_id=str(session_id),
            model_id=session.source.model_id,
            model_revision=session.source.model_revision,
            execution=None,
            limits=None,
            modules=[],
        )
        try:
            if get_services(request).neurons is None:
                raise Unavailable(
                    "runtime_unavailable",
                    "Neuron runtime is not configured.",
                    "Start the backend with its execution services.",
                )
            admitted = admit(session.source, settings.model_root, settings.device)
            feature: dict[str, object] = {"status": "available"}
            result.update(
                execution=admitted.execution, limits=admitted.limits, modules=list(admitted.modules)
            )
        except Unavailable as exc:
            feature = dict(exc.feature)
        result.update(observation=feature, intervention=feature, generation=feature)
        session.source.check_unchanged()
        if len(json.dumps(result).encode("utf-8")) > 1024 * 1024:
            raise ModelError("unsupported_size", "Neuron capabilities exceed control limits.")
        return result

    result = await sessions.work.run(inspect)
    sessions.require(session_id)
    return JSONResponse(result, headers={"Cache-Control": "no-store"})
