"""The catalogue HTTP adapter; blocking discovery never runs on the event loop."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from .dependencies import get_blocking_work, get_catalogue
from .execution import BlockingWork
from .model_files import ModelError
from .models import ModelCatalogue

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models")
async def list_models(
    catalogue: Annotated[ModelCatalogue, Depends(get_catalogue)],
    work: Annotated[BlockingWork, Depends(get_blocking_work)],
) -> JSONResponse:
    try:
        listing = await work.run(catalogue.list_catalogue)
        return JSONResponse(
            {
                "models": [
                    model.model_dump(mode="json", exclude_none=True) for model in listing.models
                ],
                "diagnostics": [
                    diagnostic.model_dump(mode="json", exclude_none=True)
                    for diagnostic in listing.diagnostics
                ],
            },
            headers={"Cache-Control": "no-store"},
        )
    except ModelError as exc:
        # /models has no session and therefore no model_content_changed response.
        code = "validation_error" if exc.code == "model_content_changed" else exc.code
        status = 422 if exc.code == "model_content_changed" else exc.status
        return JSONResponse(
            {"code": code, "message": str(exc)},
            status_code=status,
            headers={"Cache-Control": "no-store"},
        )
    except MemoryError:
        return JSONResponse(
            {"code": "resource_exhausted", "message": "Insufficient memory to list models."},
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    except Exception:
        logger.exception("Local model discovery failed")
        return JSONResponse(
            {"code": "internal_error", "message": "Unable to discover local models."},
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )
