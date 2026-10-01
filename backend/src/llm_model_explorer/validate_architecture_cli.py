"""Offline canonical validation of packaged or model-owned architecture."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from .architecture_analysis.core import AnalysisInput
from .architecture_analysis.model_defined import (
    ModelDefinedValidator,
    diagnostic_for_model_error,
)
from .architecture_analysis.model_defined_schema import MAX_DEFINITION_BYTES
from .architecture_analysis.validation import GraphError, model_finding
from .architecture_service import ArchitectureService, packaged_registry
from .model_files import ModelError
from .models import ModelCatalogue


def validate_directory(directory: Path, *, model_root: Path | None = None) -> dict[str, object]:
    """Use startup admission/validation, optionally within an explicit confinement root.

    Offline producers can validate nested staging packages whose shared shards
    remain within the publication root. The CLI defaults to the package parent.
    """
    try:
        directory = directory.resolve(strict=True)
        root = directory.parent if model_root is None else model_root.resolve(strict=True)
        entry = ModelCatalogue(root).inspect_directory(directory)
        source = entry.pin()
        inputs = AnalysisInput.from_source(
            source, tokenizer_available=entry.summary.tokenizer_available
        )
        try:
            raw = source.architecture_definition(max_bytes=MAX_DEFINITION_BYTES)
        except ModelError as exc:
            if exc.code != "unsupported_size":
                raise
            raise model_finding(
                "unsupported_size", "resource", "", "Definition exceeds the 8 MiB limit."
            ) from exc
        budget = ArchitectureService.response_budget(entry.summary.id)
        result = (
            ModelDefinedValidator.from_bytes(raw).validate(inputs, byte_limit=budget)
            if raw is not None
            else packaged_registry().analyze(inputs, byte_limit=budget)
        )
        if result.graph is None:
            return {
                "status": "invalid",
                "reason": result.reason,
                "diagnostics": [d.document() for d in result.diagnostics],
            }
        return {
            "status": "valid",
            "validation": "model-directory",
            "model_id": entry.summary.id,
            "coverage": result.graph.coverage,
            "scope": result.graph.scope,
            "graph_id": result.graph.graph_id,
            "diagnostics": [d.document() for d in result.graph.diagnostics],
        }
    except GraphError as exc:
        return {
            "status": "invalid",
            "reason": "unsupported_size" if exc.code == "unsupported_size" else "analysis_failed",
            "diagnostics": [diagnostic_for_model_error(exc).document()],
        }
    except Exception:
        # Admission failures and unexpected internal errors share a safe,
        # path-free CLI boundary, just as startup does.
        return {
            "status": "invalid",
            "reason": "analysis_failed",
            "diagnostics": [
                {
                    "code": "model_admission_failed",
                    "message": (
                        "Model directory could not be admitted from local config "
                        "and Safetensors metadata."
                    ),
                }
            ],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an architecture against its local checkpoint inventory."
    )
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit deterministic CI-friendly JSON.")
    arguments = parser.parse_args(argv)
    result = validate_directory(arguments.model_directory)
    if arguments.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    elif result["status"] == "valid":
        origin = "model-owned" if result["scope"] == "model_defined" else "packaged"
        print(f"Valid {origin} architecture ({result['coverage']}).")
    else:
        print("Invalid architecture:")
        for diagnostic in cast(list[dict[str, str]], result["diagnostics"]):
            print(f"  {diagnostic['code']}: {diagnostic['message']}")
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
