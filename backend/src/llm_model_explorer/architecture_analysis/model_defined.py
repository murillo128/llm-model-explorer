"""Validate checkpoint-owned JSON and bind its declared graph to a pinned inventory.

This module is not a numerical model loader or a source-code verifier. Only
metadata enters it; every runtime ID and storage binding is assigned here.
"""

from __future__ import annotations

import json
from typing import Any

from . import records as r
from .core import AnalysisInput, AnalysisResult, GraphBuilder, Producer, unavailable
from .model_defined_schema import MAX_DEFINITION_BYTES, ModelDefinition
from .model_defined_templates import bind_templates
from .validation import MAX_BYTES, GraphError, preflight, require


class ModelDefinedProducer(Producer):
    def provenance(self) -> list[r.ArchitectureProvenance]:
        return [
            r.ArchitectureProvenance(
                kind="description",
                source="model-supplied architecture.json",
                revision=self.source_revision,
                rule="Author-declared structure; validated against graph and inventory contracts, "
                "not verified against the model implementation.",
            )
        ]


def producer_for(definition: ModelDefinition) -> Producer:
    return ModelDefinedProducer("model-defined-json", "2", definition.architecture_revision)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "Duplicate model-definition key.")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise GraphError("invalid_graph", "Non-finite JSON constants are not supported.")


def parse_definition(raw: bytes) -> ModelDefinition:
    if len(raw) > MAX_DEFINITION_BYTES:
        raise GraphError("unsupported_size", "Model architecture definition exceeds its limit.")
    try:
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        preflight(document, MAX_DEFINITION_BYTES)
        return ModelDefinition.model_validate(document)
    except GraphError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise GraphError(
            "invalid_graph", "Invalid or unsupported model architecture definition."
        ) from exc


def build_definition(
    definition: ModelDefinition, inputs: AnalysisInput, builder: GraphBuilder
) -> None:
    producer = builder.producer
    provenance = producer.provenance()

    def local(kind: str, key: str) -> str:
        return builder.record_id(kind, "declared:" + key)

    # File-local identifiers are never accepted as global IDs. Keep a separate
    # namespace for the mandatory origin notice, even if the author uses its name.
    builder.add_node(
        r.ArchitectureLeafNode(
            id=builder.record_id("node", "model-definition-origin"),
            kind="context",
            label="Model-supplied definition",
            description="This graph was supplied with the checkpoint. Validation checks its "
            "structure and parameter bindings; it does not establish equivalence to forward().",
            ports=[],
            parameter_ids=[],
            references=[],
            attributes=[
                r.ArchitectureAttribute(name=name, value=value, provenance=provenance)
                for name, value in (
                    ("definition_origin", "model"),
                    ("semantic_verification", "not_verified"),
                    ("name", definition.name),
                    ("architecture_revision", definition.architecture_revision),
                    ("declared_scope", definition.scope),
                )
            ],
            provenance=provenance,
        )
    )
    for symbol in definition.symbols:
        builder.add_symbol(symbol.name, symbol.meaning)
    for parameter in definition.parameters:
        storage = inputs.bindings.physical.get(parameter.name)
        require(
            storage is None or storage.dtype in {"F32", "F16", "BF16"},
            "Model-definition v1 parameters require native floating-point storage.",
        )
        # GraphBuilder validates complete native geometry, resolves numeric IDs
        # from the admitted inventory, and marks missing storage explicitly partial.
        builder.native_parameter(
            "declared:" + parameter.id,
            parameter.name,
            parameter.shape,
            provenance,
        )
    for node in definition.nodes:
        record = node.document()
        record["id"] = local("node", node.id)
        record["parameter_ids"] = [local("parameter", key) for key in node.parameter_ids]
        record["references"] = [
            {"kind": "parameter", "parameter_id": local("parameter", ref.parameter_id)}
            if isinstance(ref, r.ArchitectureParameterReference)
            else ref.document()
            for ref in node.references
        ]
        record["ports"] = [port.document() for port in node.ports]
        record["attributes"] = [
            r.ArchitectureAttribute(
                name=attribute.name, value=attribute.value, provenance=provenance
            ).document()
            for attribute in node.attributes
        ]
        record["provenance"] = [
            item.document()
            for item in [
                *provenance,
                r.ArchitectureProvenance(
                    kind="description",
                    source=node.id,
                    revision=definition.architecture_revision,
                    rule="Semantic source key supplied by the model author; not source-verified.",
                ),
            ]
        ]
        if node.parent_id is not None:
            record["parent_id"] = local("node", node.parent_id)
        if node.kind == "group":
            record["children"] = [local("node", child) for child in node.children]
            builder.add_node(r.ArchitectureGroupNode.model_validate(record), semantic_key=node.id)
        else:
            builder.add_node(r.ArchitectureLeafNode.model_validate(record), semantic_key=node.id)
    for edge in definition.edges:
        record = edge.document()
        record["id"] = local("edge", edge.id)
        record["kind"] = edge.kind
        record["source"] = {
            "node_id": local("node", edge.source.node_id),
            "port_id": edge.source.port_id,
        }
        record["target"] = {
            "node_id": local("node", edge.target.node_id),
            "port_id": edge.target.port_id,
        }
        record["provenance"] = [item.document() for item in provenance]
        builder.add_edge(r.ArchitectureEdge.model_validate(record))
    for repetition in definition.repetitions:
        builder.add_repetition(
            r.ArchitectureRepetition(
                id=local("repetition", repetition.id),
                parent_id=local("node", repetition.parent_id),
                label=repetition.label,
                instances=[
                    r.ArchitectureRepetitionInstance(
                        node_id=local("node", instance.node_id),
                        index=instance.index,
                        variant=instance.variant,
                    )
                    for instance in repetition.instances
                ],
            )
        )
    if definition.coverage == "partial":
        builder.diagnose(
            r.ArchitectureDiagnostic(
                code="author_declared_partial",
                message=definition.incomplete_reason or "The author declared incomplete coverage.",
            )
        )


def analyze_definition(
    definition: ModelDefinition,
    inputs: AnalysisInput,
    *,
    byte_limit: int = MAX_BYTES,
) -> AnalysisResult:
    try:
        builder = GraphBuilder(
            inputs, producer_for(definition), "model_defined", byte_limit=byte_limit
        )
        build_definition(definition, inputs, builder)
        graph = bind_templates(definition, builder.finish(), builder)
        return AnalysisResult(graph, None)
    except GraphError as exc:
        if exc.code == "unsupported_size":
            return unavailable(
                "unsupported_size", "unsupported_size", "Model-defined graph exceeds its limit."
            )
        return unavailable(
            "analysis_failed",
            "invalid_model_definition",
            "Model-defined graph or parameter bindings are invalid.",
        )
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return unavailable(
            "analysis_failed",
            "invalid_model_definition",
            "Model-defined graph could not be validated.",
        )
