"""Validate checkpoint-owned JSON and bind its declared graph to a pinned inventory.

This module is not a numerical model loader or a source-code verifier. Only
metadata enters it; every runtime ID and storage binding is assigned here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from . import records as r
from .core import AnalysisInput, AnalysisResult, GraphBuilder, Producer, unavailable
from .model_defined_schema import MAX_DEFINITION_BYTES, DefinitionGroup, ModelDefinition
from .model_defined_templates import bind_templates
from .validation import (
    MAX_BYTES,
    GraphError,
    constants,
    preflight,
    product,
    require,
)
from .validation import (
    model_finding as _finding,
)


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
    return ModelDefinedProducer("model-defined-json", "3", definition.architecture_revision)


def _pointer(parts: tuple[str | int, ...]) -> str:
    # Pydantic inserts the discriminator branch after a union's array index.
    clean: list[str | int] = []
    for index, part in enumerate(parts):
        if (
            index >= 2
            and parts[index - 2] == "nodes"
            and isinstance(parts[index - 1], int)
            and index < len(parts) - 1
            and part in ("operation", "group", "input", "output", "context", "state")
        ):
            continue
        if isinstance(part, int) and 0 <= part < 1_000_000:
            clean.append(part)
        elif isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", part):
            clean.append(part)
        else:
            break
    return "#" + "".join(f"/{part}" for part in clean) if clean else ""


def _schema_error(exc: ValidationError) -> GraphError:
    error = exc.errors(include_url=False, include_context=False, include_input=False)[0]
    parts = tuple(part for part in error["loc"] if isinstance(part, (str, int)))
    kind = error["type"]
    if parts and parts[0] == "schema_version":
        return _finding(
            "schema_unsupported_version",
            "schema",
            "#/schema_version",
            "Unsupported schema version; expected integer 1.",
        )
    if kind == "extra_forbidden":
        return _finding(
            "schema_unknown_field",
            "schema",
            _pointer(parts),
            "Field is not allowed by the closed schema.",
        )
    if kind == "value_error" and parts and parts[-1] == "operation":
        return _finding(
            "schema_missing_operation",
            "schema",
            _pointer(parts),
            "Operation nodes require an operation name.",
        )
    if kind == "missing":
        return _finding(
            "schema_missing_field", "schema", _pointer(parts), "Required field is missing."
        )
    if kind == "value_error" and parts == ():
        return _finding(
            "schema_invalid_definition",
            "schema",
            "",
            "Definition coverage and incomplete reason disagree.",
        )
    return _finding(
        "schema_invalid_field",
        "schema",
        _pointer(parts),
        "Field does not satisfy the model-owned schema.",
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            safe = (
                json.dumps(key) if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key) else "a field"
            )
            raise _finding("json_duplicate_key", "json", "", f"Duplicate key {safe}.")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise _finding("json_non_finite", "json", "", "Non-finite JSON constants are not supported.")


def parse_definition(raw: bytes) -> ModelDefinition:
    if len(raw) > MAX_DEFINITION_BYTES:
        raise _finding("unsupported_size", "resource", "", "Definition exceeds the 8 MiB limit.")
    try:
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        preflight(document, MAX_DEFINITION_BYTES)
        return ModelDefinition.model_validate(document)
    except GraphError:
        raise
    except UnicodeDecodeError as exc:
        raise _finding("json_invalid_utf8", "json", "", "Input is not valid UTF-8.") from exc
    except json.JSONDecodeError as exc:
        raise _finding("json_malformed", "json", "", "Malformed JSON document.") from exc
    except ValidationError as exc:
        raise _schema_error(exc) from exc
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise _finding(
            "schema_invalid_definition", "schema", "", "Definition could not be validated."
        ) from exc


@dataclass(frozen=True)
class ModelDefinedValidator:
    """The shared startup and authoring boundary for a supplied sidecar."""

    definition: ModelDefinition

    @classmethod
    def from_bytes(cls, raw: bytes) -> ModelDefinedValidator:
        return cls(parse_definition(raw))

    def validate(self, inputs: AnalysisInput, *, byte_limit: int = MAX_BYTES) -> AnalysisResult:
        return analyze_definition(self.definition, inputs, byte_limit=byte_limit)


def _check_bindings(definition: ModelDefinition, inputs: AnalysisInput) -> None:
    for index, parameter in enumerate(definition.parameters):
        location = f"#/parameters/{index}"
        storage = inputs.bindings.physical.get(parameter.name)
        if storage is None:
            raise _finding(
                "binding_missing_tensor",
                "binding",
                location,
                "Declared checkpoint tensor is absent.",
            )
        if storage.dtype not in {"F32", "F16", "BF16"}:
            raise _finding(
                "binding_unsupported_dtype",
                "binding",
                location,
                "Version 1 requires native floating-point storage.",
            )
        shape = constants(parameter.shape)
        if shape is None or tuple(storage.shape) != shape:
            expected = list(shape) if shape is not None else None
            observed = storage.shape
            # Shape ranks are bounded by admission; never include a tensor name or payload.
            detail = (
                f"Declared shape {json.dumps(expected)} disagrees with checkpoint tensor "
                f"shape {json.dumps(observed)}."
                if expected is not None and len(expected) <= 16 and len(observed) <= 16
                else "Declared shape disagrees with checkpoint tensor shape."
            )
            raise _finding("binding_shape_mismatch", "binding", location, detail)


def _graph_location(message: str, definition: ModelDefinition) -> str:
    nodes = {node.id: node for node in definition.nodes}
    if message == "Duplicate record identity.":
        for collection in ("nodes", "parameters", "edges", "repetitions", "templates"):
            seen: set[str] = set()
            for index, record in enumerate(getattr(definition, collection)):
                if record.id in seen:
                    return f"#/{collection}/{index}/id"
                seen.add(record.id)
        for node_index, node in enumerate(definition.nodes):
            seen = set()
            for port_index, port in enumerate(node.ports):
                if port.id in seen:
                    return f"#/nodes/{node_index}/ports/{port_index}/id"
                seen.add(port.id)
        for repetition_index, repetition in enumerate(definition.repetitions):
            seen = set()
            for instance_index, instance in enumerate(repetition.instances):
                if instance.node_id in seen:
                    return f"#/repetitions/{repetition_index}/instances/{instance_index}/node_id"
                seen.add(instance.node_id)
    if message == "Duplicate shape symbol.":
        seen = set()
        for index, symbol in enumerate(definition.symbols):
            if symbol.name in seen:
                return f"#/symbols/{index}/name"
            seen.add(symbol.name)
    if message in {
        "Parent must be a group.",
        "Child and parent disagree.",
        "Parent and child disagree.",
        "Cyclic record linkage.",
        "Duplicate child.",
    }:
        for index, node in enumerate(definition.nodes):
            parent = nodes.get(node.parent_id) if node.parent_id else None
            if node.parent_id and (
                not isinstance(parent, DefinitionGroup) or node.id not in parent.children
            ):
                return f"#/nodes/{index}/parent_id"
            if isinstance(node, DefinitionGroup) and (
                len(set(node.children)) != len(node.children)
                or any(
                    child not in nodes or nodes[child].parent_id != node.id
                    for child in node.children
                )
            ):
                return f"#/nodes/{index}/children"
        if message == "Cyclic record linkage.":
            checked: set[str] = set()
            for index, node in enumerate(definition.nodes):
                if node.id in checked:
                    continue
                trail: set[str] = set()
                current = node
                while current.parent_id and current.parent_id in nodes:
                    if current.id in trail:
                        return f"#/nodes/{index}/parent_id"
                    if current.id in checked:
                        break
                    trail.add(current.id)
                    current = nodes[current.parent_id]
                checked.update(trail)
    if message in {"Unknown node parameter.", "Unknown parameter resource."}:
        ids = {parameter.id for parameter in definition.parameters}
        for index, node in enumerate(definition.nodes):
            if any(key not in ids for key in node.parameter_ids):
                return f"#/nodes/{index}/parameter_ids"
            if any(
                isinstance(ref, r.ArchitectureParameterReference) and ref.parameter_id not in ids
                for ref in node.references
            ):
                return f"#/nodes/{index}/references"
    if message == "Tokenizer capability is unavailable.":
        for node_index, node in enumerate(definition.nodes):
            for reference_index, reference in enumerate(node.references):
                if isinstance(reference, r.ArchitectureTokenizerReference):
                    return f"#/nodes/{node_index}/references/{reference_index}"
    if message in {"Unsafe dimension.", "Unsafe dimension product."}:
        for node_index, node in enumerate(definition.nodes):
            for port_index, port in enumerate(node.ports):
                shape = constants(port.shape)
                if shape is None:
                    continue
                try:
                    product(shape)
                except GraphError as exc:
                    if str(exc) == message:
                        return f"#/nodes/{node_index}/ports/{port_index}/shape"
        for parameter_index, parameter in enumerate(definition.parameters):
            shape = constants(parameter.shape)
            if shape is None:
                continue
            try:
                product(shape)
            except GraphError as exc:
                if str(exc) == message:
                    return f"#/parameters/{parameter_index}/shape"
    if message in {"Undeclared shape symbol.", "Undeclared expression symbol."}:
        declared = {symbol.name for symbol in definition.symbols}
        for index, node in enumerate(definition.nodes):
            for port_index, port in enumerate(node.ports):
                if port.shape and any(
                    (isinstance(dim, r.ArchitectureSymbolDimension) and dim.name not in declared)
                    or (
                        isinstance(dim, r.ArchitectureExpressionDimension)
                        and any(name not in declared for name in dim.symbols)
                    )
                    for dim in port.shape
                ):
                    return f"#/nodes/{index}/ports/{port_index}/shape"
    if message in {
        "Unknown edge endpoint.",
        "Incompatible port directions.",
        "Edge bypasses a group boundary.",
        "Known connected dimensions disagree without a diagnostic.",
    }:
        ports = {node.id: {port.id: port for port in node.ports} for node in definition.nodes}
        for index, edge in enumerate(definition.edges):
            sn = nodes.get(edge.source.node_id)
            tn = nodes.get(edge.target.node_id)
            sp = ports.get(edge.source.node_id, {}).get(edge.source.port_id)
            tp = ports.get(edge.target.node_id, {}).get(edge.target.port_id)
            if message == "Unknown edge endpoint." and (sp is None or tp is None):
                return f"#/edges/{index}"
            if sn is None or tn is None or sp is None or tp is None:
                continue
            compatible = (
                (sp.direction == "output" and tp.direction == "input")
                or (
                    sn.kind == "group"
                    and tn.parent_id == sn.id
                    and sp.direction == tp.direction == "input"
                )
                or (
                    tn.kind == "group"
                    and sn.parent_id == tn.id
                    and sp.direction == tp.direction == "output"
                )
            )
            if message == "Incompatible port directions." and not compatible:
                return f"#/edges/{index}"
            if message == "Edge bypasses a group boundary." and not (
                sn.parent_id == tn.parent_id or tn.parent_id == sn.id or sn.parent_id == tn.id
            ):
                return f"#/edges/{index}"
            if (
                message == "Known connected dimensions disagree without a diagnostic."
                and sp.shape
                and tp.shape
                and (
                    len(sp.shape) != len(tp.shape)
                    or any(
                        isinstance(a, r.ArchitectureConstantDimension)
                        and isinstance(b, r.ArchitectureConstantDimension)
                        and a.value != b.value
                        for a, b in zip(sp.shape, tp.shape, strict=False)
                    )
                )
            ):
                return f"#/edges/{index}"
    if message.startswith("Repetition"):
        for index, rep in enumerate(definition.repetitions):
            parent = nodes.get(rep.parent_id)
            if message == "Repetition parent must be a group." and not isinstance(
                parent, DefinitionGroup
            ):
                return f"#/repetitions/{index}/parent_id"
            indices = [instance.index for instance in rep.instances]
            if message == "Repetition indices must be strictly ordered." and indices != sorted(
                set(indices)
            ):
                return f"#/repetitions/{index}/instances"
            if message == "Repetition instance must be a direct child group." and any(
                not isinstance(nodes.get(instance.node_id), DefinitionGroup)
                or nodes[instance.node_id].parent_id != rep.parent_id
                for instance in rep.instances
            ):
                return f"#/repetitions/{index}/instances"
            if message == "Repetition disagrees with parent order." and isinstance(
                parent, DefinitionGroup
            ):
                positions = [parent.children.index(i.node_id) for i in rep.instances]
                if positions != sorted(positions):
                    return f"#/repetitions/{index}/instances"
    return ""


_GRAPH_CODES = {
    "Duplicate record identity.": "graph_duplicate_id",
    "Duplicate shape symbol.": "graph_duplicate_symbol",
    "Parent must be a group.": "graph_invalid_parent",
    "Child and parent disagree.": "graph_containment_mismatch",
    "Parent and child disagree.": "graph_containment_mismatch",
    "Cyclic record linkage.": "graph_containment_cycle",
    "Duplicate child.": "graph_duplicate_child",
    "Tokenizer capability is unavailable.": "graph_tokenizer_unavailable",
    "Unsafe dimension.": "graph_unsafe_dimension",
    "Unsafe dimension product.": "graph_unsafe_dimension_product",
    "Unknown node parameter.": "graph_unknown_parameter",
    "Unknown parameter resource.": "graph_unknown_parameter",
    "Undeclared shape symbol.": "graph_unknown_symbol",
    "Undeclared expression symbol.": "graph_unknown_symbol",
    "Unknown edge endpoint.": "graph_unknown_endpoint",
    "Incompatible port directions.": "graph_edge_direction",
    "Edge bypasses a group boundary.": "graph_boundary_bypass",
    "Known connected dimensions disagree without a diagnostic.": "graph_dimension_mismatch",
    "Repetition parent must be a group.": "graph_repetition_parent",
    "Repetition indices must be strictly ordered.": "graph_repetition_order",
    "Repetition instance must be a direct child group.": "graph_repetition_membership",
    "Repetition disagrees with parent order.": "graph_repetition_order",
}


def diagnostic_for_model_error(
    exc: GraphError, definition: ModelDefinition | None = None
) -> r.ArchitectureDiagnostic:
    if str(exc).startswith("architecture.json"):
        return r.ArchitectureDiagnostic(code=exc.code, message=str(exc))
    if exc.code == "unsupported_size":
        return r.ArchitectureDiagnostic(
            code="unsupported_size",
            message=str(
                _finding(
                    "unsupported_size",
                    "resource",
                    "",
                    "Architecture exceeds the supported size limit.",
                )
            ),
        )
    message = str(exc)
    code = _GRAPH_CODES.get(
        message, "template_invalid" if "template" in message.lower() else "graph_invalid"
    )
    stage = "template" if code == "template_invalid" else "graph"
    location = _graph_location(message, definition) if definition is not None else ""
    if message == "Duplicate record identity.":
        if "/ports/" in location:
            code = "graph_duplicate_port"
        elif "/instances/" in location:
            code = "graph_duplicate_repetition_instance"
    # Only static messages from our validator are exposed. Unknown errors stay generic.
    safe_message = (
        message
        if code != "graph_invalid"
        else "Graph records disagree with the architecture contract."
    )
    return r.ArchitectureDiagnostic(
        code=code, message=str(_finding(code, stage, location, safe_message))
    )


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
        _check_bindings(definition, inputs)
        builder = GraphBuilder(
            inputs, producer_for(definition), "model_defined", byte_limit=byte_limit
        )
        build_definition(definition, inputs, builder)
        graph = bind_templates(definition, builder.finish(), builder)
        return AnalysisResult(graph, None)
    except GraphError as exc:
        reason: Literal["unsupported_size", "analysis_failed"] = (
            "unsupported_size" if exc.code == "unsupported_size" else "analysis_failed"
        )
        return AnalysisResult(None, reason, (diagnostic_for_model_error(exc, definition),))
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return unavailable(
            "analysis_failed",
            "internal_validation_failure",
            "architecture.json [graph]: Definition could not be validated.",
        )
