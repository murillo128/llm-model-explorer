"""Bind explicit model-owned correspondences, never discover them from names."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from . import records as r
from .model_defined_schema import DefinitionTemplate, ModelDefinition
from .template_validation import TemplateIndex, validate_template_instances
from .validation import GraphError, model_finding, serialized_size, validate_graph

if TYPE_CHECKING:
    from .core import GraphBuilder

LOG = logging.getLogger(__name__)


def _bind_instance(
    instance: r.ArchitectureTemplateInstance, builder: GraphBuilder
) -> r.ArchitectureTemplateInstance:
    def local(kind: str, key: str) -> str:
        return builder.record_id(kind, "declared:" + key)

    return r.ArchitectureTemplateInstance(
        node_id=local("node", instance.node_id),
        nodes=[
            r.ArchitectureTemplateNodeRole(role=m.role, node_id=local("node", m.node_id))
            for m in instance.nodes
        ],
        ports=[
            r.ArchitectureTemplatePortRole(
                role=m.role, node_id=local("node", m.node_id), port_id=m.port_id
            )
            for m in instance.ports
        ],
        edges=[
            r.ArchitectureTemplateEdgeRole(role=m.role, edge_id=local("edge", m.edge_id))
            for m in instance.edges
        ],
        parameters=[
            r.ArchitectureTemplateParameterRole(
                role=m.role, parameter_id=local("parameter", m.parameter_id)
            )
            for m in instance.parameters
        ],
    )


def _bind_template(
    declared: DefinitionTemplate,
    index: TemplateIndex,
    builder: GraphBuilder,
    seen: set[str],
    remaining: int,
    has_previous: bool,
) -> tuple[r.ArchitectureTemplate | None, int]:
    envelope = {
        "id": builder.record_id("template", "declared:" + declared.id),
        "label": declared.label,
        "component_role": declared.component_role,
        "revision": builder.producer.source_revision,
        "provenance": [p.document() for p in builder.producer.provenance()],
        "instances": [],
    }
    fits = True
    used = 0
    instances: list[r.ArchitectureTemplateInstance] = []
    try:
        used = serialized_size(envelope, max(0, remaining)) + int(has_previous)
    except GraphError as exc:
        if exc.code != "unsupported_size":
            raise
        fits = False

    def mapped_instances() -> Iterator[r.ArchitectureTemplateInstance]:
        nonlocal fits, used
        for source in declared.instances:
            instance = _bind_instance(source, builder)
            if fits:
                try:
                    used += serialized_size(instance.document(), max(0, remaining - used))
                    used += int(bool(instances))
                    if used > remaining:
                        raise GraphError("unsupported_size", "Template metadata limit.")
                    instances.append(instance)
                except GraphError as exc:
                    if exc.code != "unsupported_size":
                        raise
                    fits = False
                    instances.clear()
            # Validation consumes every instance, even after the family's
            # optional output budget is exhausted. Invalid is never omitted.
            yield instance

    validate_template_instances(index, mapped_instances(), declared.component_role, seen)
    if not fits:
        return None, 0
    return (
        r.ArchitectureTemplate.model_validate(
            {**envelope, "instances": [i.document() for i in instances]}
        ),
        used,
    )


def bind_templates(
    definition: ModelDefinition, graph: r.ArchitectureGraph, builder: GraphBuilder
) -> r.ArchitectureGraph:
    if not definition.templates:
        return graph
    seen_ids: set[str] = set()
    for position, declared in enumerate(definition.templates):
        if declared.id in seen_ids:
            raise model_finding(
                "template_duplicate_id",
                "template",
                f"#/templates/{position}/id",
                "Duplicate template identity.",
            )
        seen_ids.add(declared.id)
    index = TemplateIndex(graph)
    seen: set[str] = set()
    retained: list[r.ArchitectureTemplate] = []
    remaining = builder.byte_limit - serialized_size(graph.document(), builder.byte_limit)
    remaining -= len(',"templates":[]')
    omitted = False
    for position, declared in enumerate(definition.templates):
        try:
            template, size = _bind_template(
                declared, index, builder, seen, remaining, bool(retained)
            )
        except GraphError as exc:
            if exc.code == "unsupported_size":
                raise
            raise model_finding(
                "template_invalid", "template", f"#/templates/{position}", str(exc)
            ) from exc
        if template is None:
            omitted = True
        else:
            retained.append(template)
            remaining -= size
    result = graph.model_copy(update={"templates": retained}) if retained else graph
    if omitted:
        LOG.warning("Some model-supplied shared structures exceeded the metadata budget.")
        diagnostic = r.ArchitectureDiagnostic(
            code="templates_omitted",
            message="Some valid model-supplied shared structures exceeded the metadata budget.",
        )
        diagnosed = result.model_copy(update={"diagnostics": [*graph.diagnostics, diagnostic]})
        try:
            serialized_size(diagnosed.document(), builder.byte_limit)
            result = diagnosed
        except GraphError as exc:
            if exc.code != "unsupported_size":
                raise
    serialized_size(result.document(), builder.byte_limit)
    validate_graph(result, builder.inputs.bindings)
    return result
