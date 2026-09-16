"""Optional annotations from explicitly opened, reviewed description components.

Keys below are authored construction keys, never checkpoint path discovery. A
candidate is collected while its description builds it; complete verification
still decides whether two candidates can share a structure.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from . import records as r
from .template_validation import TemplateIndex
from .validation import GraphError, serialized_size

if TYPE_CHECKING:
    from .core import GraphBuilder

REVISION = "exact-component-roles-1"
LOG = logging.getLogger(__name__)


@dataclass
class Candidate:
    node_id: str
    base: str
    family: str
    role: Literal["attention", "mlp"]
    nodes: list[r.ArchitectureTemplateNodeRole] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)
    valid: bool = True


class ComponentTemplates:
    def __init__(self) -> None:
        self.candidates: dict[str, Candidate] = {}

    def begin(
        self, node_id: str, base: str, family: str, role: Literal["attention", "mlp"]
    ) -> None:
        self.candidates[node_id] = Candidate(node_id, base + ".", family, role)

    def observe(
        self,
        node: r.ArchitectureNode,
        key: str | None,
        parameters: dict[str, r.ArchitectureParameter],
    ) -> None:
        candidate = self.candidates.get(node.id) or self.candidates.get(node.parent_id or "")
        if candidate is None:
            return
        if key is None or (node.id != candidate.node_id and not key.startswith(candidate.base)):
            candidate.valid = False
            return
        role = "component" if node.id == candidate.node_id else key.removeprefix(candidate.base)
        try:
            candidate.nodes.append(r.ArchitectureTemplateNodeRole(role=role, node_id=node.id))
            for parameter_id in node.parameter_ids:
                name = parameters[parameter_id].name
                if not name.startswith(candidate.base):
                    candidate.valid = False
                    continue
                candidate.parameters[parameter_id] = name.removeprefix(candidate.base)
        except (ValueError, KeyError):
            # Invalid optional role metadata never invalidates an ordinary source record.
            candidate.valid = False

    def annotate(self, graph: r.ArchitectureGraph, builder: GraphBuilder) -> r.ArchitectureGraph:
        if not self.candidates:
            return graph
        index = TemplateIndex(graph)
        families: dict[tuple[str, str, str], list[r.ArchitectureTemplateInstance]] = defaultdict(
            list
        )
        omitted = False
        remaining = builder.byte_limit - serialized_size(graph.document(), builder.byte_limit)
        # Account for the optional property and commas without changing ordinary records.
        remaining -= len(',"templates":[]')
        for candidate in self.candidates.values():
            try:
                if not candidate.valid:
                    raise GraphError("invalid_graph", "Unverified authored component roles.")
                roles = {n.node_id: n.role for n in candidate.nodes}
                members = set(roles)
                ports = [
                    r.ArchitectureTemplatePortRole(
                        role=roles[node_id] + "." + p.id, node_id=node_id, port_id=p.id
                    )
                    for node_id in roles
                    for p in index.nodes[node_id].ports
                ]
                edge_ids = {
                    edge_id
                    for node_id in members
                    for edge_id in index.incident[node_id]
                    if index.edges[edge_id].source.node_id in members
                    and index.edges[edge_id].target.node_id in members
                }
                # Endpoint role keys are exact authored ports, with no external producer copied.
                edges = [
                    r.ArchitectureTemplateEdgeRole(
                        role=roles[e.source.node_id]
                        + "."
                        + e.source.port_id
                        + "~"
                        + roles[e.target.node_id]
                        + "."
                        + e.target.port_id,
                        edge_id=e.id,
                    )
                    for edge_id in sorted(edge_ids)
                    for e in [index.edges[edge_id]]
                ]
                instance = r.ArchitectureTemplateInstance(
                    node_id=candidate.node_id,
                    nodes=candidate.nodes,
                    ports=ports,
                    edges=edges,
                    parameters=[
                        r.ArchitectureTemplateParameterRole(role=role, parameter_id=pid)
                        for pid, role in candidate.parameters.items()
                    ],
                )
                # Bound temporary optional mappings too, not only the final serialization.
                size = serialized_size(instance.document(), max(0, remaining))
                signature = index.signature(instance, candidate.role)
                scope, _ = index.order.get(candidate.node_id, (None, -1))
                if scope is None:
                    raise GraphError("invalid_graph", "No declared template scope.")
                families[(candidate.family, scope, signature)].append(instance)
                remaining -= size + 1
            except (GraphError, ValueError):
                omitted = True
        templates: list[r.ArchitectureTemplate] = []
        for (family, scope, _), instances in families.items():
            if len(instances) < 2:
                continue
            instances.sort(key=lambda i: index.order[i.node_id][1])
            candidate = self.candidates[instances[0].node_id]
            template = r.ArchitectureTemplate(
                id=builder.record_id("template", family + ":" + scope + ":" + instances[0].node_id),
                label=family.replace("_", " "),
                component_role=candidate.role,
                revision=REVISION,
                provenance=builder.producer.provenance(),
                instances=instances,
            )
            # Envelope overhead is charged in addition to already bounded instance mappings.
            overhead = serialized_size(template.document(), builder.byte_limit) - sum(
                serialized_size(i.document(), builder.byte_limit) + 1 for i in instances
            )
            if overhead > remaining:
                omitted = True
                continue
            remaining -= overhead
            templates.append(template)
        result = graph.model_copy(update={"templates": templates}) if templates else graph
        if omitted:
            LOG.warning(
                "Optional shared structures omitted: unverified mapping or metadata budget."
            )
            diagnostic = r.ArchitectureDiagnostic(
                code="templates_omitted",
                message="Some optional shared structures could not be verified "
                "within the metadata budget.",
            )
            diagnosed = result.model_copy(update={"diagnostics": [*graph.diagnostics, diagnostic]})
            try:
                serialized_size(diagnosed.document(), builder.byte_limit)
                result = diagnosed
            except GraphError:
                pass  # A full ordinary graph wins even when no diagnostic bytes remain.
        return result
