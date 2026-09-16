"""Conservative, metadata-only verification of optional component correspondences."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from . import records as r
from .validation import require, terminals, unique


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


class TemplateIndex:
    """One graph index; work is proportional to the graph and declared mapping sizes."""

    def __init__(self, graph: r.ArchitectureGraph):
        self.graph = graph
        self.nodes = {n.id: n for n in graph.nodes}
        self.parameters = {p.id: p for p in graph.parameters}
        self.edges = {e.id: e for e in graph.edges}
        self.incident: dict[str, set[str]] = defaultdict(set)
        for edge in graph.edges:
            self.incident[edge.source.node_id].add(edge.id)
            self.incident[edge.target.node_id].add(edge.id)
        self.aliases = terminals(self.parameters, "alias_of")
        repetitions = {
            instance.node_id: (rep.id, position)
            for rep in graph.repetitions
            for position, instance in enumerate(rep.instances)
        }
        self.order: dict[str, tuple[str | None, int]] = {}
        pending: list[tuple[str, tuple[str, int] | None]] = [
            (n.id, None) for n in graph.nodes if n.parent_id is None
        ]
        while pending:
            node_id, inherited = pending.pop()
            node = self.nodes[node_id]
            owner = repetitions.get(node_id, inherited)
            if owner is not None:
                self.order[node_id] = owner
            if node.kind == "group":
                for position, child in enumerate(node.children):
                    self.order[child] = owner or (node_id, position)
                    pending.append((child, owner))
        self.diagnosed_nodes = {d.node_id for d in graph.diagnostics if d.node_id is not None}

    def members(self, root: str) -> set[str]:
        require(root in self.nodes and self.nodes[root].kind == "group", "Template component.")
        result: set[str] = set()
        pending = [root]
        while pending:
            current = pending.pop()
            require(current not in result, "Template containment.")
            result.add(current)
            node = self.nodes[current]
            if node.kind == "group":
                pending.extend(node.children)
        return result

    def signature(self, instance: r.ArchitectureTemplateInstance, component_role: str) -> str:
        members = self.members(instance.node_id)
        require(not members.intersection(self.diagnosed_nodes), "Unverified template member.")
        node_roles = {m.node_id: m.role for m in instance.nodes}
        port_roles = {(m.node_id, m.port_id): m.role for m in instance.ports}
        edge_roles = {m.edge_id: m.role for m in instance.edges}
        parameter_roles = {m.parameter_id: m.role for m in instance.parameters}
        for mappings, targets in (
            (instance.nodes, node_roles),
            (instance.ports, port_roles),
            (instance.edges, edge_roles),
            (instance.parameters, parameter_roles),
        ):
            unique(mappings, "role")
            require(len(mappings) == len(targets), "Duplicate template mapping target.")
        require(set(node_roles) == members, "Incomplete or foreign template nodes.")
        require(
            set(port_roles)
            == {(node_id, port.id) for node_id in members for port in self.nodes[node_id].ports},
            "Incomplete or foreign template ports.",
        )
        expected_edges = {
            edge_id
            for node_id in members
            for edge_id in self.incident[node_id]
            if self.edges[edge_id].source.node_id in members
            and self.edges[edge_id].target.node_id in members
        }
        require(set(edge_roles) == expected_edges, "Incomplete or foreign template edges.")
        expected_parameters = {
            parameter_id
            for node_id in members
            for parameter_id in self.nodes[node_id].parameter_ids
        } | {
            ref.parameter_id
            for node_id in members
            for ref in self.nodes[node_id].references
            if ref.kind == "parameter"
        }
        require(
            set(parameter_roles) == expected_parameters,
            "Incomplete or foreign template parameters.",
        )

        def shape(value: r.ArchitectureShape) -> Any:
            require(value is not None, "Unknown template rank.")
            assert value is not None
            require(all(d.kind != "unknown" for d in value), "Unknown template dimension.")
            return [d.document() for d in value]

        signatures: dict[str, Any] = {"root": node_roles[instance.node_id]}
        ns: dict[str, Any] = {}
        for node_id, role in node_roles.items():
            node = self.nodes[node_id]
            attributes = unique(node.attributes, "name")
            require(
                all(
                    a.value is not None and not (isinstance(a.value, list) and None in a.value)
                    for a in attributes.values()
                ),
                "Unknown template attribute.",
            )
            if node_id == instance.node_id:
                require(
                    "semantic_role" in attributes
                    and attributes["semantic_role"].value == component_role,
                    "Template component role.",
                )
            if node.kind != "group":
                require(node.operation is not None, "Unknown template operation.")
            ns[role] = {
                "kind": node.kind,
                "operation": node.operation,
                "formula": node.formula,
                "description": node.description,
                "parent": node_roles.get(node.parent_id or ""),
                "children": [node_roles[c] for c in node.children] if node.kind == "group" else [],
                "ports": [port_roles[(node_id, p.id)] for p in node.ports],
                "parameters": [parameter_roles[p] for p in node.parameter_ids],
                "references": [
                    parameter_roles[ref.parameter_id]
                    for ref in node.references
                    if ref.kind == "parameter"
                ],
                "attributes": {name: a.value for name, a in attributes.items()},
            }
        signatures["nodes"] = ns
        signatures["ports"] = {
            port_roles[(node_id, p.id)]: [node_roles[node_id], p.id, p.direction, shape(p.shape)]
            for node_id in members
            for p in self.nodes[node_id].ports
        }
        signatures["edges"] = {
            role: [
                port_roles[(edge.source.node_id, edge.source.port_id)],
                port_roles[(edge.target.node_id, edge.target.port_id)],
                edge.kind,
            ]
            for edge_id, role in edge_roles.items()
            for edge in [self.edges[edge_id]]
        }
        alias_roles: dict[str, list[str]] = defaultdict(list)
        for parameter_id, role in parameter_roles.items():
            alias_roles[self.aliases[parameter_id]].append(role)
        signatures["parameters"] = {
            role: [
                shape(self.parameters[parameter_id].logical_shape),
                sorted(alias_roles[self.aliases[parameter_id]]),
            ]
            for parameter_id, role in parameter_roles.items()
        }
        return canonical(signatures)


def validate_templates(graph: r.ArchitectureGraph) -> None:
    if not graph.templates:
        return
    unique([*graph.nodes, *graph.edges, *graph.parameters, *graph.repetitions, *graph.templates])
    index = TemplateIndex(graph)
    seen: set[str] = set()
    for template in graph.templates:
        require(
            any(p.kind == "description" and p.revision for p in template.provenance),
            "Template requires reviewed description provenance.",
        )
        baseline: str | None = None
        scope: str | None = None
        previous = -1
        for instance in template.instances:
            require(instance.node_id not in seen, "Repeated template component.")
            seen.add(instance.node_id)
            owner, position = index.order.get(instance.node_id, (None, -1))
            require(
                owner is not None and (scope is None or scope == owner) and position > previous,
                "Template source scope/order.",
            )
            scope, previous = owner, position
            signature = index.signature(instance, template.component_role)
            require(baseline is None or signature == baseline, "Incompatible template structure.")
            baseline = signature
