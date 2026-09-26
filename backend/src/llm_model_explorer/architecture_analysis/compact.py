"""Verified, lossless wire form for repeated routed expert components.

The definition contains actual semantic node and edge records. Instance arrays
only substitute record identities and a reviewed module prefix; they never
derive an operation or a connection from a checkpoint name.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import records as r
from .validation import GraphError, require


def _prefix(
    members: list[r.ArchitectureNode], parameters: Mapping[str, r.ArchitectureParameter]
) -> str:
    names = [parameters[pid].name for pid in _parameter_ids(members)]
    require(bool(names), "Compact expert has no bound parameters.")
    first = names[0].rsplit(".", 2)
    require(len(first) == 3, "Compact expert has no source prefix.")
    prefix = first[0]
    require(all(name.startswith(prefix + ".") for name in names), "Compact expert parameter scope.")
    return prefix


def _members(root_id: str, nodes: Mapping[str, r.ArchitectureNode]) -> list[r.ArchitectureNode]:
    result: list[r.ArchitectureNode] = []
    pending = [root_id]
    while pending:
        node_id = pending.pop()
        require(node_id in nodes, "Compact expert containment.")
        node = nodes[node_id]
        require(node not in result, "Compact expert cyclic containment.")
        result.append(node)
        if isinstance(node, r.ArchitectureGroupNode):
            pending.extend(reversed(node.children))
    return result


def _parameter_ids(members: list[r.ArchitectureNode]) -> list[str]:
    result: list[str] = []
    for node in members:
        for pid in node.parameter_ids:
            if pid not in result:
                result.append(pid)
        for ref in node.references:
            if isinstance(ref, r.ArchitectureParameterReference) and ref.parameter_id not in result:
                result.append(ref.parameter_id)
    return result


def _symbols(members: list[r.ArchitectureNode]) -> list[str]:
    result: list[str] = []
    for node in members:
        for port in node.ports:
            for dimension in port.shape or []:
                names = (
                    [dimension.name]
                    if isinstance(dimension, r.ArchitectureSymbolDimension)
                    else dimension.symbols
                    if isinstance(dimension, r.ArchitectureExpressionDimension)
                    else []
                )
                for name in names:
                    if name not in result:
                        result.append(name)
    return result


def _known_structure(members: list[r.ArchitectureNode]) -> None:
    root = members[0]
    require(
        isinstance(root, r.ArchitectureGroupNode)
        and any(a.name == "semantic_role" and a.value == "mlp" for a in root.attributes),
        "Compact expert requires a reviewed MLP boundary.",
    )
    for node in members:
        require(
            isinstance(node, r.ArchitectureGroupNode) or node.operation is not None,
            "Compact expert operation is unknown.",
        )
        for port in node.ports:
            require(
                port.shape is not None and all(dim.kind != "unknown" for dim in port.shape),
                "Compact expert shape is unknown.",
            )
        require(
            all(
                a.value is not None and not (isinstance(a.value, list) and None in a.value)
                for a in node.attributes
            ),
            "Compact expert attribute is unknown.",
        )


def _replace(value: Any, ids: Mapping[str, str], before: str, after: str) -> Any:
    if isinstance(value, str):
        return ids.get(value, value.replace(before, after))
    if isinstance(value, list):
        return [_replace(item, ids, before, after) for item in value]
    if isinstance(value, dict):
        return {key: _replace(item, ids, before, after) for key, item in value.items()}
    return value


def _render(
    family: r.ArchitectureCompactComponent, instance: r.ArchitectureCompactInstance
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(len(instance.node_ids) == len(family.nodes), "Compact node mapping length.")
    require(len(instance.edge_ids) == len(family.edges), "Compact edge mapping length.")
    require(
        len(instance.parameter_ids) == len(family.parameter_ids),
        "Compact parameter mapping length.",
    )
    require(len(instance.symbols) == len(family.symbols), "Compact symbol mapping length.")
    require(instance.node_ids[0] == instance.node_id, "Compact root identity.")
    for values in (instance.node_ids, instance.edge_ids, instance.parameter_ids):
        require(len(set(values)) == len(values), "Duplicate compact mapping identity.")
    ids = dict(zip((n.id for n in family.nodes), instance.node_ids, strict=True))
    ids.update(zip((e.id for e in family.edges), instance.edge_ids, strict=True))
    ids.update(zip(family.parameter_ids, instance.parameter_ids, strict=True))
    ids.update(zip(family.symbols, instance.symbols, strict=True))
    nodes = [_replace(n.document(), ids, family.base_prefix, instance.prefix) for n in family.nodes]
    edges = [_replace(e.document(), ids, family.base_prefix, instance.prefix) for e in family.edges]
    nodes[0]["label"] = instance.label
    for attribute in nodes[0]["attributes"]:
        if attribute["name"] == "expert_index":
            attribute["value"] = float(instance.index)
    return nodes, edges


def expand_graph(graph: r.ArchitectureGraph) -> r.ArchitectureGraph:
    """Materialize all concrete records and check explicit identity/binding maps."""
    if not graph.compact_components:
        return graph
    document = graph.document()
    document.pop("compact_components")
    nodes = document["nodes"]
    edges = document["edges"]
    parameters = {p["id"]: p for p in document["parameters"]}
    repetitions = {rep["id"]: rep for rep in document["repetitions"]}
    seen = {
        record["id"]
        for key in ("nodes", "edges", "parameters", "repetitions")
        for record in document[key]
    }
    prefixes: set[str] = set()
    for family in graph.compact_components:
        require(family.id not in seen, "Duplicate compact family identity.")
        seen.add(family.id)
        repetition = repetitions.get(family.repetition_id)
        if repetition is None:
            raise GraphError("invalid_graph", "Compact repetition does not exist.")
        expected = [(item["node_id"], item["index"]) for item in repetition["instances"]]
        actual = [(item.node_id, item.index) for item in family.instances]
        require(expected == actual, "Compact instances disagree with repetition order.")
        require(
            all(
                item["variant"] in {"routed_expert", "routed_swiglu"}
                for item in repetition["instances"]
            ),
            "Compact family is not a routed expert repetition.",
        )
        prototype = family.nodes[0]
        require(isinstance(prototype, r.ArchitectureGroupNode), "Compact root must be a group.")
        _known_structure(list(family.nodes))
        first_instance = family.instances[0]
        require(
            first_instance.prefix == family.base_prefix
            and first_instance.node_ids == [n.id for n in family.nodes]
            and first_instance.edge_ids == [e.id for e in family.edges]
            and first_instance.parameter_ids == family.parameter_ids
            and first_instance.symbols == family.symbols,
            "Compact prototype is not the first instance.",
        )
        require(
            _parameter_ids(list(family.nodes)) == family.parameter_ids,
            "Compact parameter role coverage.",
        )
        require(_symbols(list(family.nodes)) == family.symbols, "Compact symbol role coverage.")
        require(
            all(
                name in {symbol["name"] for symbol in document["symbols"]}
                for name in family.symbols
            ),
            "Unknown compact prototype symbol.",
        )
        prototype_names = [parameters[pid]["name"] for pid in family.parameter_ids]
        require(
            all(name.startswith(family.base_prefix + ".") for name in prototype_names),
            "Compact prototype binding scope.",
        )
        for instance in family.instances:
            require(instance.prefix not in prefixes, "Duplicate compact source prefix.")
            prefixes.add(instance.prefix)
            require(
                instance.prefix != family.base_prefix or instance.node_id == prototype.id,
                "Compact prototype mismatch.",
            )
            require(
                len(instance.node_ids) == len(family.nodes)
                and len(instance.edge_ids) == len(family.edges)
                and len(instance.parameter_ids) == len(family.parameter_ids)
                and len(instance.symbols) == len(family.symbols),
                "Compact mapping length.",
            )
            for expected_name, pid in zip(prototype_names, instance.parameter_ids, strict=True):
                parameter = parameters.get(pid)
                require(
                    parameter is not None
                    and parameter["name"]
                    == expected_name.replace(family.base_prefix, instance.prefix),
                    "Compact expert binding is missing or swapped.",
                )
            require(
                all(
                    name in {symbol["name"] for symbol in document["symbols"]}
                    for name in instance.symbols
                ),
                "Unknown compact instance symbol.",
            )
            rendered_nodes, rendered_edges = _render(family, instance)
            require(
                rendered_nodes[0]["parent_id"] == repetition["parent_id"],
                "Compact parent mismatch.",
            )
            for record in [*rendered_nodes, *rendered_edges]:
                require(record["id"] not in seen, "Duplicate reconstructed identity.")
                seen.add(record["id"])
            nodes.extend(rendered_nodes)
            edges.extend(rendered_edges)
    return r.ArchitectureGraph.model_validate(document)


def compact_graph(graph: r.ArchitectureGraph) -> r.ArchitectureGraph:
    """Share only expert records whose exact reconstruction matches the source."""
    nodes = {node.id: node for node in graph.nodes}
    edges = list(graph.edges)
    parameters = {p.id: p for p in graph.parameters}
    diagnosed_nodes = {d.node_id for d in graph.diagnostics if d.node_id is not None}
    families: list[r.ArchitectureCompactComponent] = []
    removed_nodes: set[str] = set()
    removed_edges: set[str] = set()
    for repetition in graph.repetitions:
        if len(repetition.instances) < 2 or not all(
            item.variant in {"routed_expert", "routed_swiglu"} for item in repetition.instances
        ):
            continue
        first = nodes.get(repetition.instances[0].node_id)
        if not isinstance(first, r.ArchitectureGroupNode):
            continue
        try:
            prototypes = _members(first.id, nodes)
            _known_structure(prototypes)
            require(
                not {n.id for n in prototypes}.intersection(diagnosed_nodes),
                "Diagnosed expert is not compactable.",
            )
            base_prefix = _prefix(prototypes, parameters)
            member_ids = {n.id for n in prototypes}
            prototype_edges = [
                e
                for e in edges
                if e.source.node_id in member_ids and e.target.node_id in member_ids
            ]
            parameter_ids = _parameter_ids(prototypes)
            symbols = _symbols(prototypes)
            instances: list[r.ArchitectureCompactInstance] = []
            for item in repetition.instances:
                root = nodes.get(item.node_id)
                if not isinstance(root, r.ArchitectureGroupNode):
                    raise GraphError("invalid_graph", "Compact expert root.")
                members = _members(root.id, nodes)
                _known_structure(members)
                require(
                    not {n.id for n in members}.intersection(diagnosed_nodes),
                    "Diagnosed expert is not compactable.",
                )
                prefix = _prefix(members, parameters)
                member_ids = {n.id for n in members}
                internal_edges = [
                    e
                    for e in edges
                    if e.source.node_id in member_ids and e.target.node_id in member_ids
                ]
                actual_parameter_ids = _parameter_ids(members)
                actual_symbols = _symbols(members)
                instance = r.ArchitectureCompactInstance(
                    node_id=root.id,
                    prefix=prefix,
                    label=root.label,
                    index=item.index,
                    node_ids=[n.id for n in members],
                    edge_ids=[e.id for e in internal_edges],
                    parameter_ids=actual_parameter_ids,
                    symbols=actual_symbols,
                )
                probe = r.ArchitectureCompactComponent(
                    id=repetition.id,
                    repetition_id=repetition.id,
                    base_prefix=base_prefix,
                    nodes=prototypes,
                    edges=prototype_edges,
                    parameter_ids=parameter_ids,
                    symbols=symbols,
                    instances=[instance, instance],
                )
                rendered_nodes, rendered_edges = _render(probe, instance)
                require(
                    rendered_nodes == [n.document() for n in members],
                    "Unverified expert node equivalence.",
                )
                require(
                    rendered_edges == [e.document() for e in internal_edges],
                    "Unverified expert edge equivalence.",
                )
                require(
                    all(
                        parameters[pid].name == parameters[source].name.replace(base_prefix, prefix)
                        for source, pid in zip(parameter_ids, actual_parameter_ids, strict=True)
                    ),
                    "Unverified expert binding correspondence.",
                )
                instances.append(instance)
            family = r.ArchitectureCompactComponent(
                id=repetition.id + "~compact",
                repetition_id=repetition.id,
                base_prefix=base_prefix,
                nodes=prototypes,
                edges=prototype_edges,
                parameter_ids=parameter_ids,
                symbols=symbols,
                instances=instances,
            )
        except (GraphError, KeyError, ValueError):
            continue  # Structural exceptions remain concrete and count toward the wire budget.
        families.append(family)
        for instance in instances:
            removed_nodes.update(instance.node_ids)
            removed_edges.update(instance.edge_ids)
    if not families:
        return graph
    compact = graph.model_copy(
        update={
            "nodes": [n for n in graph.nodes if n.id not in removed_nodes],
            "edges": [e for e in graph.edges if e.id not in removed_edges],
            "compact_components": families,
        }
    )
    expanded = expand_graph(compact)
    require(
        {n.id: n.document() for n in expanded.nodes} == {n.id: n.document() for n in graph.nodes},
        "Compact node round trip.",
    )
    require(
        {e.id: e.document() for e in expanded.edges} == {e.id: e.document() for e in graph.edges},
        "Compact edge round trip.",
    )
    return compact
