"""Independent test navigation through exact group ports, never through operations."""

from llm_model_explorer.architecture_analysis import records as r


def semantic_key(node: r.ArchitectureNode) -> str:
    return next(
        (
            p.source
            for p in node.provenance
            if p.rule == "Semantic source key in the reviewed packaged description"
        ),
        node.label,
    )


def transparent_edges(graph: r.ArchitectureGraph) -> set[tuple[str, str, str, str]]:
    """Include direct links and their transparent boundary compositions for assertions.

    Multiplicity/conservation has a separate multiset oracle in issue #119; this
    helper keeps existing hand-authored dependency assertions readable.
    """
    groups = {node.id for node in graph.nodes if node.kind == "group"}
    outgoing: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for edge in graph.edges:
        outgoing.setdefault((edge.source.node_id, edge.source.port_id), []).append(
            (edge.target.node_id, edge.target.port_id)
        )
    result = set()
    for source, targets in outgoing.items():
        pending, seen = list(targets), set()
        while pending:
            target = pending.pop()
            if target in seen:
                continue
            seen.add(target)
            result.add((*source, *target))
            if target[0] in groups:
                pending.extend(outgoing.get(target, []))
    return result
