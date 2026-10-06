"""Bounded depth motifs over exact Shared roles and real boundary wiring.

Annotations reference intervals of the existing repetition; no source wrappers or
computations are introduced. Period strings only nominate candidates.
"""

from __future__ import annotations

from collections import defaultdict

from . import records as r
from .template_validation import TemplateIndex
from .validation import GraphError, require, serialized_size


def ranges(slots: list[str]) -> list[r.ArchitectureBodyRange]:
    result: list[r.ArchitectureBodyRange] = []
    for offset, family in enumerate(slots):
        if offset and slots[offset - 1] == family:
            result[-1] = result[-1].model_copy(update={"count": result[-1].count + 1})
        else:
            result.append(r.ArchitectureBodyRange(start=offset, count=1))
    return result


class BodyIndex:
    def __init__(self, graph: r.ArchitectureGraph):
        self.graph = graph
        self.index = TemplateIndex(graph)
        self.families = {
            instance.node_id: (template, instance)
            for template in graph.templates or []
            if template.component_role == "layer"
            for instance in template.instances
        }
        self.incoming: dict[tuple[str, str], list[r.ArchitectureEdge]] = defaultdict(list)
        self.outgoing: dict[tuple[str, str], list[r.ArchitectureEdge]] = defaultdict(list)
        for edge in graph.edges:
            self.incoming[(edge.target.node_id, edge.target.port_id)].append(edge)
            self.outgoing[(edge.source.node_id, edge.source.port_id)].append(edge)

    def verify(self, rep: r.ArchitectureRepetition, body: r.ArchitectureRepeatedBody) -> None:
        start, width, count = body.start, body.width, body.count
        require(start >= 0 and width > 1 and count >= 2, "Repeated body bounds.")
        require(start + width * count <= len(rep.instances), "Repeated body interval.")
        require(len(body.slots) == width and len(set(body.slots)) > 1, "Repeated body slots.")
        require(body.ranges == ranges(body.slots), "Repeated body nested ranges.")
        instances = rep.instances[start : start + width * count]
        roots = [i.node_id for i in instances]
        parent = self.index.nodes[rep.parent_id]
        require(isinstance(parent, r.ArchitectureGroupNode), "Repeated body parent.")
        assert isinstance(parent, r.ArchitectureGroupNode)
        position = parent.children.index(roots[0])
        require(
            parent.children[position : position + len(roots)] == roots, "Repeated body siblings."
        )
        require(
            all(i.index == instances[0].index + at for at, i in enumerate(instances)),
            "Repeated body depth indices.",
        )
        require(
            all(
                root in self.families
                and self.families[root][0].id == body.slots[at % width]
                and instances[at].variant == instances[at % width].variant
                for at, root in enumerate(roots)
            ),
            "Repeated body Shared slot mapping.",
        )
        members = {n.node_id for root in roots for n in self.families[root][1].nodes}
        root_set = set(roots)
        # Only declared root interfaces may cross a layer closure.
        for root in roots:
            closure = {n.node_id for n in self.families[root][1].nodes}
            for member in closure:
                for edge_id in self.index.incident[member]:
                    edge = self.index.edges[edge_id]
                    if (edge.source.node_id in closure) != (edge.target.node_id in closure):
                        internal = edge.source if edge.source.node_id in closure else edge.target
                        require(internal.node_id == root, "Repeated body bypasses layer boundary.")

        def path_to(root: str, port: str) -> list[r.ArchitectureEdge]:
            current = (root, port)
            path: list[r.ArchitectureEdge] = []
            seen: set[str] = set()
            while True:
                incoming = [
                    e
                    for e in self.incoming[current]
                    if not (
                        e.target.node_id == root
                        and e.source.node_id in members
                        and e.source.node_id not in root_set
                    )
                ]
                require(len(incoming) == 1, "Repeated body ambiguous input.")
                edge = incoming[0]
                require(edge.id not in seen, "Repeated body cyclic forwarding.")
                seen.add(edge.id)
                path.insert(0, edge)
                current = (edge.source.node_id, edge.source.port_id)
                if (
                    current[0] in root_set
                    or self.index.nodes[current[0]].kind != "group"
                    or not self.incoming[current]
                ):
                    return path

        side_ports = {(s.port_id, s.direction): s for s in rep.side_ports or []}
        invariant: dict[tuple[int, str], tuple[str, str, tuple[str, ...]] | None] = {}
        for at, root in enumerate(roots):
            node = self.index.nodes[root]
            ports = {p.id: p.direction for p in node.ports}
            require(
                ports.get(body.input_port) == "input" and ports.get(body.output_port) == "output",
                "Repeated body activation ports.",
            )
            activation = path_to(root, body.input_port)
            require(all(e.kind == "data" for e in activation), "Repeated body activation kind.")
            expected = (roots[at - 1], body.output_port) if at else None
            source = activation[0].source
            require(
                (source.node_id, source.port_id) == expected
                if at
                else source.node_id not in members,
                "Repeated body activation order.",
            )
            # All outgoing branches from a non-final activation must reach only
            # the next input, including forwarding after the first segment.
            pending = list(self.outgoing[(root, body.output_port)])
            seen: set[str] = set()
            ends: list[tuple[str, str]] = []
            while pending:
                edge = pending.pop()
                require(edge.id not in seen and edge.kind == "data", "Repeated body output wiring.")
                seen.add(edge.id)
                target = (edge.target.node_id, edge.target.port_id)
                following = (
                    self.outgoing[target]
                    if target[0] not in root_set and self.index.nodes[target[0]].kind == "group"
                    else []
                )
                if following:
                    pending.extend(following)
                else:
                    ends.append(target)
            require(
                ends == [(roots[at + 1], body.input_port)]
                if at + 1 < len(roots)
                else bool(ends) and all(n not in members for n, _ in ends),
                "Repeated body output bypass.",
            )
            closure = {n.node_id for n in self.families[root][1].nodes}
            for port in node.ports:
                if port.id in {body.input_port, body.output_port}:
                    continue
                external = [
                    e
                    for e in (self.incoming if port.direction == "input" else self.outgoing)[
                        (root, port.id)
                    ]
                    if (e.source.node_id if port.direction == "input" else e.target.node_id)
                    not in closure
                ]
                key = (at % width, port.id)
                if not external:
                    require(
                        key not in invariant or invariant[key] is None,
                        "Repeated body missing side input.",
                    )
                    invariant[key] = None
                    continue
                if (port.id, port.direction) in side_ports:
                    # validate_graph owns exact indexed state bindings.
                    require(
                        all(e.kind == "state" for e in external),
                        "Repeated body indexed state kind.",
                    )
                    continue
                require(port.direction == "input", "Repeated body undeclared side output.")
                path = path_to(root, port.id)
                source = path[0].source
                require(source.node_id not in members, "Repeated body non-invariant side input.")
                key = (at % width, port.id)
                signature = (source.node_id, source.port_id, tuple(e.kind for e in path))
                require(
                    key not in invariant or invariant[key] == signature,
                    "Repeated body side correspondence.",
                )
                invariant[key] = signature
        # Preserve alias relationships across slots, in addition to those inside
        # each independently verified Shared layer.
        baseline = None
        for block in range(count):
            aliases: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for slot, root in enumerate(roots[block * width : (block + 1) * width]):
                for parameter in self.families[root][1].parameters:
                    aliases[self.index.aliases[parameter.parameter_id]].append(
                        (slot, parameter.role)
                    )
            alias_signature = sorted(sorted(v) for v in aliases.values())
            require(
                baseline is None or alias_signature == baseline,
                "Repeated body parameter alias correspondence.",
            )
            baseline = alias_signature


def validate_bodies(graph: r.ArchitectureGraph) -> None:
    if not any(rep.bodies for rep in graph.repetitions):
        return
    index = BodyIndex(graph)
    for rep in graph.repetitions:
        end = 0
        for body in rep.bodies or []:
            require(body.start >= end, "Repeated body overlapping intervals.")
            index.verify(rep, body)
            end = body.start + body.width * body.count


def annotate_bodies(graph: r.ArchitectureGraph, byte_limit: int) -> r.ArchitectureGraph:
    if not graph.templates:
        return graph
    index = BodyIndex(graph)
    repetitions = []
    remaining = max(100_000, len(graph.nodes) * 32)
    for rep in graph.repetitions:
        families = [
            index.families[i.node_id][0].id if i.node_id in index.families else None
            for i in rep.instances
        ]
        bodies = []
        if len(set(families) - {None}) < 2:
            repetitions.append(rep)
            continue
        start = 0
        while start < len(families) and remaining > 0:
            found = None
            # Prefer the smallest complete verified period. Homogeneous runs keep
            # the existing period-one projection, and exceptions remain explicit.
            for width in range(2, (len(families) - start) // 2 + 1):
                remaining -= width
                if remaining <= 0:
                    break
                slots = families[start : start + width]
                if None in slots or len(set(slots)) < 2:
                    continue
                count = 1
                while families[start + count * width : start + (count + 1) * width] == slots:
                    count += 1
                if count < 2:
                    continue
                root = index.index.nodes[rep.instances[start].node_id]
                while count >= 2 and remaining > 0:
                    for input_port in (p.id for p in root.ports if p.direction == "input"):
                        for output_port in (p.id for p in root.ports if p.direction == "output"):
                            remaining -= sum(
                                len(index.families[i.node_id][1].nodes)
                                for i in rep.instances[start : start + width * count]
                            )
                            candidate = r.ArchitectureRepeatedBody(
                                start=start,
                                width=width,
                                count=count,
                                slots=[s for s in slots if s is not None],
                                ranges=ranges([s for s in slots if s is not None]),
                                input_port=input_port,
                                output_port=output_port,
                            )
                            try:
                                index.verify(rep, candidate)
                            except GraphError:
                                continue
                            found = candidate
                            break
                        if found:
                            break
                    if found:
                        break
                    count -= 1
                if found:
                    break
            if found:
                bodies.append(found)
                start += found.width * found.count
            else:
                start += 1
        repetitions.append(rep.model_copy(update={"bodies": bodies}) if bodies else rep)
    result = graph.model_copy(update={"repetitions": repetitions})
    try:
        serialized_size(result.document(), byte_limit)
    except GraphError:
        return graph
    return result
