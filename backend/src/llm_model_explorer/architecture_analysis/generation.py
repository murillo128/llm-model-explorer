"""Authored greedy full-sequence orchestration, never an inference implementation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from . import records as r
from .validation import require

if TYPE_CHECKING:
    from .core import GraphBuilder

ROLE = "autoregressive_generation"
STATE = "generation_sequence_state"
PREPARE = "generation_prepare_inputs"
SELECT = "generation_greedy_next_token"
APPEND = "generation_append_token"


@dataclass(frozen=True)
class GenerationInterface:
    """Reviewed passive declarations to materialize as one single-pass boundary."""

    model: str
    tokens: str
    positions: str
    mask: str
    logits: str


def wrap_generation(
    graph: r.ArchitectureGraph, builder: "GraphBuilder", interface: GenerationInterface
) -> r.ArchitectureGraph:
    nodes = {n.id: n for n in graph.nodes}

    def nid(key: str) -> str:
        return builder.record_id("node", key)

    model = nodes[nid(interface.model)]
    require(isinstance(model, r.ArchitectureGroupNode), "Generation requires a model group.")
    assert isinstance(model, r.ArchitectureGroupNode)
    declarations = [
        nodes[nid(key)]
        for key in (interface.tokens, interface.positions, interface.mask, interface.logits)
    ]
    ids, positions, mask, logits = [n.ports[0].shape for n in declarations]
    require(ids is not None and len(ids) == 2, "Generation requires rank-two token IDs.")
    assert ids is not None
    length = ids[1]
    require(
        isinstance(length, r.ArchitectureSymbolDimension), "Generation requires a sequence symbol."
    )
    assert isinstance(length, r.ArchitectureSymbolDimension)
    initial = [ids[0], r.ArchitectureSymbolDimension(kind="symbol", name="S_initial")]
    final = [ids[0], r.ArchitectureSymbolDimension(kind="symbol", name="S_final")]
    updated = [
        ids[0],
        r.ArchitectureExpressionDimension(
            kind="expression", text=f"{length.name} + 1", symbols=[length.name]
        ),
    ]
    selected = [ids[0], r.ArchitectureConstantDimension(kind="constant", value=1)]
    provenance = [
        r.ArchitectureProvenance(
            kind="description",
            source="greedy-no-cache-generation",
            revision="1",
            rule=(
                "Chosen explanatory policy: unpadded single sequence, full-sequence calls; "
                "not checkpoint generation defaults."
            ),
        )
    ]

    def attr(name: str, value: str) -> r.ArchitectureAttribute:
        return r.ArchitectureAttribute(name=name, value=value, provenance=provenance)

    def port(
        id: str,
        label: str,
        shape: r.ArchitectureShape,
        direction: Literal["input", "output"] = "input",
    ) -> r.ArchitecturePort:
        return r.ArchitecturePort(id=id, label=label, direction=direction, shape=shape)

    generation = nid("generation")
    state = nid("generation.sequence")
    prepare = nid("generation.prepare")
    select = nid("generation.select")
    append = nid("generation.append")
    new_nodes: list[r.ArchitectureNode] = [
        r.ArchitectureGroupNode(
            id=generation,
            kind="group",
            label="Generation",
            children=[state, prepare, model.id, select, append],
            ports=[
                port("prompt_ids", "prompt_ids", initial),
                port("token_ids", "token_ids[T]", final, "output"),
            ],
            attributes=[attr("semantic_role", ROLE), attr("policy", "greedy_no_cache")],
            parameter_ids=[],
            references=[],
            provenance=provenance,
        ),
        r.ArchitectureLeafNode(
            id=state,
            parent_id=generation,
            kind="state",
            label="Sequence state",
            operation=STATE,
            description=(
                "initial binds prompt as current at t=0; next commits appended IDs at t+1; "
                "final reads the terminal sequence including prompt. No KV cache; phases are "
                "alternatives in time."
            ),
            ports=[
                port("initial", "token_ids[0]", initial),
                port("current", "token_ids[t]", ids, "output"),
                port("next", "token_ids[t+1]", updated),
                port("final", "token_ids[T]", final, "output"),
            ],
            attributes=[attr("semantic_role", STATE)],
            parameter_ids=[],
            references=[],
            provenance=provenance,
        ),
        r.ArchitectureLeafNode(
            id=prepare,
            parent_id=generation,
            kind="operation",
            label="Prepare inputs",
            operation=PREPARE,
            formula="token_ids[t], positions[t], causal_mask[t] = prepare_inputs(token_ids[t])",
            description=(
                "Preserve the entire integer sequence; derive integer positions 0..S-1 and "
                "the additive causal mask [B,1,S,S] for this full no-cache call."
            ),
            ports=[
                port("sequence", "token_ids[t]", ids),
                port("tokens", "token_ids[t]", ids, "output"),
                port("positions", "positions[t]", positions, "output"),
                port("mask", "causal_mask[t]", mask, "output"),
            ],
            attributes=[
                attr("semantic_role", PREPARE),
                attr("convention", "unpadded_full_sequence"),
            ],
            parameter_ids=[],
            references=[],
            provenance=provenance,
        ),
        r.ArchitectureLeafNode(
            id=select,
            parent_id=generation,
            kind="operation",
            label="Next token",
            operation=SELECT,
            formula="next_token_id[t] = argmax(logits[t][:, -1, :], axis=-1, keepdims=true)",
            description=(
                "Unpadded single-sequence illustration: -1 is the last valid sequence "
                "position; the vocabulary index is already a token ID."
            ),
            ports=[
                port("logits", "logits[t]", logits),
                port("token", "next_token_id[t]", selected, "output"),
            ],
            attributes=[attr("semantic_role", SELECT)],
            parameter_ids=[],
            references=[],
            provenance=provenance,
        ),
        r.ArchitectureLeafNode(
            id=append,
            parent_id=generation,
            kind="operation",
            label="Append token",
            operation=APPEND,
            formula="token_ids[t+1] = concat(token_ids[t], next_token_id[t], axis=1)",
            ports=[
                port("sequence", "token_ids[t]", ids),
                port("token", "next_token_id[t]", selected),
                port("updated", "token_ids[t+1]", updated, "output"),
            ],
            attributes=[attr("semantic_role", APPEND)],
            parameter_ids=[],
            references=[],
            provenance=provenance,
        ),
    ]
    # Passive interface declarations become actual ports. Neural operations and
    # all original edge identities remain, including every layer/adapter binding.
    removed = {n.id for n in declarations}
    boundary = dict(
        zip([n.id for n in declarations], ["tokens", "positions", "mask", "logits"], strict=True)
    )
    model_ports = [
        port("tokens", "token_ids", ids),
        port("positions", "positions", positions),
        port("mask", "causal_mask", mask),
        port("logits", "logits", logits, "output"),
    ]
    for node in graph.nodes:
        if node.id in removed:
            continue
        if node.id == model.id:
            node = node.model_copy(
                update={
                    "parent_id": generation,
                    "ports": model_ports,
                    "children": [id for id in model.children if id not in removed],
                }
            )
        new_nodes.append(node)
    edges = []
    for edge in graph.edges:

        def endpoint(ep: r.ArchitectureEndpoint) -> r.ArchitectureEndpoint:
            return (
                r.ArchitectureEndpoint(node_id=model.id, port_id=boundary[ep.node_id])
                if ep.node_id in removed
                else ep
            )

        edges.append(
            edge.model_copy(
                update={"source": endpoint(edge.source), "target": endpoint(edge.target)}
            )
        )

    def connect(
        source: str, sp: str, target: str, tp: str, kind: Literal["data", "state"] = "data"
    ) -> None:
        edges.append(
            r.ArchitectureEdge(
                id=builder.record_id("edge", f"generation:{source}:{sp}:{target}:{tp}"),
                source=r.ArchitectureEndpoint(node_id=source, port_id=sp),
                target=r.ArchitectureEndpoint(node_id=target, port_id=tp),
                kind=kind,
                provenance=provenance,
            )
        )

    connect(generation, "prompt_ids", state, "initial")
    connect(state, "current", prepare, "sequence", "state")
    for role in ("tokens", "positions", "mask"):
        connect(prepare, role, model.id, role)
    connect(model.id, "logits", select, "logits")
    connect(prepare, "tokens", append, "sequence")
    connect(select, "token", append, "token")
    connect(append, "updated", state, "next", "state")
    connect(state, "final", generation, "token_ids", "state")
    return graph.model_copy(
        update={
            "nodes": new_nodes,
            "edges": edges,
            "symbols": [
                *graph.symbols,
                r.ArchitectureSymbol(
                    name="S_initial", meaning="Prompt sequence length at generation step zero."
                ),
                r.ArchitectureSymbol(
                    name="S_final", meaning="Terminal sequence length, including the prompt."
                ),
            ],
        }
    )


def validate_generation(graph: r.ArchitectureGraph) -> None:
    """Recognize the finite pattern at import/preparation, never by display labels."""
    nodes = {node.id: node for node in graph.nodes}
    incident: dict[str, list[r.ArchitectureEdge]] = {}
    for edge in graph.edges:
        for id in {edge.source.node_id, edge.target.node_id}:
            incident.setdefault(id, []).append(edge)

    def attributes(node: r.ArchitectureNode) -> dict[str, object]:
        return {a.name: a.value for a in node.attributes}

    owners = [node for node in graph.nodes if attributes(node).get("semantic_role") == ROLE]
    states = {
        node.id
        for node in graph.nodes
        if node.operation == STATE or attributes(node).get("semantic_role") == STATE
    }
    recognized: set[str] = set()
    for owner in owners:
        require(isinstance(owner, r.ArchitectureGroupNode), "Generation owner must be a group.")
        assert isinstance(owner, r.ArchitectureGroupNode)
        require(
            attributes(owner).get("policy") == "greedy_no_cache", "Unsupported generation policy."
        )
        children = [nodes[id] for id in owner.children]
        roles = {}
        for operation in (STATE, PREPARE, SELECT, APPEND):
            matches = [n for n in children if n.operation == operation]
            require(
                len(matches) == 1, "Generation requires one sequence state and three operations."
            )
            roles[operation] = matches[0]
            require(
                attributes(matches[0]).get("semantic_role") == operation,
                "Generation role disagrees with operation.",
            )
        state, prepare, select, append = [roles[role] for role in (STATE, PREPARE, SELECT, APPEND)]
        require(
            state.kind == "state" and all(n.kind == "operation" for n in (prepare, select, append)),
            "Invalid generation record kinds.",
        )
        model_nodes = [n for n in children if n.id not in {v.id for v in roles.values()}]
        require(
            len(model_nodes) == 1 and model_nodes[0].kind == "group",
            "Generation requires one single-pass model.",
        )
        model = model_nodes[0]
        recognized.add(state.id)
        expected_ports = {
            owner.id: {"prompt_ids": "input", "token_ids": "output"},
            state.id: {"initial": "input", "current": "output", "next": "input", "final": "output"},
            prepare.id: {
                "sequence": "input",
                "tokens": "output",
                "positions": "output",
                "mask": "output",
            },
            select.id: {"logits": "input", "token": "output"},
            append.id: {"sequence": "input", "token": "input", "updated": "output"},
            model.id: {
                "tokens": "input",
                "positions": "input",
                "mask": "input",
                "logits": "output",
            },
        }
        ports = {
            n.id: {p.id: p for p in n.ports} for n in (owner, state, prepare, select, append, model)
        }
        for id, expected in expected_ports.items():
            require(
                {key: p.direction for key, p in ports[id].items()} == expected,
                "Invalid generation port roles.",
            )
        current = ports[state.id]["current"].shape
        require(
            current is not None
            and len(current) == 2
            and isinstance(current[1], r.ArchitectureSymbolDimension),
            "Generation current IDs require batch and current sequence dimensions.",
        )
        assert current is not None and isinstance(current[1], r.ArchitectureSymbolDimension)
        length = current[1].name
        require(
            ports[state.id]["next"].shape
            == [
                current[0],
                r.ArchitectureExpressionDimension(
                    kind="expression", text=f"{length} + 1", symbols=[length]
                ),
            ],
            "Generation next phase must carry the enlarged sequence.",
        )
        for role in ("initial", "final"):
            dims = ports[state.id][role].shape
            require(
                dims is not None
                and len(dims) == 2
                and dims[0] == current[0]
                and isinstance(dims[1], r.ArchitectureSymbolDimension)
                and dims[1] != current[1],
                "Generation phase lengths must remain distinct.",
            )
        require(
            ports[state.id]["initial"].shape != ports[state.id]["final"].shape,
            "Generation initial/final lengths must remain distinct.",
        )
        require(
            ports[prepare.id]["tokens"].shape == current,
            "Preparation must preserve the entire sequence.",
        )
        require(
            ports[select.id]["token"].shape
            == [current[0], r.ArchitectureConstantDimension(kind="constant", value=1)],
            "Selection must yield one vocabulary ID per sequence.",
        )
        logits = ports[select.id]["logits"].shape
        require(
            logits is not None and len(logits) == 3 and logits[:2] == current,
            "Generation requires full-sequence vocabulary logits.",
        )
        require(
            ports[prepare.id]["positions"].shape == current
            and ports[prepare.id]["mask"].shape
            == [
                current[0],
                r.ArchitectureConstantDimension(kind="constant", value=1),
                current[1],
                current[1],
            ],
            "Unsupported generation position/mask convention.",
        )
        expected_edges = {
            (owner.id, "prompt_ids", state.id, "initial", "data"),
            (state.id, "current", prepare.id, "sequence", "state"),
            (prepare.id, "tokens", model.id, "tokens", "data"),
            (prepare.id, "positions", model.id, "positions", "data"),
            (prepare.id, "mask", model.id, "mask", "data"),
            (model.id, "logits", select.id, "logits", "data"),
            (prepare.id, "tokens", append.id, "sequence", "data"),
            (select.id, "token", append.id, "token", "data"),
            (append.id, "updated", state.id, "next", "state"),
            (state.id, "final", owner.id, "token_ids", "state"),
        }
        scope = set(ports)
        require(
            all(
                e.source.node_id in scope and e.target.node_id in scope
                for node in (state, prepare, select, append)
                for e in incident.get(node.id, [])
            ),
            "Generation orchestration must connect through the declared model boundary.",
        )
        actual = list(
            {
                e.id: e
                for id in scope
                for e in incident.get(id, [])
                if e.source.node_id in scope and e.target.node_id in scope
            }.values()
        )
        require(
            len(actual) == len(expected_edges)
            and {
                (e.source.node_id, e.source.port_id, e.target.node_id, e.target.port_id, e.kind)
                for e in actual
            }
            == expected_edges,
            "Generation sequence dependencies or phase mappings are invalid.",
        )
        for e in actual:
            require(
                ports[e.source.node_id][e.source.port_id].shape
                == ports[e.target.node_id][e.target.port_id].shape,
                (
                    "Connected generation shapes must agree exactly; length changes belong to "
                    "state phases."
                ),
            )
    require(states == recognized, "Sequence state must belong to a declared generation pattern.")
