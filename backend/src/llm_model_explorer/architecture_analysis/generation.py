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
    positions: str | None
    mask: str
    logits: str
    current_mask: str | None = None
    convention: Literal[
        "unpadded_full_sequence", "qwen35_text_full_sequence", "kimi_full_sequence"
    ] = "unpadded_full_sequence"
    boundary: bool = False


def wrap_generation(
    graph: r.ArchitectureGraph, builder: "GraphBuilder", interface: GenerationInterface
) -> r.ArchitectureGraph:
    nodes = {n.id: n for n in graph.nodes}

    def nid(key: str) -> str:
        return builder.record_id("node", key)

    model = nodes[nid(interface.model)]
    require(isinstance(model, r.ArchitectureGroupNode), "Generation requires a model group.")
    assert isinstance(model, r.ArchitectureGroupNode)
    bindings = {
        role: key
        for role, key in {
            "tokens": interface.tokens,
            "positions": interface.positions,
            "mask": interface.mask,
            "logits": interface.logits,
            "current_mask": interface.current_mask,
        }.items()
        if key is not None
    }
    declarations = (
        {} if interface.boundary else {role: nodes[nid(key)] for role, key in bindings.items()}
    )
    shapes = (
        {role: next(p.shape for p in model.ports if p.id == key) for role, key in bindings.items()}
        if interface.boundary
        else {role: node.ports[0].shape for role, node in declarations.items()}
    )
    ids, logits = shapes["tokens"], shapes["logits"]
    prepared = {role: shape for role, shape in shapes.items() if role != "logits"}
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
            revision="2",
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

    labels = {
        role: "token_ids"
        if role == "tokens"
        else "causal_mask"
        if role == "mask" and interface.convention != "qwen35_text_full_sequence"
        else role
        for role in prepared
    }
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
            formula=", ".join(f"{labels[role]}[t]" for role in prepared)
            + " = prepare_inputs(token_ids[t])",
            description=(
                "Preserve the entire integer sequence; derive integer positions 0..S-1 and "
                "the additive causal mask [B,1,S,S] for this full no-cache call."
                if interface.convention == "unpadded_full_sequence"
                else "Preserve the entire integer sequence. "
                + (
                    "Derive text THW positions by repeating 0..T-1 on three axes, an all-valid "
                    "attention padding mask and a separate current-sequence padding mask. "
                    "The model derives causal attention internally; no prior KV cache."
                    if interface.convention == "qwen35_text_full_sequence"
                    else "Derive the additive causal mask and independent all-valid padding mask. "
                    "KDA starts with absent convolution/recurrent cache; MLA has no prior KV. "
                    "This reviewed MLA path uses no positional encoding."
                    if interface.convention == "kimi_full_sequence"
                    else "Derive integer positions 0..S-1 and the additive causal mask [B,1,S,S]."
                )
            ),
            ports=[
                port("sequence", "token_ids[t]", ids),
                *[
                    port(role, f"{labels[role]}[t]", shape, "output")
                    for role, shape in prepared.items()
                ],
            ],
            attributes=[
                attr("semantic_role", PREPARE),
                attr("convention", interface.convention),
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
    removed = {n.id for n in declarations.values()}
    boundary = {node.id: role for role, node in declarations.items()}
    boundary_ports = (
        {key: role for role, key in bindings.items()} if interface.boundary else {"out": "logits"}
    )
    model_ports = [
        *[port(role, labels[role], shape) for role, shape in prepared.items()],
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
                else r.ArchitectureEndpoint(node_id=model.id, port_id=boundary_ports[ep.port_id])
                if ep.node_id == model.id and ep.port_id in boundary_ports
                else ep
            )

        source, target = endpoint(edge.source), endpoint(edge.target)
        # Existing true group forwarding already crosses this passive declaration.
        if source == target:
            continue
        edges.append(edge.model_copy(update={"source": source, "target": target}))

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
    for role in prepared:
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
        convention = attributes(prepare).get("convention")
        require(
            convention
            in {"unpadded_full_sequence", "qwen35_text_full_sequence", "kimi_full_sequence"},
            "Unsupported generation preparation convention.",
        )
        prepared_roles = ["tokens", "positions", "mask"]
        if convention == "qwen35_text_full_sequence":
            prepared_roles.append("current_mask")
        elif convention == "kimi_full_sequence":
            prepared_roles = ["tokens", "mask", "current_mask"]
        expected_ports = {
            owner.id: {"prompt_ids": "input", "token_ids": "output"},
            state.id: {"initial": "input", "current": "output", "next": "input", "final": "output"},
            prepare.id: {
                "sequence": "input",
                **dict.fromkeys(prepared_roles, "output"),
            },
            select.id: {"logits": "input", "token": "output"},
            append.id: {"sequence": "input", "token": "input", "updated": "output"},
            model.id: {
                **dict.fromkeys(prepared_roles, "input"),
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
        expected_shapes = {
            "positions": current,
            "mask": [
                current[0],
                r.ArchitectureConstantDimension(kind="constant", value=1),
                current[1],
                current[1],
            ],
        }
        if convention == "qwen35_text_full_sequence":
            expected_shapes = {
                "positions": [r.ArchitectureConstantDimension(kind="constant", value=3), *current],
                "mask": current,
                "current_mask": current,
            }
        elif convention == "kimi_full_sequence":
            expected_shapes.pop("positions")
            expected_shapes["current_mask"] = current
        require(
            all(ports[prepare.id][role].shape == shape for role, shape in expected_shapes.items()),
            "Unsupported generation position/mask convention.",
        )
        pending = [model.id]
        while pending:
            member = nodes[pending.pop()]
            if isinstance(member, r.ArchitectureGroupNode):
                pending.extend(member.children)
            attrs = attributes(member)
            initializers = {
                "initial_delta_state": "zeros_per_call",
                "short_convolution": "absent_per_call",
                "kda_delta_state_update": "absent_per_call",
            }
            if member.operation in initializers:
                require(
                    attrs.get("initialization") == initializers[member.operation],
                    "Generation initializer disagrees with reviewed invocation.",
                )
            require(
                member.operation
                not in {
                    "prior_kv",
                    "prior_delta_state",
                    "prior_convolution_state",
                    "state_concat",
                    "select_layer_state",
                    "stack_layer_states",
                },
                "Generation contains conditional cross-call cache bookkeeping.",
            )
        expected_edges = {
            (owner.id, "prompt_ids", state.id, "initial", "data"),
            (state.id, "current", prepare.id, "sequence", "state"),
            *((prepare.id, role, model.id, role, "data") for role in prepared_roles),
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
