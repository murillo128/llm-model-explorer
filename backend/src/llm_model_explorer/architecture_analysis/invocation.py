"""Source-reviewed no-cache specializations of the retained low-level descriptions.

Only the three enumerated descriptions opt in. Keys below are authored source
keys, not inferred labels or user graph patterns. No tensor values are produced.
"""

from typing import TYPE_CHECKING, Literal

from . import records as r
from .validation import require

if TYPE_CHECKING:
    from .core import GraphBuilder

Invocation = Literal["deepseek", "glm", "kimi"]


def no_cache_invocation(
    graph: r.ArchitectureGraph, builder: "GraphBuilder", family: Invocation
) -> r.ArchitectureGraph:
    def nid(key: str) -> str:
        return builder.record_id("node", key)

    nodes = {n.id: n for n in graph.nodes}
    removed: set[str] = set()
    dropped: set[tuple[str, str]] = set()
    bypass: dict[str, r.ArchitectureEndpoint] = {}
    local_initializers: set[str] = set()

    def remove(key: str) -> None:
        identity = nid(key)
        require(identity in nodes, "Missing reviewed cache bookkeeping node.")
        removed.add(identity)

    def ports(key: str, names: list[str]) -> None:
        identity = nid(key)
        actual = {p.id for p in nodes[identity].ports}
        require(set(names) <= actual, "Missing reviewed cache boundary role.")
        dropped.update((identity, name) for name in names)

    def current(key: str, port: str) -> None:
        identity = nid(key)
        incoming = [
            e for e in graph.edges if e.target.node_id == identity and e.target.port_id == port
        ]
        require(len(incoming) == 1, "No-cache specialization needs one exact current value.")
        bypass[identity] = incoming[0].source
        remove(key)

    layers = next(rep for rep in graph.repetitions if rep.parent_id == nid("model"))
    for instance in layers.instances:
        index = instance.index
        layer = f"model.layers.{index}"
        attention = layer + ".self_attn"
        if family == "kimi" and instance.variant.startswith("kda"):
            roles = ["q_conv", "k_conv", "v_conv", "recurrent"]
            for role in roles:
                key = (
                    f"model.kda_state.prior_{role}_state.{index}"
                    if role != "recurrent"
                    else f"model.kda_state.recurrent.{index}"
                )
                remove(key)
            names = [f"{phase}_{role}_state" for phase in ("prior", "next") for role in roles]
            ports(layer, names)
            ports(attention, names)
            for branch in ("q", "k", "v"):
                key = f"{attention}.{branch}_conv1d"
                ports(key, ["prior_state", "next_state"])
                local_initializers.add(nid(key))
            key = attention + ".kda_delta_update"
            ports(key, ["prior_state"])
            local_initializers.add(nid(key))
        else:
            roles = ["prior_key_state", "prior_value_state", "next_key_state", "next_value_state"]
            ports(layer, roles)
            ports(attention, roles)
            if family == "deepseek":
                for role in ("key", "value"):
                    remove(f"model.layer_state.{index}.prior_{role}")
                    current(f"{attention}.{role}_cache_update", f"current_{role}")
                    remove(f"{attention}.next_{role}_state")
            elif family == "glm":
                remove(f"model.layer_state.{index}.prior_kv_latent")
                remove(f"model.layer_state.{index}.prior_rotary_key")
                for role in ("kv_latent", "rotary_key"):
                    current(f"{attention}.{role}_cache_update", "current_state")
            else:
                for role in ("key", "value"):
                    remove(f"model.mla_state.{role}.{index}")
                    current(f"{attention}.{role}_cache_update", "current")

    if family == "deepseek":
        banks = ["key", "value"]
        ports(
            "model", [f"{phase}_key_values_{role}" for phase in ("past", "next") for role in banks]
        )
        for role in banks:
            remove(f"model.next_{role}_values")
    elif family == "glm":
        banks = ["kv_latent", "rotary_key"]
        ports(
            "model", [f"{phase}_key_values_{role}" for phase in ("past", "next") for role in banks]
        )
        for role in banks:
            remove(f"model.next_{role}_values")
    else:
        banks = ["kda_q_conv", "kda_k_conv", "kda_v_conv", "kda_recurrent", "mla_key", "mla_value"]
        ports("model", [f"{phase}_{role}_state" for phase in ("prior", "next") for role in banks])
        for role in banks:
            remove(f"model.next_{role}_state")

    def shape(value: r.ArchitectureShape) -> r.ArchitectureShape:
        if value is None:
            return None
        result: list[r.ArchitectureDimension] = []
        for d in value:
            if (
                isinstance(d, r.ArchitectureExpressionDimension)
                and d.text == "K + S"
                and d.symbols == ["K", "S"]
            ):
                d = r.ArchitectureSymbolDimension(kind="symbol", name="S")
            result.append(d)
        return result

    output_nodes: list[r.ArchitectureNode] = []
    for node in graph.nodes:
        if node.id in removed:
            require(not node.parameter_ids, "Cannot remove a neural parameter operation.")
            continue
        update: dict[str, object] = {
            "ports": [
                p.model_copy(update={"shape": shape(p.shape)})
                for p in node.ports
                if (node.id, p.id) not in dropped
            ]
        }
        if node.kind == "group":
            update["children"] = [c for c in node.children if c not in removed]
        if node.id in local_initializers:
            update["attributes"] = [
                *node.attributes,
                r.ArchitectureAttribute(
                    name="initialization",
                    value="absent_per_call",
                    provenance=builder.producer.provenance(),
                ),
            ]
            update["description"] = (
                "Full-sequence call with cache_params=None: convolution cache and KDA "
                "initial_state are absent; internal causal recurrence remains."
            )
            if node.operation == "short_convolution":
                update["formula"] = (
                    "out = causal_short_convolution(x, weight, sequence_offsets; cache=None)"
                )

        if node.id == nid("model"):
            update["attributes"] = [
                *node.attributes,
                r.ArchitectureAttribute(
                    name="invocation",
                    value="full_sequence_use_cache_false_past_none",
                    provenance=builder.producer.provenance(),
                ),
            ]
        output_nodes.append(node.model_copy(update=update))
    edges = []
    for edge in graph.edges:
        if edge.target.node_id in removed or (edge.target.node_id, edge.target.port_id) in dropped:
            continue
        if (edge.source.node_id, edge.source.port_id) in dropped:
            require(
                (edge.target.node_id, edge.target.port_id) in dropped,
                "Removed cache input still feeds computation.",
            )
            continue
        source = bypass.get(edge.source.node_id, edge.source)
        require(source.node_id not in removed, "Removed cache node still feeds computation.")
        edges.append(
            edge.model_copy(
                update={
                    "source": source,
                    **({"kind": "data"} if edge.source.node_id in bypass else {}),
                }
            )
        )
    return graph.model_copy(update={"nodes": output_nodes, "edges": edges})
