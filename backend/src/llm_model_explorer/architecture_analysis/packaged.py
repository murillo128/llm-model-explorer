"""Publish a packaged semantic definition through the shared graph/binding validator."""

from typing import Any

from pydantic import TypeAdapter

from . import records as r
from .core import GraphBuilder
from .semantic import source_key


def publish_definition(definition: dict[str, Any], builder: GraphBuilder) -> None:
    provenance = builder.producer.provenance()
    shapes: TypeAdapter[r.ArchitectureShape] = TypeAdapter(r.ArchitectureShape)
    nodes: TypeAdapter[r.ArchitectureNode] = TypeAdapter(r.ArchitectureNode)
    for symbol in definition["symbols"]:
        builder.add_symbol(symbol["name"], symbol["meaning"])
    parameter_ids = {
        p["id"]: builder.native_parameter(
            p["id"],
            p["name"],
            shapes.validate_python(p["shape"]),
            provenance
            + [r.ArchitectureProvenance.model_validate(v) for v in p.get("provenance", [])]
            + [r.ArchitectureProvenance(kind="storage", source=p["name"])],
        )
        for p in definition["parameters"]
    }
    node_ids = {n["id"]: builder.record_id("node", n["id"]) for n in definition["nodes"]}
    for original in definition["nodes"]:
        node = dict(original)
        key = node["id"]
        node["id"] = node_ids[key]
        node["parameter_ids"] = [parameter_ids[p] for p in node.get("parameter_ids", [])]
        node["references"] = [
            {**ref, "parameter_id": parameter_ids[ref["parameter_id"]]}
            if ref["kind"] == "parameter"
            else ref
            for ref in node.get("references", [])
        ]
        node["attributes"] = [
            {**a, "provenance": a.get("provenance", provenance)} for a in node.get("attributes", [])
        ]
        node["provenance"] = [
            *provenance,
            *[
                p
                for p in node.get("provenance", [])
                if p.get("rule") != "Semantic source key in the reviewed packaged description"
            ],
            source_key(builder.producer, key),
        ]
        if "parent_id" in node:
            node["parent_id"] = node_ids[node["parent_id"]]
        if "children" in node:
            node["children"] = [node_ids[c] for c in node["children"]]
        semantic_key = next(
            (
                p["source"]
                for p in original.get("provenance", [])
                if p.get("rule") == "Semantic source key in the reviewed packaged description"
            ),
            key,
        )
        builder.add_node(nodes.validate_python(node), semantic_key=semantic_key)
        role = next((a["value"] for a in node["attributes"] if a["name"] == "semantic_role"), None)
        if role in ("layer", "attention", "mlp") and node["kind"] == "group":
            builder.templates.begin(node["id"], semantic_key, role, role)
    for original in definition["edges"]:
        edge = dict(original)
        edge["id"] = builder.record_id("edge", edge["id"])
        edge["source"] = {**edge["source"], "node_id": node_ids[edge["source"]["node_id"]]}
        edge["target"] = {**edge["target"], "node_id": node_ids[edge["target"]["node_id"]]}
        edge["provenance"] = provenance
        builder.add_edge(r.ArchitectureEdge.model_validate(edge))
    for original in definition["repetitions"]:
        repetition = dict(original)
        repetition["id"] = builder.record_id("repetition", repetition["id"])
        repetition["parent_id"] = node_ids[repetition["parent_id"]]
        repetition["instances"] = [
            {**i, "node_id": node_ids[i["node_id"]]} for i in repetition["instances"]
        ]
        builder.add_repetition(r.ArchitectureRepetition.model_validate(repetition))
