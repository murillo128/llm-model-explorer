"""Portable Shared declarations must preserve the complete, instance-bound graph."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest

from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.core import AnalysisInput
from llm_model_explorer.architecture_analysis.model_defined import (
    analyze_definition,
    parse_definition,
    producer_for,
)
from llm_model_explorer.architecture_analysis.model_defined_schema import definition_schema
from llm_model_explorer.architecture_analysis.validation import (
    BindingContext,
    GraphError,
    NumericTensor,
    serialized_size,
    validate_graph,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/model-owned-architecture/shared-architecture.json"


def document() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(EXAMPLE.read_text()))


def inventory() -> AnalysisInput:
    # Deliberately independent of the example/importer's parameter declarations.
    physical = {}
    numeric = {}
    for layer in range(2):
        entries = [(f"attn.{p}", [4, 4]) for p in ("q", "k", "v", "o")]
        entries += [("mlp.up", [8, 4]), ("mlp.down", [4, 8])]
        for suffix, shape in entries:
            name = f"blocks.{layer}.{suffix}.weight"
            tensor_id = f"tensor-{layer}-{suffix}"
            physical[name] = r.ArchitectureStorage(name=name, dtype="F32", shape=shape)
            numeric[tensor_id] = NumericTensor(tensor_id, name, tuple(shape), "F32")
    return AnalysisInput("shared-example-checkpoint", {}, BindingContext(physical, numeric, False))


def analyze(value: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    definition = parse_definition(json.dumps(document() if value is None else value).encode())
    return analyze_definition(definition, inventory(), **kwargs)


def test_imports_attention_and_mlp_with_exact_nonzero_bindings() -> None:
    result = analyze()
    assert result.status == "complete", result.diagnostics
    graph = result.graph
    assert graph is not None
    validate_graph(graph, inventory().bindings)
    assert [t.label for t in graph.templates] == ["Dense attention", "GELU MLP"]
    assert [i.index for i in graph.repetitions[0].instances] == [0, 1]
    params = {p.id: p for p in graph.parameters}
    attention = graph.templates[0]
    for layer, instance in enumerate(attention.instances):
        mapped = {p.role: params[p.parameter_id] for p in instance.parameters}
        assert set(mapped) == {f"{p}.weight" for p in ("q", "k", "v", "o")}
        for projection in ("q", "k", "v", "o"):
            parameter = mapped[f"{projection}.weight"]
            assert parameter.name == f"blocks.{layer}.attn.{projection}.weight"
            assert parameter.inspection.tensor_id == f"tensor-{layer}-attn.{projection}"
    assert graph.scope == "model_defined"
    notice = next(n for n in graph.nodes if n.label == "Model-supplied definition")
    assert {a.name: a.value for a in notice.attributes}["semantic_verification"] == "not_verified"
    assert all(t.revision == "shared-example-v1" for t in graph.templates)
    assert all("not verified" in t.provenance[0].rule for t in graph.templates)
    local = {n["id"] for n in document()["nodes"]}
    assert not local.intersection(n.id for n in graph.nodes)
    assert graph.document() == analyze().graph.document()


def test_absence_and_empty_templates_preserve_the_ordinary_graph() -> None:
    value = document()
    del value["templates"]
    without = analyze(value).graph.document()
    value["templates"] = []
    assert analyze(value).graph.document() == without
    with_templates = analyze().graph.document()
    del with_templates["templates"]
    assert with_templates == without


def test_schema_generation_and_both_example_shapes() -> None:
    schema = definition_schema()
    assert schema == json.loads(
        (ROOT / "docs/spec/backend/architecture-definition.schema.json").read_text()
    )
    assert schema["properties"]["schema_version"]["const"] == 1
    assert "templates" not in schema["required"]
    assert schema["$defs"]["DefinitionTemplate"]["additionalProperties"] is False
    assert schema["$defs"]["DefinitionTemplate"]["properties"]["instances"]["minItems"] == 2
    assert parse_definition(EXAMPLE.read_bytes()).schema_version == 1


@pytest.mark.parametrize("field", ["provenance", "revision", "shared", "count", "verified"])
def test_author_cannot_supply_verification_or_runtime_envelope(field: str) -> None:
    value = document()
    value["templates"][0][field] = "untrusted"
    with pytest.raises(GraphError):
        parse_definition(json.dumps(value).encode())


@pytest.mark.parametrize("edit", ["role", "singleton", "null", "foreign_instance_field"])
def test_closed_template_schema(edit: str) -> None:
    value = document()
    family = value["templates"][0]
    if edit == "role":
        family["component_role"] = "anything"
    elif edit == "singleton":
        family["instances"].pop()
    elif edit == "null":
        value["templates"] = None
    else:
        family["instances"][0]["tensor_id"] = "not-an-author-field"
    with pytest.raises(GraphError):
        parse_definition(json.dumps(value).encode())


@pytest.mark.parametrize(
    "edit",
    [
        "duplicate_family",
        "duplicate_component",
        "duplicate_role",
        "duplicate_target",
        "missing_node",
        "foreign_node",
        "missing_port",
        "foreign_port",
        "missing_edge",
        "foreign_edge",
        "missing_parameter",
        "foreign_parameter",
        "reverse_instances",
        "mismatched_formula",
        "mismatched_attribute",
        "wrong_semantic_role",
        "unknown_shape",
        "different_scope",
    ],
)
def test_invalid_declarations_fail_closed(edit: str) -> None:
    value = document()
    family = value["templates"][0]
    second = family["instances"][1]
    nodes = {n["id"]: n for n in value["nodes"]}
    if edit == "duplicate_family":
        value["templates"].append(copy.deepcopy(family))
    elif edit == "duplicate_component":
        other = copy.deepcopy(family)
        other["id"] = "same-components-different-family"
        value["templates"].append(other)
    elif edit == "duplicate_role":
        second["nodes"][1]["role"] = second["nodes"][0]["role"]
    elif edit == "duplicate_target":
        second["nodes"][1]["node_id"] = second["nodes"][0]["node_id"]
    elif edit == "missing_node":
        second["nodes"].pop()
    elif edit == "foreign_node":
        second["nodes"][-1]["node_id"] = "a-attention-o"
    elif edit == "missing_port":
        second["ports"].pop()
    elif edit == "foreign_port":
        second["ports"][-1]["port_id"] = "absent"
    elif edit == "missing_edge":
        second["edges"].pop()
    elif edit == "foreign_edge":
        second["edges"][-1]["edge_id"] = "next-layer"
    elif edit == "missing_parameter":
        second["parameters"].pop()
    elif edit == "foreign_parameter":
        second["parameters"][0]["parameter_id"] = "a-attention-q-weight"
    elif edit == "reverse_instances":
        family["instances"].reverse()
    elif edit == "mismatched_formula":
        nodes["b-attention-scores"]["formula"] = "S = K Q^T"
    elif edit == "mismatched_attribute":
        nodes["b-attention"]["attributes"].append({"name": "heads", "value": 2})
    elif edit == "wrong_semantic_role":
        nodes["b-attention"]["attributes"][0]["value"] = "mlp"
    elif edit == "unknown_shape":
        nodes["b-attention-q"]["ports"][0]["shape"] = None
    else:
        # Without repetitions the components reside in independent layer scopes.
        value["repetitions"] = []
    result = analyze(value)
    assert result.status == "unavailable", edit
    assert result.diagnostics[0].code == "invalid_model_definition"


def test_unknown_leaf_and_incompatible_parameter_geometry_cannot_establish_equivalence() -> None:
    value = document()
    node = next(n for n in value["nodes"] if n["id"] == "b-attention-q")
    node["kind"] = "context"
    del node["operation"]
    assert analyze(value).status == "unavailable"
    value = document()
    parameter = next(p for p in value["parameters"] if p["id"] == "b-attention-q-weight")
    parameter["shape"][0]["value"] = 5
    assert analyze(value).status == "unavailable"


def test_partial_storage_keeps_real_binding_availability() -> None:
    source = inventory()
    physical = dict(source.bindings.physical)
    missing = "blocks.1.attn.k.weight"
    del physical[missing]
    numeric = {k: t for k, t in source.bindings.numeric.items() if t.name != missing}
    result = analyze_definition(
        parse_definition(EXAMPLE.read_bytes()),
        replace(source, bindings=BindingContext(physical, numeric, False)),
    )
    assert result.status == "partial"
    assert result.graph is not None and result.graph.templates is not None
    assert len(result.graph.templates) == 2
    parameter = next(p for p in result.graph.parameters if p.name == missing)
    assert parameter.inspection.status == "unavailable"
    assert parameter.binding == "unresolved" and parameter.storage == []


def test_nonconsecutive_repetition_and_sibling_order() -> None:
    value = document()
    value["repetitions"][0]["instances"][1]["index"] = 7
    assert analyze(value).status == "complete"
    # Make the two attention groups siblings; no name/repetition heuristics.
    value = document()
    value["nodes"] = [n for n in value["nodes"] if n["id"] not in {"layer-a", "layer-b"}]
    stack = next(n for n in value["nodes"] if n["id"] == "stack")
    stack["children"] = ["a-attention", "a-mlp", "b-attention", "b-mlp"]
    for n in value["nodes"]:
        if n.get("parent_id") in {"layer-a", "layer-b"}:
            n["parent_id"] = "stack"
    value["edges"] = [
        e
        for e in value["edges"]
        if e["source"]["node_id"] not in {"layer-a", "layer-b"}
        and e["target"]["node_id"] not in {"layer-a", "layer-b"}
    ]
    value["repetitions"] = []
    assert analyze(value).status == "complete"


def test_ids_do_not_require_paths_or_prefix_conventions() -> None:
    value = document()
    rename = {n["id"]: f"node{index}" for index, n in enumerate(value["nodes"])}
    for n in value["nodes"]:
        n["id"] = rename[n["id"]]
        if "parent_id" in n:
            n["parent_id"] = rename[n["parent_id"]]
        if "children" in n:
            n["children"] = [rename[c] for c in n["children"]]
    for edge in value["edges"]:
        for ep in (edge["source"], edge["target"]):
            ep["node_id"] = rename[ep["node_id"]]
    for repetition in value["repetitions"]:
        repetition["parent_id"] = rename[repetition["parent_id"]]
        for instance in repetition["instances"]:
            instance["node_id"] = rename[instance["node_id"]]
    for template in value["templates"]:
        for instance in template["instances"]:
            instance["node_id"] = rename[instance["node_id"]]
            for mapped in [*instance["nodes"], *instance["ports"]]:
                mapped["node_id"] = rename[mapped["node_id"]]
    assert analyze(value).status == "complete"


def test_valid_optional_metadata_budget_and_invalid_after_exhaustion(caplog: pytest.LogCaptureFixture) -> None:
    base = document()
    del base["templates"]
    graph = analyze(base).graph
    limit = serialized_size(graph.document()) + 250
    result = analyze(byte_limit=limit)
    assert result.status == "complete", result.diagnostics
    assert not result.graph.templates
    assert any(d.code == "templates_omitted" for d in result.graph.diagnostics)
    assert "metadata budget" in caplog.text
    assert serialized_size(result.graph.document(), limit) <= limit
    # A later malformed instance must fail even though no family fits the budget.
    value = document()
    value["templates"][1]["instances"][1]["parameters"].pop()
    assert analyze(value, byte_limit=limit).status == "unavailable"


def test_producer_revision_invalidates_pre_extension_cache() -> None:
    definition = parse_definition(EXAMPLE.read_bytes())
    current = producer_for(definition)
    old = replace(current, revision="1")
    assert current.revision == "2"
    assert current.graph_id("unchanged-checkpoint", "model_defined") != old.graph_id(
        "unchanged-checkpoint", "model_defined"
    )


def test_shared_import_is_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    import torch

    forbidden = Mock(side_effect=AssertionError("numeric execution is forbidden"))
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    monkeypatch.setattr(torch.jit, "trace", forbidden)
    assert analyze().status == "complete"
    forbidden.assert_not_called()


def test_separate_key_value_state_ports_and_edges_survive_mapping() -> None:
    value = document()
    for tag, instance in zip(("a", "b"), value["templates"][0]["instances"], strict=True):
        root = next(n for n in value["nodes"] if n["id"] == f"{tag}-attention")
        for branch in ("key", "value"):
            child_id = f"{tag}-state-{branch}"
            root["children"].append(child_id)
            ports = [
                {"id": f"{branch}_past", "direction": "input", "label": "past", "shape": []},
                {"id": f"{branch}_next", "direction": "output", "label": "next", "shape": []},
            ]
            root["ports"].extend(ports)
            value["nodes"].append(
                {
                    "id": child_id,
                    "kind": "operation",
                    "label": f"{branch} state",
                    "parent_id": root["id"],
                    "operation": "state_passthrough",
                    "ports": ports,
                }
            )
            instance["nodes"].append({"role": f"{branch}_state", "node_id": child_id})
            for owner, role in ((root["id"], "root"), (child_id, f"{branch}_state")):
                instance["ports"].extend(
                    {"role": f"{role}.{p['id']}", "node_id": owner, "port_id": p["id"]}
                    for p in ports
                )
            for direction, source, target, port_id in (
                ("in", root["id"], child_id, f"{branch}_past"),
                ("out", child_id, root["id"], f"{branch}_next"),
            ):
                edge_id = f"{tag}-{branch}-{direction}"
                value["edges"].append(
                    {
                        "id": edge_id,
                        "kind": "state",
                        "source": {"node_id": source, "port_id": port_id},
                        "target": {"node_id": target, "port_id": port_id},
                    }
                )
                instance["edges"].append({"role": f"{branch}_{direction}", "edge_id": edge_id})
    result = analyze(value)
    assert result.status == "complete", result.diagnostics
    family = result.graph.templates[0]
    for instance in family.instances:
        roles = {p.role: (p.node_id, p.port_id) for p in instance.ports}
        assert roles["root.key_past"] != roles["root.value_past"]
        assert roles["root.key_next"] != roles["root.value_next"]
    # A still-valid ordinary graph that swaps just the state paths is not equivalent.
    for edge in value["edges"]:
        if edge["id"] in {"b-key-in", "b-value-in"}:
            old = edge["source"]["port_id"]
            edge["source"]["port_id"] = "value_past" if old == "key_past" else "key_past"
    assert analyze(value).status == "unavailable"


def test_parameter_resource_references_are_mapped_not_guessed() -> None:
    value = document()
    for tag in ("a", "b"):
        node = next(n for n in value["nodes"] if n["id"] == f"{tag}-attention-q")
        pid = node.pop("parameter_ids")[0]
        node["references"] = [{"kind": "parameter", "parameter_id": pid}]
    assert analyze(value).status == "complete"
    value["templates"][0]["instances"][1]["parameters"] = [
        m for m in value["templates"][0]["instances"][1]["parameters"] if m["role"] != "q.weight"
    ]
    assert analyze(value).status == "unavailable"
