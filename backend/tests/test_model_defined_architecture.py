"""Independent declared-graph cases; the importer never generates its own oracle."""

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
from llm_model_explorer.architecture_analysis.model_defined_schema import (
    MAX_DEFINITION_BYTES,
    ModelDefinition,
    definition_schema,
)
from llm_model_explorer.architecture_analysis.validation import (
    BindingContext,
    GraphError,
    NumericTensor,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/model-owned-architecture/architecture.json"


def document() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(EXAMPLE.read_text()))


def definition(value: dict[str, Any] | None = None) -> ModelDefinition:
    return parse_definition(json.dumps(document() if value is None else value).encode())


def inputs() -> AnalysisInput:
    # Independent inventory with exact names/geometry, not derived by the importer.
    physical = {
        name: r.ArchitectureStorage(name=name, dtype="F32", shape=shape)
        for name, shape in [("encoder.proj.weight", [4, 3]), ("encoder.proj.bias", [4])]
    }
    numeric = {
        key: NumericTensor(key, name, tuple(shape), "F32")
        for key, name, shape in [
            ("actual-weight-id", "encoder.proj.weight", [4, 3]),
            ("actual-bias-id", "encoder.proj.bias", [4]),
        ]
    }
    return AnalysisInput(
        "checkpoint-fingerprint",
        {"model_type": "never_seen_before"},
        BindingContext(physical, numeric, False),
    )


def test_named_native_bindings_and_truthful_origin() -> None:
    result = analyze_definition(definition(), inputs())
    assert result.status == "complete", result.diagnostics
    assert result.graph is not None
    graph = result.graph
    assert graph.scope == "model_defined"
    assert len(graph.nodes) == 6
    assert len(graph.edges) == 5
    assert {
        p.inspection.tensor_id for p in graph.parameters if p.inspection.status == "available"
    } == {
        "actual-weight-id",
        "actual-bias-id",
    }
    assert {p.name for p in graph.parameters} == {"encoder.proj.weight", "encoder.proj.bias"}
    assert graph.graph_id != inputs().fingerprint
    assert not {n.id for n in graph.nodes} & {n["id"] for n in document()["nodes"]}
    notice = next(n for n in graph.nodes if n.label == "Model-supplied definition")
    assert notice.kind == "context"
    assert {a.name: a.value for a in notice.attributes}["semantic_verification"] == "not_verified"
    assert all(
        "Reviewed implementation" not in (p.rule or "") for n in graph.nodes for p in n.provenance
    )


def test_unknown_operations_need_no_model_family_adapter() -> None:
    value = document()
    value["architecture_revision"] = "new-attention-experiment"
    value["scope"] = "another_sport_or_domain"
    value["nodes"][3]["operation"] = "experimental_relational_update"
    value["nodes"][3]["label"] = "Custom operation"
    result = analyze_definition(definition(value), inputs())
    assert result.status == "complete", result.diagnostics
    assert result.graph is not None
    assert any(n.operation == "experimental_relational_update" for n in result.graph.nodes)


def test_present_but_empty_operation_is_not_reported_as_missing() -> None:
    value = document()
    value["nodes"][3]["operation"] = ""
    with pytest.raises(GraphError) as error:
        definition(value)
    assert error.value.code == "schema_invalid_field"
    assert "#/nodes/3/operation" in str(error.value)


def test_schema_is_reproducible_and_example_is_not_a_runtime_graph() -> None:
    expected = json.loads(
        (ROOT / "docs/spec/backend/architecture-definition.schema.json").read_text()
    )
    assert definition_schema() == expected
    value = document()
    assert "graph_id" not in value
    assert all("inspection" not in p and "storage" not in p for p in value["parameters"])


@pytest.mark.parametrize("extra", ["python", "include", "$ref", "graph_id", "producer"])
def test_top_level_executable_or_runtime_fields_are_rejected(extra: str) -> None:
    value = document()
    value[extra] = "untrusted"
    with pytest.raises(GraphError):
        definition(value)


@pytest.mark.parametrize("version", [0, 2, True, 1.0, "1", None])
def test_unknown_or_non_integer_version_is_rejected(version: Any) -> None:
    value = document()
    value["schema_version"] = version
    with pytest.raises(GraphError):
        definition(value)


@pytest.mark.parametrize("raw", [b"{", b"[]", b"\xff", b'{"a":1,"a":2}', b'{"x":NaN}'])
def test_malformed_json_is_safe(raw: bytes) -> None:
    with pytest.raises(GraphError) as error:
        parse_definition(raw)
    assert "/private" not in str(error.value)


def test_bounds_before_graph_construction() -> None:
    with pytest.raises(GraphError, match="limit"):
        parse_definition(b" " * (MAX_DEFINITION_BYTES + 1))
    nested: dict[str, Any] = {"x": []}
    for _ in range(40):
        nested = {"x": nested}
    with pytest.raises(GraphError):
        parse_definition(json.dumps(nested).encode())
    result = analyze_definition(definition(), inputs(), byte_limit=100)
    assert result.status == "unavailable" and result.reason == "unsupported_size"


def test_missing_declared_storage_is_invalid() -> None:
    original = inputs()
    binding = BindingContext({}, {}, False)
    result = analyze_definition(definition(), replace(original, bindings=binding))
    assert result.status == "unavailable"
    assert result.diagnostics[0].code == "binding_missing_tensor"
    assert "#/parameters/0" in result.diagnostics[0].message


def test_wrong_native_geometry_is_unavailable() -> None:
    original = inputs()
    physical = dict(original.bindings.physical)
    physical["encoder.proj.weight"] = r.ArchitectureStorage(
        name="encoder.proj.weight", dtype="F32", shape=[3, 4]
    )
    result = analyze_definition(
        definition(), replace(original, bindings=replace(original.bindings, physical=physical))
    )
    assert result.status == "unavailable"


@pytest.mark.parametrize(
    "edit", ["endpoint", "parameter", "duplicate", "bypass", "symbol", "cycle"]
)
def test_invalid_graph_relations_are_rejected(edit: str) -> None:
    value = document()
    if edit == "endpoint":
        value["edges"][0]["target"]["node_id"] = "absent"
    elif edit == "parameter":
        value["nodes"][2]["parameter_ids"] = ["absent"]
    elif edit == "duplicate":
        value["nodes"].append(copy.deepcopy(value["nodes"][0]))
    elif edit == "bypass":
        value["edges"][0]["target"]["node_id"] = "projection"
    elif edit == "symbol":
        value["symbols"] = []
    else:
        value["nodes"][1]["parent_id"] = "encoder"
        value["nodes"][1]["children"].append("encoder")
    result = analyze_definition(definition(value), inputs())
    assert result.status == "unavailable", edit


def test_author_partial_and_definition_revision_identity() -> None:
    value = document()
    value["coverage"] = "partial"
    with pytest.raises(GraphError):
        definition(value)
    value["incomplete_reason"] = "Only the encoder is described."
    partial = analyze_definition(definition(value), inputs())
    assert partial.status == "partial"
    old = analyze_definition(definition(), inputs())
    value = document()
    value["architecture_revision"] = "v2"
    new = analyze_definition(definition(value), inputs())
    assert old.graph is not None and new.graph is not None
    assert old.graph.graph_id != new.graph.graph_id
    assert producer_for(definition()).description == "model-defined-json"


def test_analysis_never_executes_model_code_or_reads_tensor_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    import torch

    forbidden = Mock(side_effect=AssertionError("execution is forbidden"))
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    monkeypatch.setattr(torch.jit, "trace", forbidden)
    result = analyze_definition(definition(), inputs())
    assert result.status == "complete", result.diagnostics
    forbidden.assert_not_called()


def test_repetition_and_shared_parameter_ids_survive_remapping() -> None:
    value = document()
    value["nodes"] = [
        {"id": "stack", "kind": "group", "label": "Stack", "children": ["first", "second"]},
        {"id": "first", "kind": "group", "parent_id": "stack", "label": "First", "children": ["a"]},
        {
            "id": "second",
            "kind": "group",
            "parent_id": "stack",
            "label": "Second",
            "children": ["b"],
        },
        {
            "id": "a",
            "kind": "operation",
            "parent_id": "first",
            "label": "A",
            "operation": "declared_projection",
            "parameter_ids": ["weight"],
        },
        {
            "id": "b",
            "kind": "operation",
            "parent_id": "second",
            "label": "B",
            "operation": "declared_projection",
            "parameter_ids": ["weight"],
        },
    ]
    value["edges"] = []
    value["repetitions"] = [
        {
            "id": "layers",
            "parent_id": "stack",
            "label": "Layers",
            "instances": [
                {"node_id": "first", "index": 0, "variant": "shared"},
                {"node_id": "second", "index": 1, "variant": "shared"},
            ],
        }
    ]
    result = analyze_definition(definition(value), inputs())
    assert result.status == "complete", result.diagnostics
    assert result.graph is not None
    rep = result.graph.repetitions[0]
    assert [i.index for i in rep.instances] == [0, 1]
    nodes = {n.label: n for n in result.graph.nodes}
    assert [i.node_id for i in rep.instances] == [nodes["First"].id, nodes["Second"].id]
    assert nodes["A"].parameter_ids == nodes["B"].parameter_ids
    value["repetitions"][0]["instances"][0]["node_id"] = "second"
    assert analyze_definition(definition(value), inputs()).status == "unavailable"


@pytest.mark.parametrize("extra", ["tensor_id", "storage", "inspection", "binding"])
def test_author_cannot_inject_runtime_parameter_metadata(extra: str) -> None:
    value = document()
    value["parameters"][0][extra] = "untrusted"
    with pytest.raises(GraphError):
        definition(value)
