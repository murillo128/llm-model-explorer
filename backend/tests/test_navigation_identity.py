"""Navigation correspondence is independent of runtime IDs and numeric authority."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from architecture_grouping_cases import cases
from test_architecture_templates import build
from test_model_defined_architecture import definition, document, inputs

from llm_model_explorer.architecture_analysis import AnalysisInput, DescriptionRegistry, parse_graph
from llm_model_explorer.architecture_analysis.compact import compact_graph, expand_graph
from llm_model_explorer.architecture_analysis.model_defined import analyze_definition
from llm_model_explorer.architecture_analysis.records import ArchitectureGraph
from llm_model_explorer.architecture_analysis.validation import GraphError


def model_owned_pair() -> tuple[ArchitectureGraph, ArchitectureGraph]:
    old = analyze_definition(definition(), inputs()).graph
    authored = document()
    authored["architecture_revision"] = "navigation-second-revision"
    authored["nodes"].reverse()
    projection = next(n for n in authored["nodes"] if n.get("operation") == "linear")
    projection["label"] = "Renamed projection"
    projection["description"] = "Updated projection description after a model edit."
    new = analyze_definition(
        definition(authored), replace(inputs(), fingerprint="updated-content")
    ).graph
    assert old is not None and new is not None
    return old, new


@pytest.mark.parametrize("name,data,registry", cases(), ids=[case[0] for case in cases()])
def test_registered_native_families_keep_keys_across_all_runtime_id_changes(
    name: str, data: AnalysisInput, registry: DescriptionRegistry
) -> None:
    old_builder = build(data, registry)
    old = old_builder.finish()
    new_builder = build(replace(data, fingerprint="updated-content"), registry)
    new = new_builder.finish()
    assert old.navigation_namespace == new.navigation_namespace
    for category in ("nodes", "repetitions", "templates"):
        before, after = getattr(old, category) or [], getattr(new, category) or []
        assert not {record.id for record in before} & {record.id for record in after}
        assert {record.navigation_key for record in before} == {
            record.navigation_key for record in after
        }
        assert all(n.navigation_key for n in before), (name, category)
        assert len({n.navigation_key for n in before}) == len(before)
    # Select a real nonzero instance; storage belongs to that exact invocation.
    rep = old.repetitions[0]
    second = next(i for i in rep.instances if i.index > 0)
    node = next(n for n in old.nodes if n.id == second.node_id)
    replacement = next(n for n in new.nodes if n.navigation_key == node.navigation_key)
    assert any(
        i.index == second.index and i.node_id == replacement.id
        for r in new.repetitions
        for i in r.instances
    )
    assert [(p.id, p.navigation_key) for p in node.ports] == [
        (p.id, p.navigation_key) for p in replacement.ports
    ]


def test_model_owned_ids_survive_label_order_content_and_revision_changes() -> None:
    old, new = model_owned_pair()
    assert old.navigation_namespace == new.navigation_namespace
    assert not {n.id for n in old.nodes} & {n.id for n in new.nodes}
    assert {n.navigation_key for n in old.nodes} == {n.navigation_key for n in new.nodes}
    before = next(n for n in old.nodes if n.operation == "linear")
    after = next(n for n in new.nodes if n.navigation_key == before.navigation_key)
    assert after.label == "Renamed projection"
    assert after.description == "Updated projection description after a model edit."
    assert before.parameter_ids != after.parameter_ids
    changed_scope = document() | {"scope": "incompatible-domain"}
    incompatible = analyze_definition(definition(changed_scope), inputs()).graph
    assert incompatible is not None
    assert incompatible.navigation_namespace != old.navigation_namespace


def test_navigation_keys_are_bounded_and_unique_per_kind_and_local_port_scope() -> None:
    graph, _ = model_owned_pair()
    for edit in ("duplicate-node", "duplicate-port", "oversize"):
        value = graph.document()
        if edit == "duplicate-node":
            value["nodes"][1]["navigation_key"] = value["nodes"][0]["navigation_key"]
        elif edit == "duplicate-port":
            node = next(n for n in value["nodes"] if len(n["ports"]) > 1)
            node["ports"][1]["navigation_key"] = node["ports"][0]["navigation_key"]
        else:
            value["navigation_namespace"] = "x" * 129
        with pytest.raises(GraphError):
            parse_graph(value, inputs().bindings)


def test_small_routed_family_roundtrip_keeps_each_expert_key() -> None:
    from test_deepseek_v2_architecture import inputs as metadata
    from test_deepseek_v2_architecture import native_configuration

    from llm_model_explorer.architecture_analysis import GraphBuilder
    from llm_model_explorer.architecture_analysis.deepseek_v2 import (
        PRODUCER,
        DeepseekGraph,
        checked,
    )

    config = native_configuration() | {
        "num_hidden_layers": 2,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
    }
    data = metadata(config)
    # Exercise the packaged constructor with bounded geometry, not admission of
    # a new public model family or a full reference-sized instance matrix.
    native = checked(native_configuration())
    assert native is not None
    configuration = native | config
    builder = GraphBuilder(data, PRODUCER, "language_model")
    DeepseekGraph(data, builder, configuration).build()
    graph = builder.finish()
    compact = compact_graph(graph)
    assert compact.compact_components
    restored = expand_graph(compact)
    assert {n.id: n.navigation_key for n in restored.nodes} == {
        n.id: n.navigation_key for n in graph.nodes
    }
    family = compact.compact_components[0]
    assert family.instances[0].node_navigation_keys != family.instances[1].node_navigation_keys
    bad = compact.document()
    bad["compact_components"][0]["instances"][1]["node_navigation_keys"].pop()
    with pytest.raises(GraphError, match="navigation mapping length"):
        parse_graph(bad, data.bindings)


def test_browser_fixture_is_two_real_importer_results() -> None:
    path = Path(__file__).resolve().parents[2] / "ui/tests/fixtures/architecture-refresh.json"
    old, new = model_owned_pair()
    assert json.loads(path.read_text()) == {"before": old.document(), "after": new.document()}


@pytest.mark.parametrize("family", ["clm", "kev"])
def test_packaged_invocations_do_not_use_backbone_runtime_ids(tmp_path: Path, family: str) -> None:
    import importlib

    from llm_model_explorer.architecture_analysis import AnalysisInput
    from llm_model_explorer.architecture_service import packaged_registry
    from llm_model_explorer.models import ModelCatalogue

    root = tmp_path / "models"
    base, head, _ = importlib.import_module(family + "_fixtures").fixture(root)
    destination = root / family
    importlib.import_module("test_" + family + "_export").export(base, head, destination)
    data = AnalysisInput.from_source(
        ModelCatalogue(root).inspect_directory(destination).pin(), tokenizer_available=True
    )
    old = packaged_registry().analyze(data).graph
    new = packaged_registry().analyze(replace(data, fingerprint="replacement-content")).graph
    assert old is not None and new is not None
    for category in ("nodes", "repetitions", "templates"):
        before, after = getattr(old, category) or [], getattr(new, category) or []
        assert all(n.navigation_key for n in before), (family, category)
        assert {n.navigation_key for n in before} == {n.navigation_key for n in after}
        assert len({n.navigation_key for n in before}) == len(before)
        assert not {n.id for n in before} & {n.id for n in after}
    if family == "clm":
        # The two semantic invocations share storage, never navigation identity.
        consumers = [
            n
            for n in old.nodes
            if any(
                p.name == "model.layers.1.self_attn.q_proj.weight" and p.id in n.parameter_ids
                for p in old.parameters
            )
        ]
        assert len(consumers) == 2
        assert consumers[0].navigation_key != consumers[1].navigation_key
