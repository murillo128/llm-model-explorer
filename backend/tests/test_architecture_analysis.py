"""Consume the independently audited API oracle; no model-family implementation oracle."""

import builtins
import copy
import importlib.util
import json
import shutil
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from test_models import make_model
from test_quantized_models import quantized_model

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    BindingContext,
    Description,
    DescriptionRegistry,
    GraphBuilder,
    GraphError,
    NumericTensor,
    Producer,
    parse_graph,
    serialize_graph,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.validation import bounded_chunks, serialized_size
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.tensor_source import ModelSource

ROOT = Path(__file__).resolve().parents[2]
ORACLE = json.loads((ROOT / "api/fixtures/architecture.json").read_text())
PRODUCER = Producer("test-structure", "1", "audited-api-d65c92a")


def edits(value: Any, changes: list[dict[str, Any]]) -> Any:
    result = copy.deepcopy(value)
    for change in changes:
        target = result
        for key in change["path"][:-1]:
            target = target[int(key)] if isinstance(target, list) else target[key]
        key = int(change["path"][-1]) if isinstance(target, list) else change["path"][-1]
        if change.get("delete"):
            del target[key]
        else:
            target[key] = change["value"]
    return result


def context(value: dict[str, Any] | None = None) -> BindingContext:
    c = ORACLE["context"] if value is None else value
    # The fixture's physical oracle is independent from graph edits under test.
    physical = {
        s["name"]: r.ArchitectureStorage.model_validate(s)
        for p in ORACLE["response"]["graph"]["parameters"]
        for s in p["storage"]
    }
    # Fixed independent NVFP4 storage for the API's alternate packed cases.
    # Invalid graph mutations must not redefine the observed physical inventory.
    for name, dtype, shape in [
        ("fp4.weight", "U8", [2, 16]),
        ("fp4.weight_scale", "F8_E4M3", [2, 2]),
        ("fp4.weight_scale_2", "F32", []),
        ("fp4.input_scale", "F32", []),
    ]:
        physical[name] = r.ArchitectureStorage(name=name, dtype=dtype, shape=shape)
    # The API's rank1 variant changes its input inventory and expected native geometry.
    for tensor in c["inventory"]["tensors"]:
        if (
            tensor["name"] in physical
            and tensor.get("storage_format", "safetensors") == "safetensors"
        ):
            physical[tensor["name"]] = r.ArchitectureStorage(
                name=tensor["name"], dtype=tensor["storage_dtype"], shape=tensor["shape"]
            )
    numeric = {
        t["id"]: NumericTensor(
            t["id"],
            t["name"],
            tuple(t["shape"]),
            t["storage_dtype"],
            t.get("storage_format", "safetensors"),
        )
        for t in c["inventory"]["tensors"]
    }
    return BindingContext(physical, numeric, c["tokenizer_available"])


def inputs() -> AnalysisInput:
    return AnalysisInput(
        "content-fingerprint",
        {"model_type": "synthetic", "architectures": ["ReviewedSynthetic"]},
        context(),
    )


def construct(_: AnalysisInput, builder: GraphBuilder) -> None:
    expected = r.ArchitectureGraph.model_validate(ORACLE["response"]["graph"])
    for symbol in expected.symbols:
        builder.add_symbol(symbol.name, symbol.meaning)
    for node in expected.nodes:
        builder.add_node(node)
    for edge in expected.edges:
        builder.add_edge(edge)
    for param in expected.parameters:
        builder.add_parameter(param)
    for rep in expected.repetitions:
        builder.add_repetition(rep)


def description(**changes: Any) -> Description:
    return replace(
        Description(
            PRODUCER,
            "language_model",
            frozenset({"synthetic"}),
            frozenset({"ReviewedSynthetic"}),
            lambda _: True,
            construct,
        ),
        **changes,
    )


def registry(**changes: Any) -> DescriptionRegistry:
    result = DescriptionRegistry()
    result.register(description(**changes))
    return result


# Response/session envelopes are intentionally outside the graph core.
GRAPH_CASES = [
    case
    for case in ORACLE["cases"]
    if all(edit["path"][0] == "graph" and len(edit["path"]) > 1 for edit in case["edits"])
    and all(edit["path"][0] != "session" for edit in case["context_edits"])
]


@pytest.mark.parametrize("case", GRAPH_CASES, ids=lambda case: case["name"])
def test_independently_reviewed_graph_cases(case: dict[str, Any]) -> None:
    response = edits(ORACLE[case.get("base", "response")], case["edits"])
    binding = context(edits(ORACLE["context"], case["context_edits"]))
    if case["name"] == "rank1-native":
        # This core also verifies physical descriptors, which the API fixture context omits.
        response["graph"]["parameters"][1]["storage"] = copy.deepcopy(
            response["graph"]["parameters"][0]["storage"]
        )
    if case["name"] == "zero-product":
        # These independent API cases omit the backend's physical descriptor map.
        binding = replace(
            binding,
            physical={
                s["name"]: r.ArchitectureStorage.model_validate(s)
                for p in response["graph"]["parameters"]
                for s in p["storage"]
            },
        )
    if case.get("physical_storage"):
        # API cases carry an independent physical inventory for alternate packed groups.
        physical = dict(binding.physical)
        physical.update(
            {
                storage["name"]: r.ArchitectureStorage.model_validate(storage)
                for storage in case["physical_storage"]
            }
        )
        binding = replace(binding, physical=physical)
    if case["valid"]:
        result = parse_graph(response["graph"], binding)
        assert json.loads(serialize_graph(result)) == response["graph"]
    else:
        with pytest.raises((GraphError, ValueError)):
            parse_graph(response["graph"], binding)


def test_builder_preserves_independent_full_instance_graph() -> None:
    result = registry().analyze(inputs())
    assert result.status == "complete", result.diagnostics
    assert result.graph is not None
    actual = result.graph.document()
    actual["graph_id"] = ORACLE["response"]["graph"]["graph_id"]
    assert actual == ORACLE["response"]["graph"]


def test_generated_records_match_current_published_schema() -> None:
    spec = importlib.util.spec_from_file_location(
        "generator", ROOT / "backend/scripts/generate_architecture_records.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.generate() == Path(r.__file__).read_text()


@pytest.mark.parametrize(
    "field", ["revision", "source_revision", "analyzer_revision", "schema_revision"]
)
def test_semantic_revision_invalidates_graph_and_record_ids(field: str) -> None:
    old = GraphBuilder(inputs(), PRODUCER, "language_model")
    new = GraphBuilder(inputs(), replace(PRODUCER, **{field: "changed"}), "language_model")
    assert old.graph_id != new.graph_id
    assert old.record_id("node", "layer.0") != new.record_id("node", "layer.0")
    assert old.graph_id != inputs().fingerprint


def test_identity_uses_semantic_keys_not_order_labels_or_ambiguous_concatenation() -> None:
    a = GraphBuilder(inputs(), PRODUCER, "language_model")
    b = GraphBuilder(inputs(), PRODUCER, "language_model")
    keys = [f"layer.{index}" for index in range(200)]
    assert {key: a.record_id("node", key) for key in keys} == {
        key: b.record_id("node", key) for key in reversed(keys)
    }
    assert len({a.record_id("node", key) for key in keys}) == 200
    assert a.record_id("ab", "c") != a.record_id("a", "bc")
    assert a.record_id("node", "x") != a.record_id("parameter", "x")
    assert (
        a.graph_id
        != GraphBuilder(
            replace(inputs(), fingerprint="changed"), PRODUCER, "language_model"
        ).graph_id
    )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"model_type": "synthetic"},
        {"model_type": "marketing-name", "architectures": ["ReviewedSynthetic"]},
        {"model_type": "synthetic", "architectures": ["ReviewedSynthetic", "UnknownVariant"]},
        {"model_type": "synthetic", "architectures": '__import__("checkpoint")'},
    ],
)
def test_selection_never_guesses_or_imports(config: dict[str, object]) -> None:
    build = Mock(side_effect=AssertionError("Unselected descriptions must not run"))
    result = registry(build=build).analyze(replace(inputs(), configuration=config))
    assert result.status == "unavailable"
    assert result.reason == "unsupported_architecture"
    build.assert_not_called()


def test_checked_options_and_ambiguous_descriptions_fail_closed() -> None:
    assert registry(supports=lambda _: False).analyze(inputs()).reason == "unsupported_architecture"
    registered = registry()
    registered.register(description(producer=replace(PRODUCER, description="overlap")))
    assert registered.analyze(inputs()).reason == "unsupported_architecture"
    with pytest.raises(GraphError):
        registered.register(description())


def test_explicit_unknown_and_missing_required_parameter_are_partial() -> None:
    def partial(_: AnalysisInput, builder: GraphBuilder) -> None:
        builder.unknown("unverified", "Unknown block", "Layer variant has not been reviewed.")
        builder.native_parameter("missing", "absent.weight", None, PRODUCER.provenance())

    result = registry(build=partial).analyze(inputs())
    assert result.status == "partial"
    assert result.graph is not None
    assert result.graph.edges == []
    assert result.graph.nodes[0].operation is None
    assert {d.code for d in result.graph.diagnostics} == {"unknown_region", "unresolved_binding"}


def test_unavailable_never_exposes_empty_interrupted_graph_or_exception_text() -> None:
    assert registry(build=lambda *_: None).analyze(inputs()).status == "unavailable"

    def failed(_: AnalysisInput, builder: GraphBuilder) -> None:
        builder.unknown("x", "X", "Unknown")
        raise RuntimeError("/private/model/config.py secret traceback")

    result = registry(build=failed).analyze(inputs())
    assert result.graph is None
    assert result.reason == "analysis_failed"
    assert "/private" not in str(result)


def test_native_parameter_binding_and_rank_limitation() -> None:
    b = GraphBuilder(inputs(), PRODUCER, "language_model")
    b.unknown("scope", "Known boundary", "Fixture only.")
    for name, shape in [("linear.weight", [2, 3]), ("patch.weight", [2, 3, 4])]:
        b.native_parameter(
            name,
            name,
            [r.ArchitectureConstantDimension(kind="constant", value=n) for n in shape],
            PRODUCER.provenance(),
        )
    graph = b.finish()
    assert graph.parameters[0].inspection.status == "available"
    assert graph.parameters[1].inspection.status == "unavailable"
    assert graph.parameters[1].inspection.reason == "unsupported_rank"
    with pytest.raises(GraphError, match="geometry"):
        b.native_parameter(
            "wrong",
            "linear.weight",
            [r.ArchitectureConstantDimension(kind="constant", value=6)],
            [],
        )


def test_incremental_builder_limit_aborts_before_consuming_all_instances() -> None:
    calls = []

    def big(_: AnalysisInput, b: GraphBuilder) -> None:
        for i in range(100_000):
            calls.append(i)
            b.unknown(str(i), "Block", "Unknown semantics")

    result = registry(build=big).analyze(inputs(), byte_limit=2048)
    assert result.reason == "unsupported_size"
    assert result.graph is None
    assert 0 < len(calls) < 10


def test_utf8_exact_serialization_limit_and_structure_bounds() -> None:
    doc = ORACLE["response"]["graph"]
    encoded = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode()
    assert serialized_size(doc, len(encoded)) == len(encoded)
    with pytest.raises(GraphError, match="limit"):
        serialized_size(doc, len(encoded) - 1)
    for value in [{"x": "é" * 16_385}, {"x": list(range(100))}]:
        with pytest.raises(GraphError):
            serialized_size(value, 50)
    deep: Any = []
    for _ in range(40):
        deep = [deep]
    with pytest.raises(GraphError):
        serialized_size(deep)


@pytest.mark.parametrize("case", ORACLE["byte_cases"], ids=lambda case: case["name"])
def test_published_32_mib_byte_bound(case: dict[str, Any]) -> None:
    consumed = []

    def chunks() -> Any:
        for _ in range(case["repeat"]):
            yield b"x" * case["chunk_bytes"]
        yield b"x" * case["tail_bytes"]
        consumed.append("complete")

    if case["valid"]:
        assert bounded_chunks(chunks()) == 33_554_432
        assert consumed == ["complete"]
    else:
        with pytest.raises(GraphError):
            bounded_chunks(chunks())
        assert consumed == []


def test_mutation_after_add_cannot_change_builder_records() -> None:
    b = GraphBuilder(inputs(), PRODUCER, "language_model")
    node = r.ArchitectureLeafNode(
        id="input",
        kind="input",
        label="Input",
        ports=[],
        parameter_ids=[],
        references=[],
        attributes=[],
        provenance=[],
    )
    b.add_node(node)
    node.parameter_ids.append("foreign")
    assert b.finish().nodes[0].parameter_ids == []


@pytest.mark.parametrize("kind", ["JunHowie", "AxionML"])
def test_guarded_quantized_metadata_separates_storage_from_inspection(
    tmp_path: Path, kind: str
) -> None:
    root = tmp_path / "models"
    root.mkdir()
    quantized_model(root, kind)
    source = ModelCatalogue(root).discover()[0].pin()
    data = AnalysisInput.from_source(source, tokenizer_available=False)
    assert len(data.bindings.physical) > len(data.bindings.numeric)
    packed = [t for t in data.bindings.numeric.values() if t.storage_format != "safetensors"]
    assert len(packed) == 1
    assert packed[0].storage_format == ("gptq-int4" if kind == "JunHowie" else "nvfp4")
    assert packed[0].dtype == ("I32" if kind == "JunHowie" else "U8")
    assert packed[0].name.endswith(".weight")
    assert not any(
        hasattr(t, "file") or hasattr(t, "offset") for t in data.bindings.physical.values()
    )


def test_metadata_analysis_is_static_readonly_and_relocation_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "original"
    root.mkdir()
    directory = make_model(root)
    # No metadata name: use directory fallback and show it is absent from graph identity.
    config_path = directory / "config.json"
    config_path.write_text(
        json.dumps({"model_type": "synthetic", "architectures": ["ReviewedSynthetic"]})
    )
    (directory / "modeling_custom.py").write_text(
        'raise AssertionError("checkpoint code executed")'
    )
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copytree(directory, relocated / "different-name")
    for path in (directory, relocated / "different-name"):
        for file in path.iterdir():
            file.chmod(0o444)
        path.chmod(0o555)
    sources = [ModelCatalogue(path).discover()[0].pin() for path in (root, relocated)]
    assert sources[0].model_id != sources[1].model_id
    assert sources[0].fingerprint == sources[1].fingerprint
    denied = Mock(side_effect=AssertionError("Forbidden numerical/network/checkpoint execution"))
    import safetensors
    import safetensors.torch
    import transformers

    for owner, names in [
        (socket, ["socket", "create_connection"]),
        (torch, ["load", "tensor", "empty", "zeros", "ones", "from_file", "compile"]),
        (safetensors, ["safe_open"]),
        (safetensors.torch, ["load_file", "load"]),
        (torch.nn.Module, ["__init__", "__call__"]),
        (torch.jit, ["trace", "script"]),
        (torch.cuda, ["init", "set_device", "_lazy_init"]),
        (torch.Tensor, ["cuda", "to"]),
        (torch.fx, ["symbolic_trace"]),
        (transformers.AutoModel, ["from_config", "from_pretrained"]),
        (transformers.AutoModelForCausalLM, ["from_config", "from_pretrained"]),
        (transformers.GenerationMixin, ["generate"]),
        (ModelSource, ["iter_tensor", "iter_rows", "local_directory"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, denied)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert not name.startswith(("modeling_custom", "checkpoint"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    def build(_: AnalysisInput, b: GraphBuilder) -> None:
        b.add_node(
            r.ArchitectureLeafNode(
                id=b.record_id("node", "input"),
                kind="input",
                label="Input",
                ports=[],
                parameter_ids=[],
                references=[],
                attributes=[],
                provenance=PRODUCER.provenance(),
            )
        )

    results = [
        registry(build=build).analyze(AnalysisInput.from_source(s, tokenizer_available=False))
        for s in sources
    ]
    assert all(result.status == "complete" for result in results), results
    assert results[0].graph == results[1].graph
    assert "/original" not in str(results)
    denied.assert_not_called()


def test_long_alias_and_containment_chains_terminate_without_recursion() -> None:
    graph = copy.deepcopy(ORACLE["response"]["graph"])
    for index in range(1500):
        graph["parameters"].append(
            {
                **copy.deepcopy(graph["parameters"][1]),
                "id": f"alias_{index}",
                "alias_of": "weight" if index == 0 else f"alias_{index - 1}",
            }
        )
        graph["nodes"].append(
            {
                "id": f"group_{index}",
                "kind": "group",
                "label": "Nested group",
                "ports": [],
                "parameter_ids": [],
                "references": [],
                "attributes": [],
                "provenance": [],
                "children": [f"group_{index + 1}"] if index < 1499 else [],
                **({"parent_id": f"group_{index - 1}"} if index > 0 else {}),
            }
        )
    parsed = parse_graph(graph, context())
    assert len(parsed.parameters) == 1506
    graph["parameters"][-1]["alias_of"] = "alias_1499"
    with pytest.raises(GraphError, match="Cyclic"):
        parse_graph(graph, context())


def test_foreign_physical_storage_is_rejected_even_without_numeric_inspection() -> None:
    graph = copy.deepcopy(ORACLE["response"]["graph"])
    graph["parameters"][2]["storage"][0]["name"] = "invented.qweight"
    with pytest.raises(GraphError, match="physical storage"):
        parse_graph(graph, context())


def test_unrelated_diagnostic_cannot_hide_shape_disagreement() -> None:
    graph = copy.deepcopy(ORACLE["response"]["graph"])
    graph["nodes"][2]["ports"][0]["shape"][0]["value"] = 3
    graph["diagnostics"] = [
        {"code": "unknown_region", "message": "Unrelated fact.", "node_id": "layer1"}
    ]
    with pytest.raises(GraphError, match="dimensions disagree"):
        parse_graph(graph, context())


def test_oversized_typed_record_rejected_before_serialization_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = r.ArchitectureLeafNode(
        id="x",
        kind="input",
        label="X",
        ports=[],
        parameter_ids=[],
        references=[],
        attributes=[],
        provenance=[],
    )
    # Simulate a packaged producer mutating a nested collection after typed construction.
    node.parameter_ids.extend(["x"] * 10_000)
    denied = Mock(side_effect=AssertionError("Must preflight before copying"))
    monkeypatch.setattr(r.ArchitectureLeafNode, "document", denied)
    with pytest.raises(GraphError, match="limit"):
        GraphBuilder(inputs(), PRODUCER, "language_model", byte_limit=100).add_node(node)
    denied.assert_not_called()


def test_package_import_has_no_numerical_or_model_library_dependency() -> None:
    import subprocess
    import sys

    check = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class Forbid:
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'safetensors', 'httpx'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Forbid())
import llm_model_explorer.architecture_analysis
import llm_model_explorer.architecture_analysis.vjepa2
""",
        ],
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr
