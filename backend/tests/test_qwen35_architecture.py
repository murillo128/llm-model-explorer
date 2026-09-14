"""Independent source connections and captured checkpoint storage, never model execution."""

import copy
import hashlib
import json
import socket
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from test_quantized_models import write_storage

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    BindingContext,
    DescriptionRegistry,
    GraphBuilder,
    NumericTensor,
    serialize_graph,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.qwen35 import PREFIX, PRODUCER, register_qwen35
from llm_model_explorer.architecture_analysis.validation import constants, validate_graph
from llm_model_explorer.models import ModelCatalogue
from llm_model_explorer.tensor_source import ModelSource

FIXTURES = Path(__file__).parent / "fixtures"
TINY = json.loads((FIXTURES / "qwen35-tiny.json").read_text())
REFERENCE = json.loads((FIXTURES / "qwen35-reference.json").read_text())


def metadata(fixture: dict[str, Any] = TINY) -> AnalysisInput:
    f = copy.deepcopy(fixture)
    physical = {k: r.ArchitectureStorage(name=k, **v) for k, v in f["storage"].items()}
    numeric = {
        hashlib.sha256(k.encode()).hexdigest(): NumericTensor(
            hashlib.sha256(k.encode()).hexdigest(), k, tuple(v["shape"]), v["dtype"]
        )
        for k, v in f["storage"].items()
        if v["dtype"] in ("BF16", "F16")
    }
    return AnalysisInput(
        "independent-metadata-fixture", f["configuration"], BindingContext(physical, numeric, False)
    )


def registry() -> DescriptionRegistry:
    result = DescriptionRegistry()
    register_qwen35(result)
    return result


def graph(inputs: AnalysisInput | None = None) -> r.ArchitectureGraph:
    result = registry().analyze(metadata() if inputs is None else inputs)
    assert result.status == "complete", result
    assert result.graph is not None
    return result.graph


def node(g: r.ArchitectureGraph, key: str) -> r.ArchitectureNode:
    nid = GraphBuilder(metadata(), PRODUCER, "language_model").record_id("node", key)
    return next(n for n in g.nodes if n.id == nid)


def linked(
    g: r.ArchitectureGraph,
    source: str,
    target: str,
    source_port: str = "out",
    target_port: str = "x",
) -> bool:
    return any(
        e.source.node_id == node(g, source).id
        and e.target.node_id == node(g, target).id
        and e.source.port_id == source_port
        and e.target.port_id == target_port
        for e in g.edges
    )


def test_both_interiors_and_order() -> None:
    g = graph()
    assert [i.variant for i in g.repetitions[0].instances] == ["linear_attention", "full_attention"]
    assert linked(g, PREFIX + ".layers.0", PREFIX + ".layers.1")
    for i in range(2):
        p = f"{PREFIX}.layers.{i}"
        assert linked(g, p, p + ".input_layernorm", "x")
        assert linked(g, p, p + ".attention_residual", "x", "skip")
        assert linked(g, p + ".attention_residual", p + ".post_attention_layernorm")
        assert linked(g, p + ".attention_residual", p + ".mlp_residual", target_port="skip")
        for name in ["gate_proj", "up_proj"]:
            assert linked(g, p + ".post_attention_layernorm", p + ".mlp." + name)
        assert linked(g, p + ".mlp.gate_proj", p + ".mlp.silu")
        assert linked(g, p + ".mlp.silu", p + ".mlp.multiply", target_port="gate")
        assert linked(g, p + ".mlp.up_proj", p + ".mlp.multiply", target_port="up")
        assert linked(g, p + ".mlp.multiply", p + ".mlp.down_proj")
        assert linked(g, p + ".mlp.down_proj", p + ".mlp_residual", target_port="branch")
    assert linked(g, PREFIX + ".layers.1", PREFIX + ".norm")
    assert linked(g, PREFIX, "lm_head")


def test_full_attention_connections_and_gate_regions() -> None:
    g = graph()
    p = PREFIX + ".layers.1.self_attn"
    for source, target, sp, tp in [
        ("q_proj", "query_gate_heads", "out", "x"),
        ("query_gate_heads", "query_gate_split", "out", "x"),
        ("query_gate_split", "q_norm", "query", "x"),
        ("query_gate_split", "gate_flatten", "gate", "x"),
        ("q_norm", "query_rope", "out", "x"),
        ("k_norm", "key_rope", "out", "x"),
        ("gate_flatten", "sigmoid", "out", "x"),
        ("sigmoid", "output_gate", "out", "gate"),
        ("merge_heads", "output_gate", "out", "attention"),
        ("output_gate", "o_proj", "out", "x"),
        ("query_transpose", "scores", "out", "query"),
        ("key_matrix_transpose", "scores", "out", "key"),
        ("scores", "causal_mask", "out", "scores"),
        ("causal_mask", "softmax", "out", "x"),
        ("softmax", "weighted_values", "out", "probabilities"),
        ("repeat_value", "weighted_values", "out", "value"),
        ("key_transpose", "kv_concat", "out", "key"),
        ("prior_kv", "kv_concat", "key_state", "prior_key_state"),
        ("kv_concat", "next_kv", "value_state", "value_state"),
    ]:
        assert linked(g, p + "." + source, p + "." + target, sp, tp), (source, target)
    params = {v.name: v for v in g.parameters}
    for role, start in [("query", 0), ("gate", 16)]:
        v = params[p + "." + role + ".weight"]
        assert isinstance(v, r.ArchitectureFusedParameter)
        assert v.binding == "fused_region" and v.inspection.status == "unavailable"
        assert v.inspection.reason == "requires_view"
        assert f"j*32+{start}" in v.region.description
        assert constants(v.logical_shape) == (32, 32)
        assert next(s for s in v.storage if s.name == v.region.storage_name).shape == [64, 16]
    assert node(g, p + ".q_norm").operation == "rms_norm_zero_centered"
    assert node(g, p + ".query_rope").ports[0].shape == node(g, p + ".q_norm").ports[-1].shape


def test_linear_attention_connections_shapes_and_states() -> None:
    g = graph()
    p = PREFIX + ".layers.0.linear_attn"
    expected = [
        ("padding_mask", "in_proj_qkv", "out", "x"),
        ("in_proj_qkv", "conv_transpose", "out", "x"),
        ("conv_transpose", "conv1d", "out", "x"),
        ("prior_conv", "conv1d", "out", "prior_state"),
        ("conv1d", "next_conv", "next_state", "state"),
        ("conv1d", "conv_silu", "out", "x"),
        ("conv_silu", "conv_to_sequence", "out", "x"),
        ("conv_to_sequence", "qkv_split", "out", "x"),
        ("in_proj_a", "decay", "out", "a"),
        ("in_proj_b", "beta", "out", "x"),
        ("in_proj_z", "z_heads", "out", "x"),
        ("z_heads", "z_silu", "out", "x"),
        ("z_silu", "output_gate", "out", "gate"),
        ("norm", "output_gate", "out", "normalized"),
        ("delta_rule", "norm", "out", "x"),
        ("output_gate", "merge_heads", "out", "x"),
        ("merge_heads", "out_proj", "out", "x"),
        ("prior_recurrent", "delta_rule", "out", "prior_state"),
        ("delta_rule", "next_recurrent", "next_state", "state"),
        ("beta", "delta_rule", "out", "beta"),
        ("decay", "delta_rule", "out", "log_decay"),
        ("qkv_split", "delta_rule", "value", "value"),
        ("query_scale", "delta_rule", "out", "query"),
        ("key_l2", "delta_rule", "out", "key"),
    ]
    for source, target, sp, tp in expected:
        assert linked(g, p + "." + source, p + "." + target, sp, tp), (source, target)
    for role in ["query", "key"]:
        assert linked(g, p + ".qkv_split", p + "." + role + "_repeat", role)
        assert linked(g, p + "." + role + "_repeat", p + "." + role + "_l2")
    assert node(g, p + ".norm").operation == "rms_norm"
    state = node(g, p + ".prior_recurrent").ports[0].shape
    assert state == [
        r.ArchitectureSymbolDimension(kind="symbol", name="B"),
        *[r.ArchitectureConstantDimension(kind="constant", value=n) for n in (2, 16, 16)],
    ]
    assert "exp(g_t)" in (node(g, p + ".delta_rule").formula or "")
    assert "softplus(a + dt_bias)" in (node(g, p + ".decay").formula or "")
    state_ids = {
        node(g, p + "." + x).id
        for x in ["prior_recurrent", "prior_conv", "next_recurrent", "next_conv"]
    }
    assert all(
        e.kind == "state"
        for e in g.edges
        if e.source.node_id in state_ids or e.target.node_id in state_ids
    )


def test_selected_checkpoint_metadata_complete_and_schema_valid() -> None:
    inputs = metadata(REFERENCE)
    g = graph(inputs)
    assert len(g.repetitions[0].instances) == 24
    assert [i.index for i in g.repetitions[0].instances if i.variant == "full_attention"] == [
        3,
        7,
        11,
        15,
        19,
        23,
    ]
    assert len({i.node_id for i in g.repetitions[0].instances}) == 24
    assert not g.diagnostics
    assert len(serialize_graph(g)) < 33554432
    validate_graph(g, inputs.bindings)
    params = {p.name: p for p in g.parameters}
    q = params[PREFIX + ".layers.3.self_attn.q_proj.weight"]
    assert constants(q.logical_shape) == (4096, 1024)
    assert [(s.dtype, s.shape) for s in q.storage] == [
        ("U8", [4096, 512]),
        ("F8_E4M3", [4096, 64]),
        ("F32", []),
        ("F32", []),
    ]
    assert q.inspection.status == "unavailable"
    head = params["lm_head.weight"]
    assert isinstance(head, r.ArchitectureAliasParameter)
    assert head.alias_of == params[PREFIX + ".embed_tokens.weight"].id
    assert head.inspection == params[PREFIX + ".embed_tokens.weight"].inspection
    conv = params[PREFIX + ".layers.0.linear_attn.conv1d.weight"]
    assert conv.binding == "native" and conv.inspection.status == "unavailable"
    assert conv.inspection.reason == "unsupported_rank"
    assert not any("mtp." in p.name or "visual." in p.name for p in g.parameters)


@pytest.mark.parametrize(
    "name",
    [
        PREFIX + ".norm.weight",
        PREFIX + ".layers.1.self_attn.q_proj.weight",
        PREFIX + ".embed_tokens.weight",
    ],
)
def test_missing_parameter_retains_partial_structure(name: str) -> None:
    f = copy.deepcopy(TINY)
    prefix = name.removesuffix(".weight")
    f["storage"] = {
        k: v
        for k, v in f["storage"].items()
        if k != name and not (k.startswith(prefix + ".") and k.endswith(("_scale", "_scale_2")))
    }
    result = registry().analyze(metadata(f))
    assert result.status == "partial", result
    assert result.graph is not None and result.graph.diagnostics


@pytest.mark.parametrize(
    "change",
    [
        "unknown_layer",
        "gate_disabled",
        "bias",
        "activation",
        "mlp_only",
        "moe",
        "rope",
        "tying",
        "missing_quant",
        "malformed_quant",
        "missing_scale",
        "wrong_scale",
        "wrong_projection",
        "wrong_norm",
        "unknown_weight",
        "unknown_option",
        "missing_head_source_only",
        "extra_head",
        "wrong_mrope",
    ],
)
def test_reject_contradictory_or_unreviewed_metadata(change: str) -> None:
    f = copy.deepcopy(TINY)
    c = f["configuration"]
    t = c["text_config"]
    s = f["storage"]
    p = PREFIX + ".layers.1.self_attn.q_proj"
    if change == "unknown_layer":
        t["layer_types"][0] = "mamba"
    elif change == "gate_disabled":
        t["attn_output_gate"] = False
    elif change == "bias":
        t["attention_bias"] = True
    elif change == "activation":
        t["hidden_act"] = "gelu"
    elif change == "mlp_only":
        t["mlp_only_layers"] = [0]
    elif change == "moe":
        c["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    elif change == "rope":
        t["rope_parameters"]["rope_type"] = "yarn"
    elif change == "tying":
        t["tie_word_embeddings"] = False
    elif change == "missing_quant":
        del c["quantization_config"]
    elif change == "malformed_quant":
        c["quantization_config"]["quant_algo"] = "FP8"
    elif change == "missing_scale":
        del s[p + ".weight_scale"]
    elif change == "wrong_scale":
        s[p + ".weight_scale"]["dtype"] = "F16"
    elif change == "wrong_projection":
        t["num_attention_heads"] = 4
    elif change == "wrong_norm":
        s[PREFIX + ".norm.weight"]["shape"] = [64]
    elif change == "unknown_weight":
        s[PREFIX + ".unknown.weight"] = {"dtype": "BF16", "shape": [32]}
    elif change == "unknown_option":
        t["new_attention_algorithm"] = True
    elif change == "missing_head_source_only":
        del s[p + ".weight"]
    elif change == "extra_head":
        s["lm_head.weight"] = {"dtype": "BF16", "shape": [64, 32]}
    elif change == "wrong_mrope":
        t["rope_parameters"]["mrope_section"] = [11, 11, 10]
    result = registry().analyze(metadata(f))
    assert result.status == "unavailable", result


def test_changed_order_and_default_pattern() -> None:
    f = copy.deepcopy(TINY)
    f["configuration"]["text_config"]["layer_types"] = ["full_attention", "linear_attention"]
    f["storage"] = {
        k.replace("layers.0.", "layers.X.")
        .replace("layers.1.", "layers.0.")
        .replace("layers.X.", "layers.1."): v
        for k, v in f["storage"].items()
    }
    g = graph(metadata(f))
    assert [i.variant for i in g.repetitions[0].instances] == ["full_attention", "linear_attention"]
    f = copy.deepcopy(TINY)
    del f["configuration"]["text_config"]["layer_types"]
    f["configuration"]["text_config"]["full_attention_interval"] = 2
    assert graph(metadata(f)).coverage == "complete"


def test_local_readonly_analysis_has_no_execution_network_or_weight_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models"
    folder = root / "tiny"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(TINY["configuration"]))
    write_storage(
        folder / "model.safetensors",
        [(k, v["dtype"], v["shape"]) for k, v in TINY["storage"].items()],
    )
    for f in folder.iterdir():
        f.chmod(0o444)
    folder.chmod(0o555)
    source = ModelCatalogue(root).discover()[0].pin()
    forbidden = Mock(side_effect=AssertionError("Forbidden execution or data access"))
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
        (torch.fx, ["symbolic_trace"]),
        (transformers.AutoModel, ["from_config", "from_pretrained"]),
        (transformers.AutoProcessor, ["from_pretrained"]),
        (transformers.GenerationMixin, ["generate"]),
        (ModelSource, ["iter_tensor", "iter_rows", "local_directory"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    inputs = AnalysisInput.from_source(source, tokenizer_available=False)
    result = registry().analyze(inputs)
    assert result.status == "complete", result
    forbidden.assert_not_called()
