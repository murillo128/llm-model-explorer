"""Static BDB-2025 world-model architecture description tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import Mock

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    AnalysisResult,
    BindingContext,
    DescriptionRegistry,
    NumericTensor,
)
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.nfl_world_model import (
    PRODUCER,
    register_nfl_world_model,
)
from llm_model_explorer.architecture_analysis.nfl_world_model_config import (
    configuration,
    parameter_shapes,
)
from llm_model_explorer.architecture_service import ArchitectureService


def fixture_config() -> dict[str, object]:
    return {
        "model_type": "nfl_world_model",
        "architectures": ["BDB2025LeWorldModel"],
        "torch_dtype": "float32",
        "synthetic_checkpoint": True,
        "weights_status": "random_initialization",
        "random_seed": 20260917,
        "preset": "tiny-model-explorer",
        "source_model": "src/nfl_world_model/bdb2025/le_world_model.py::BDB2025LeWorldModel",
        "encoder": {
            "entity_continuous_width": 11,
            "entity_categorical_cardinalities": [4, 5, 4],
            "frame_continuous_width": 8,
            "frame_categorical_cardinalities": [7, 6],
            "categorical_padding_index": 0,
            "d_model": 8,
            "attention_heads": 2,
            "spatial_layers": 1,
            "temporal_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
        },
        "predictor": {
            "d_model": 8,
            "hidden_dim": 16,
            "hidden_layers": 1,
            "dropout": 0.0,
        },
        "sigreg": {"num_proj": 1024, "num_knots": 17, "t_max": 3.0, "weight": 0.1},
        "input_schema": {
            "entity_continuous": [
                "x", "y", "s", "a", "dis", "orientation_sin", "orientation_cos",
                "direction_sin", "direction_cos", "height_in", "weight_lb",
            ],
            "entity_categorical": ["entity_type", "side", "position"],
            "frame_continuous": [
                "yards_to_go", "game_clock_s", "pre_snap_home_score",
                "pre_snap_visitor_score", "field_position_x", "elapsed_play_s",
                "delta_t_s", "time_since_snap_s",
            ],
            "frame_categorical": ["quarter", "down"],
        },
    }


def make_inputs(
    *, partial: bool = False, wrong_shape: str | None = None, config_update: dict[str, Any] | None = None
) -> AnalysisInput:
    config = fixture_config()
    if partial:
        config["checkpoint_scope"] = "partial_structural_fixture"
        config["available_tensor_count"] = 10
    if config_update:
        config.update(config_update)

    empty = BindingContext({}, {}, False)
    provisional = AnalysisInput("nfl-fixture", config, empty)
    parsed = configuration(provisional)
    if parsed is None:
        return provisional
    shapes = parameter_shapes(parsed)
    if partial:
        selected = {
            "encoder.encoder.entity_continuous_projection.weight",
            "encoder.encoder.frame_continuous_projection.weight",
            "encoder.encoder.play_token",
            "encoder.encoder.spatial_encoder.layers.0.self_attn.in_proj_weight",
            "encoder.encoder.spatial_encoder.layers.0.self_attn.out_proj.weight",
            "encoder.encoder.temporal_encoder.layers.0.self_attn.in_proj_weight",
            "encoder.encoder.temporal_encoder.layers.0.self_attn.out_proj.weight",
            "predictor.input_projection.weight",
            "predictor.output_projection.weight",
            "predictor.time_projection.0.weight",
        }
    else:
        selected = set(shapes)

    if wrong_shape is not None:
        shapes = dict(shapes)
        shapes[wrong_shape] = [999]

    physical = {
        name: r.ArchitectureStorage(name=name, dtype="F32", shape=shapes[name])
        for name in selected
    }
    numeric = {
        f"tensor-{index}": NumericTensor(
            f"tensor-{index}", name, tuple(shapes[name]), "F32", "safetensors"
        )
        for index, name in enumerate(sorted(selected))
    }
    return AnalysisInput("nfl-fixture", config, BindingContext(physical, numeric, False))


def analyze(inputs: AnalysisInput) -> AnalysisResult:
    registry = DescriptionRegistry()
    register_nfl_world_model(registry)
    return registry.analyze(inputs)


def attributes(node: r.ArchitectureNode) -> dict[str, object]:
    return {attribute.name: attribute.value for attribute in node.attributes}


def test_complete_graph_has_structured_causal_encoder_predictor_and_training_boundary() -> None:
    result = analyze(make_inputs())
    assert result.status == "complete", result.diagnostics
    assert result.graph is not None
    graph = result.graph
    assert graph.scope == "visual_encoder_predictor"
    assert not graph.diagnostics

    by_operation: dict[str, list[r.ArchitectureNode]] = {}
    for node in graph.nodes:
        if node.operation is not None:
            by_operation.setdefault(node.operation, []).append(node)

    structured = by_operation["structured_input"]
    assert len(structured) == 11
    root = next(node for node in graph.nodes if node.label == "BDB-2025 LeWorldModel inference path")
    assert attributes(root)["input_modality"] == "structured_nfl_tracking"
    assert attributes(root)["future_observation_input"] is False

    scores = by_operation["scaled_query_key_product"]
    assert len(scores) == 2
    assert {attributes(node)["causal"] for node in scores} == {False, True}
    temporal = next(node for node in graph.nodes if node.label == "Temporal encoder")
    assert attributes(temporal)["causal"] is True
    assert attributes(temporal)["future_visible"] is False

    assert "last_valid_play_latent" in by_operation
    assert "repeat_current_latent_over_horizons" in by_operation
    assert "mask_invalid_horizon_outputs" in by_operation
    predictor = next(node for node in graph.nodes if node.label == "Horizon latent predictor")
    assert attributes(predictor)["future_observation_input"] is False

    training = next(node for node in graph.nodes if node.label == "Training-only objective")
    assert attributes(training) == {"training_only": True, "live_inference": False}
    target_context = next(node for node in graph.nodes if node.operation == "training_context")
    assert target_context.kind == "context"
    assert "masked_next_embedding_mse" in by_operation
    assert "sketched_isotropic_gaussian_regularization" in by_operation

    repetitions = {rep.label: rep for rep in graph.repetitions}
    assert len(repetitions["Spatial layers"].instances) == 1
    assert len(repetitions["Temporal layers"].instances) == 1
    assert len(repetitions["Predictor residual MLP blocks"].instances) == 1

    qkv = next(
        parameter
        for parameter in graph.parameters
        if parameter.name
        == "encoder.encoder.spatial_encoder.layers.0.self_attn.in_proj_weight"
    )
    assert qkv.logical_shape is not None
    assert [
        dimension.value
        for dimension in qkv.logical_shape
        if isinstance(dimension, r.ArchitectureConstantDimension)
    ] == [24, 8]
    assert qkv.inspection.status == "available"


def test_partial_structural_fixture_preserves_graph_and_localizes_missing_weights() -> None:
    result = analyze(make_inputs(partial=True))
    assert result.status == "partial"
    assert result.graph is not None
    assert any(node.operation == "last_valid_play_latent" for node in result.graph.nodes)
    unresolved = [d for d in result.graph.diagnostics if d.code == "unresolved_binding"]
    assert unresolved
    assert len(result.graph.parameters) > 10


def test_unknown_structure_changing_config_fails_closed() -> None:
    inputs = make_inputs()
    config = dict(inputs.configuration)
    raw_encoder = config["encoder"]
    assert isinstance(raw_encoder, dict)
    encoder = dict(raw_encoder)
    encoder["mystery_attention_mode"] = "future"
    config["encoder"] = encoder
    result = analyze(replace(inputs, configuration=config))
    assert result.status == "unavailable"
    assert result.reason == "unsupported_architecture"


def test_wrong_observed_parameter_geometry_fails_selection() -> None:
    name = "encoder.encoder.spatial_encoder.layers.0.self_attn.in_proj_weight"
    result = analyze(make_inputs(wrong_shape=name))
    assert result.status == "unavailable"
    assert result.reason == "unsupported_architecture"


def test_architecture_service_registers_nfl_description_without_model_construction() -> None:
    service = ArchitectureService(Mock(), Mock())
    selected = service.registry.select(make_inputs())
    assert selected is not None
    assert selected.producer == PRODUCER
