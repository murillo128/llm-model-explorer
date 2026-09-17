"""Reviewed static description for the BDB-2025 structured NFL world model."""
from __future__ import annotations

from . import records as r
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .nfl_world_model_config import configuration, supports
from .nfl_world_model_graph import Graph, expression, shape

PRODUCER = Producer(
    "nfl-bdb2025-leworldmodel", "1",
    "murillo128/nfl-world-model@7fed1257eb74f45f98a9d5d66c0488fbe91b5109",
)


def build(inputs: AnalysisInput, builder: GraphBuilder) -> None:
    c = configuration(inputs); assert c is not None
    for name, meaning in {
        "B": "Symbolic batch size; no checkpoint input has been executed.",
        "T": "Observed time steps in the causal play prefix.",
        "N": "Entity slots (players plus football when present).",
        "K": "Requested future physical-time horizons.",
        "BT": "B * T same-time spatial-attention sequences.",
        "M": "B * (N + 1) play/entity trajectories for the shared temporal encoder.",
    }.items(): builder.add_symbol(name, meaning)
    g = Graph(builder, c, PRODUCER); d = c.d
    ec, fc = c.ec, c.fc; ce, cf = len(c.entity_cards), len(c.frame_cards)

    ins = {
        "entity_continuous": g.input("entity_continuous", shape("B", "T", "N", ec)),
        "entity_categorical": g.input("entity_categorical", shape("B", "T", "N", ce)),
        "entity_mask": g.input("entity_mask", shape("B", "T", "N")),
        "entity_continuous_mask": g.input("entity_continuous_mask", shape("B", "T", "N", ec)),
        "frame_continuous": g.input("frame_continuous", shape("B", "T", fc)),
        "frame_categorical": g.input("frame_categorical", shape("B", "T", cf)),
        "frame_continuous_mask": g.input("frame_continuous_mask", shape("B", "T", fc)),
        "time_mask": g.input("time_mask", shape("B", "T")),
        "entity_slot_mask": g.input("entity_slot_mask", shape("B", "N")),
        "delta_t_s": g.input("delta_t_s", shape("B", "K")),
        "horizon_mask": g.input("horizon_mask", shape("B", "K")),
    }
    model = g.group("world_model", ins)
    enc_names = tuple(name for name in ins if name not in {"delta_t_s", "horizon_mask"})
    enc_in = {name: model[name] for name in enc_names}; enc = g.group("encoder", enc_in)

    entity_features = g.op("encoder.entity_features", "masked_value_and_presence_concat",
        {"values": enc["entity_continuous"], "availability": enc["entity_continuous_mask"],
         "entity_mask": enc["entity_mask"], "time_mask": enc["time_mask"], "slot_mask": enc["entity_slot_mask"]},
        shape("B", "T", "N", 2*ec), parent="encoder")
    entity_numeric = g.affine("encoder.encoder.entity_continuous_projection", entity_features,
        shape("B", "T", "N", d), "encoder", weight="encoder.encoder.entity_continuous_projection.weight",
        bias="encoder.encoder.entity_continuous_projection.bias")
    entity_cat = g.op("encoder.entity_categorical_embeddings", "categorical_embedding_sum",
        {"indices": enc["entity_categorical"], "entity_mask": enc["entity_mask"]}, shape("B", "T", "N", d),
        parent="encoder", parameters=tuple(f"encoder.encoder.entity_categorical_embeddings.{i}.weight" for i in range(ce)),
        attributes={"cardinalities": list(c.entity_cards)})
    entity_sum = g.op("encoder.entity_input_sum", "add", {"numeric": entity_numeric, "categorical": entity_cat}, shape("B", "T", "N", d), parent="encoder")
    entity_tokens = g.affine("encoder.encoder.entity_input_norm", entity_sum, shape("B", "T", "N", d), "encoder",
        weight="encoder.encoder.entity_input_norm.weight", bias="encoder.encoder.entity_input_norm.bias",
        operation="layer_norm", attributes={"epsilon": 1e-5})

    frame_features = g.op("encoder.frame_features", "masked_value_and_presence_concat",
        {"values": enc["frame_continuous"], "availability": enc["frame_continuous_mask"], "time_mask": enc["time_mask"]},
        shape("B", "T", 2*fc), parent="encoder")
    frame_numeric = g.affine("encoder.encoder.frame_continuous_projection", frame_features, shape("B", "T", d), "encoder",
        weight="encoder.encoder.frame_continuous_projection.weight", bias="encoder.encoder.frame_continuous_projection.bias")
    frame_cat = g.op("encoder.frame_categorical_embeddings", "categorical_embedding_sum",
        {"indices": enc["frame_categorical"], "time_mask": enc["time_mask"]}, shape("B", "T", d), parent="encoder",
        parameters=tuple(f"encoder.encoder.frame_categorical_embeddings.{i}.weight" for i in range(cf)),
        attributes={"cardinalities": list(c.frame_cards)})
    frame_sum = g.op("encoder.frame_input_sum", "add", {"numeric": frame_numeric, "categorical": frame_cat}, shape("B", "T", d), parent="encoder")
    play = g.op("encoder.add_play_token", "add_learned_play_token", {"x": frame_sum}, shape("B", "T", d), parent="encoder",
        parameters=("encoder.encoder.play_token",))
    play = g.affine("encoder.encoder.frame_input_norm", play, shape("B", "T", d), "encoder",
        weight="encoder.encoder.frame_input_norm.weight", bias="encoder.encoder.frame_input_norm.bias",
        operation="layer_norm", attributes={"epsilon": 1e-5})

    spatial = g.node("encoder.spatial_pack", "concat_play_and_entities_flatten_time",
        {"play": play, "entities": entity_tokens, "time_mask": enc["time_mask"], "entity_mask": enc["entity_mask"]},
        {"tokens": [*shape("BT"), expression("N + 1", "N"), *shape(d)],
         "padding_mask": [*shape("BT"), expression("N + 1", "N")]}, parent="encoder")
    s_in = g.group("encoder.encoder.spatial_encoder", {"x": spatial["tokens"], "padding_mask": spatial["padding_mask"]})
    sx = g.transformer("encoder.encoder.spatial_encoder", s_in["x"], s_in["padding_mask"], layers=c.spatial_layers,
        batch=r.ArchitectureSymbolDimension(kind="symbol", name="BT"), sequence=expression("N + 1", "N"), causal=False)
    sx = g.end_group("encoder.encoder.spatial_encoder", s_in, sx, parent="encoder", label="Spatial encoder",
        attributes={"same_time_attention": True, "causal": False})
    split = g.node("encoder.spatial_unpack", "unflatten_time_split_play_entities", {"x": sx},
        {"play": shape("B", "T", d), "entities": shape("B", "T", "N", d)}, parent="encoder")

    packed = g.node("encoder.temporal_pack", "pack_play_and_entity_trajectories",
        {"play": split["play"], "entities": split["entities"], "time_mask": enc["time_mask"], "entity_mask": enc["entity_mask"]},
        {"values": shape("M", "T", d), "validity": shape("M", "T")}, parent="encoder")
    positioned = g.node("encoder.temporal_positions", "first_valid_suffix_sinusoidal_positions",
        {"x": packed["values"], "validity": packed["validity"]},
        {"x": shape("M", "T", d), "padding_mask": shape("M", "T")}, parent="encoder",
        attributes={"causal_mask": True, "position_base": 10000})
    t_in = g.group("encoder.encoder.temporal_encoder", {"x": positioned["x"], "padding_mask": positioned["padding_mask"]})
    tx = g.transformer("encoder.encoder.temporal_encoder", t_in["x"], t_in["padding_mask"], layers=c.temporal_layers,
        batch=r.ArchitectureSymbolDimension(kind="symbol", name="M"), sequence=r.ArchitectureSymbolDimension(kind="symbol", name="T"), causal=True)
    tx = g.end_group("encoder.encoder.temporal_encoder", t_in, tx, parent="encoder", label="Temporal encoder",
        attributes={"causal": True, "future_visible": False})
    latents = g.node("encoder.temporal_unpack", "unpack_play_and_entity_trajectories", {"x": tx},
        {"play_latents": shape("B", "T", d), "entity_latents": shape("B", "T", "N", d)}, parent="encoder")
    play_latents = g.end_group("encoder", enc_in, latents["play_latents"], parent="world_model",
        label="BDB-2025 causal play encoder", attributes={"structured_entities": True, "future_visible": False})
    current = g.op("current_play_latent", "last_valid_play_latent", {"play_latents": play_latents, "time_mask": model["time_mask"]},
        shape("B", d), parent="world_model")

    pred_in = {"current_play_latent": current, "delta_t_s": model["delta_t_s"], "horizon_mask": model["horizon_mask"]}
    pred = g.group("predictor", pred_in)
    horizon = g.op("predictor.safe_horizons", "mask_invalid_horizons", {"delta_t_s": pred["delta_t_s"], "horizon_mask": pred["horizon_mask"]}, shape("B", "K"), parent="predictor")
    horizon = g.op("predictor.horizon_scalar", "unsqueeze", {"x": horizon}, shape("B", "K", 1), parent="predictor")
    time = g.affine("predictor.time_projection.0", horizon, shape("B", "K", d), "predictor", weight="predictor.time_projection.0.weight", bias="predictor.time_projection.0.bias")
    time = g.op("predictor.time_projection.silu", "silu", {"x": time}, time.shape, parent="predictor")
    time = g.affine("predictor.time_projection.2", time, shape("B", "K", d), "predictor", weight="predictor.time_projection.2.weight", bias="predictor.time_projection.2.bias")
    repeated = g.op("predictor.repeat_current_state", "repeat_current_latent_over_horizons", {"x": pred["current_play_latent"], "delta_t_s": pred["delta_t_s"]}, shape("B", "K", d), parent="predictor")
    joined = g.op("predictor.concat_state_time", "concatenate", {"state": repeated, "time": time}, shape("B", "K", 2*d), parent="predictor", attributes={"axis": -1})
    hidden = g.affine("predictor.input_projection", joined, shape("B", "K", c.predictor_hidden), "predictor", weight="predictor.input_projection.weight", bias="predictor.input_projection.bias")
    blocks = []
    for i in range(c.predictor_layers):
        key = f"predictor.hidden_blocks.{i}"; block = g.group(key, {"x": hidden})
        norm = g.affine(key + ".net.0", block["x"], hidden.shape, key, weight=key + ".net.0.weight", bias=key + ".net.0.bias", operation="layer_norm", attributes={"epsilon": 1e-5})
        delta = g.affine(key + ".net.1", norm, hidden.shape, key, weight=key + ".net.1.weight", bias=key + ".net.1.bias")
        delta = g.op(key + ".gelu", "gelu", {"x": delta}, delta.shape, parent=key)
        delta = g.affine(key + ".net.4", delta, hidden.shape, key, weight=key + ".net.4.weight", bias=key + ".net.4.bias")
        hidden = g.op(key + ".residual", "residual_add", {"skip": block["x"], "branch": delta}, hidden.shape, parent=key)
        hidden = g.end_group(key, block, hidden, parent="predictor", label=f"Residual MLP block {i}", role="mlp",
            attributes={"dropout": c.predictor_dropout, "evaluation_path": True})
        blocks.append(r.ArchitectureRepetitionInstance(node_id=g.nid(key), index=i, variant="residual_horizon_mlp"))
    builder.add_repetition(r.ArchitectureRepetition(id=builder.record_id("repetition", "predictor.hidden_blocks"),
        parent_id=g.nid("predictor"), label="Predictor residual MLP blocks", instances=blocks))
    delta = g.affine("predictor.output_projection", hidden, shape("B", "K", d), "predictor", weight="predictor.output_projection.weight", bias="predictor.output_projection.bias")
    output = g.op("predictor.state_residual", "residual_add", {"current": repeated, "delta": delta}, shape("B", "K", d), parent="predictor")
    output = g.op("predictor.masked_output", "mask_invalid_horizon_outputs", {"x": output, "horizon_mask": pred["horizon_mask"]}, output.shape, parent="predictor")
    output = g.end_group("predictor", pred_in, output, parent="world_model", label="Horizon latent predictor",
        attributes={"physical_time_conditioning": True, "future_observation_input": False})
    output = g.end_group("world_model", ins, output, label="BDB-2025 LeWorldModel inference path",
        attributes={"input_modality": "structured_nfl_tracking", "future_observation_input": False,
                    "scope_identifier_note": "visual_encoder_predictor is the legacy non-language scope identifier; this model has no visual input"},
        module_reference=False)
    g.node("predicted_play_latents", "predicted_play_latents", {"x": output}, {}, kind="output")

    target = g.input("training.target_play_latents", shape("B", "K", d), True)
    prefix = g.input("training.target_prefix_play_latents", shape("B", "T", d), True)
    prefix_mask = g.input("training.target_prefix_time_mask", shape("B", "T"), True)
    target_mask = g.input("training.target_mask", shape("B", "K"), True)
    train_in = {"predicted": output, "target": target, "target_prefix": prefix, "target_prefix_time_mask": prefix_mask, "target_mask": target_mask}
    train = g.group("training_objective", train_in)
    mse = g.op("training.prediction_loss", "masked_next_embedding_mse", {"predicted": train["predicted"], "target": train["target"], "mask": train["target_mask"]}, shape(), parent="training_objective", attributes={"training_only": True})
    sigreg = g.op("training.sigreg", "sketched_isotropic_gaussian_regularization", {"target_prefix": train["target_prefix"], "time_mask": train["target_prefix_time_mask"]}, shape(), parent="training_objective",
        parameters=("sigreg.t", "sigreg.phi", "sigreg.weights"), attributes={"training_only": True, "num_proj": c.sigreg_proj, "num_knots": c.sigreg_knots, "t_max": c.sigreg_t_max})
    total = g.op("training.total_loss", "weighted_sum", {"prediction_loss": mse, "sigreg_loss": sigreg}, shape(), parent="training_objective",
        formula="prediction_loss + sigreg_weight * sigreg_loss", attributes={"training_only": True, "sigreg_weight": c.sigreg_weight})
    g.end_group("training_objective", train_in, total, label="Training-only objective",
        attributes={"training_only": True, "live_inference": False}, module_reference=False)


DESCRIPTION = Description(PRODUCER, "visual_encoder_predictor", frozenset({"nfl_world_model"}),
                          frozenset({"BDB2025LeWorldModel"}), supports, build)


def register_nfl_world_model(registry: DescriptionRegistry) -> None:
    registry.register(DESCRIPTION)
