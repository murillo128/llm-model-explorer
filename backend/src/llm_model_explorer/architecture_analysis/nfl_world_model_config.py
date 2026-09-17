"""Checked metadata for the BDB-2025 structured NFL world model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .core import AnalysisInput

TOP = {
    "model_type", "architectures", "_name_or_path", "name_or_path", "_commit_hash",
    "revision", "torch_dtype", "dtype", "synthetic_checkpoint", "weights_status",
    "random_seed", "preset", "source_model", "encoder", "predictor", "sigreg",
    "input_schema", "available_tensor_count", "checkpoint_scope",
}
ENCODER = {
    "entity_continuous_width", "entity_categorical_cardinalities", "frame_continuous_width",
    "frame_categorical_cardinalities", "categorical_padding_index", "d_model",
    "attention_heads", "spatial_layers", "temporal_layers", "dim_feedforward", "dropout",
}
PREDICTOR = {"d_model", "hidden_dim", "hidden_layers", "dropout"}
SIGREG = {"num_proj", "num_knots", "t_max", "weight"}
SCHEMA = {"entity_continuous", "entity_categorical", "frame_continuous", "frame_categorical"}


@dataclass(frozen=True)
class Config:
    ec: int
    entity_cards: tuple[int, ...]
    fc: int
    frame_cards: tuple[int, ...]
    padding: int
    d: int
    heads: int
    spatial_layers: int
    temporal_layers: int
    ff: int
    encoder_dropout: float
    predictor_hidden: int
    predictor_layers: int
    predictor_dropout: float
    sigreg_proj: int
    sigreg_knots: int
    sigreg_t_max: float
    sigreg_weight: float

    @property
    def head_dim(self) -> int:
        return self.d // self.heads


def _map(value: object, fields: set[str]) -> Mapping[str, object] | None:
    return value if isinstance(value, dict) and set(value) == fields else None


def _int(value: object, maximum: int = 1_000_000) -> int | None:
    return value if type(value) is int and 0 < value <= maximum else None


def _float(value: object, *, zero: bool = False) -> float | None:
    if type(value) not in (int, float):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not zero and result == 0):
        return None
    return result


def _cards(value: object, padding: int) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    result = tuple(value)
    if any(type(x) is not int or x <= 0 or not 0 <= padding < x for x in result):
        return None
    return result


def configuration(inputs: AnalysisInput) -> Config | None:
    raw = inputs.configuration
    if set(raw) - TOP:
        return None
    e = _map(raw.get("encoder"), ENCODER)
    p = _map(raw.get("predictor"), PREDICTOR)
    s = _map(raw.get("sigreg"), SIGREG)
    schema = _map(raw.get("input_schema"), SCHEMA)
    if None in (e, p, s, schema):
        return None
    assert e is not None and p is not None and s is not None and schema is not None
    padding = e["categorical_padding_index"]
    if type(padding) is not int or padding < 0:
        return None
    values = {
        "ec": _int(e["entity_continuous_width"]),
        "fc": _int(e["frame_continuous_width"]),
        "d": _int(e["d_model"]),
        "heads": _int(e["attention_heads"], 1024),
        "spatial": _int(e["spatial_layers"], 256),
        "temporal": _int(e["temporal_layers"], 256),
        "ff": _int(e["dim_feedforward"]),
        "hidden": _int(p["hidden_dim"]),
        "pred_layers": _int(p["hidden_layers"], 256),
        "proj": _int(s["num_proj"]),
        "knots": _int(s["num_knots"]),
    }
    if any(v is None for v in values.values()):
        return None
    ec, fc, d, heads = values["ec"], values["fc"], values["d"], values["heads"]
    spatial, temporal, ff = values["spatial"], values["temporal"], values["ff"]
    hidden, pred_layers = values["hidden"], values["pred_layers"]
    proj, knots = values["proj"], values["knots"]
    assert all(isinstance(v, int) for v in (ec, fc, d, heads, spatial, temporal, ff, hidden, pred_layers, proj, knots))
    entity_cards = _cards(e["entity_categorical_cardinalities"], padding)
    frame_cards = _cards(e["frame_categorical_cardinalities"], padding)
    edrop = _float(e["dropout"], zero=True)
    pdrop = _float(p["dropout"], zero=True)
    tmax = _float(s["t_max"])
    weight = _float(s["weight"], zero=True)
    if None in (entity_cards, frame_cards, edrop, pdrop, tmax, weight):
        return None
    assert entity_cards is not None and frame_cards is not None
    assert edrop is not None and pdrop is not None and tmax is not None and weight is not None
    if d % heads or p["d_model"] != d or knots < 3 or knots % 2 == 0:
        return None
    widths = {
        "entity_continuous": ec,
        "entity_categorical": len(entity_cards),
        "frame_continuous": fc,
        "frame_categorical": len(frame_cards),
    }
    for key, width in widths.items():
        names = schema[key]
        if not isinstance(names, list) or len(names) != width or len(set(names)) != width:
            return None
        if not all(isinstance(name, str) and name for name in names):
            return None
    if raw.get("checkpoint_scope") not in (None, "partial_structural_fixture"):
        return None
    count = raw.get("available_tensor_count")
    if count is not None and (type(count) is not int or count <= 0):
        return None
    return Config(ec, entity_cards, fc, frame_cards, padding, d, heads, spatial, temporal, ff,
                  edrop, hidden, pred_layers, pdrop, proj, knots, tmax, weight)


def parameter_shapes(c: Config) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    def linear(name: str, o: int, i: int) -> None:
        out[name + ".weight"] = [o, i]; out[name + ".bias"] = [o]
    def norm(name: str, width: int) -> None:
        out[name + ".weight"] = [width]; out[name + ".bias"] = [width]
    base = "encoder.encoder"
    linear(base + ".entity_continuous_projection", c.d, 2 * c.ec)
    for i, card in enumerate(c.entity_cards): out[f"{base}.entity_categorical_embeddings.{i}.weight"] = [card, c.d]
    norm(base + ".entity_input_norm", c.d)
    linear(base + ".frame_continuous_projection", c.d, 2 * c.fc)
    for i, card in enumerate(c.frame_cards): out[f"{base}.frame_categorical_embeddings.{i}.weight"] = [card, c.d]
    out[base + ".play_token"] = [c.d]; norm(base + ".frame_input_norm", c.d)
    for stack, count in (("spatial_encoder", c.spatial_layers), ("temporal_encoder", c.temporal_layers)):
        for i in range(count):
            layer = f"{base}.{stack}.layers.{i}"
            out[layer + ".self_attn.in_proj_weight"] = [3 * c.d, c.d]
            out[layer + ".self_attn.in_proj_bias"] = [3 * c.d]
            linear(layer + ".self_attn.out_proj", c.d, c.d)
            linear(layer + ".linear1", c.ff, c.d); linear(layer + ".linear2", c.d, c.ff)
            norm(layer + ".norm1", c.d); norm(layer + ".norm2", c.d)
    linear("predictor.time_projection.0", c.d, 1); linear("predictor.time_projection.2", c.d, c.d)
    linear("predictor.input_projection", c.predictor_hidden, 2 * c.d)
    for i in range(c.predictor_layers):
        block = f"predictor.hidden_blocks.{i}.net"
        norm(block + ".0", c.predictor_hidden)
        linear(block + ".1", c.predictor_hidden, c.predictor_hidden)
        linear(block + ".4", c.predictor_hidden, c.predictor_hidden)
    linear("predictor.output_projection", c.d, c.predictor_hidden)
    for name in ("sigreg.t", "sigreg.phi", "sigreg.weights"): out[name] = [c.sigreg_knots]
    return out


def supports(inputs: AnalysisInput) -> bool:
    c = configuration(inputs)
    if c is None:
        return False
    expected = parameter_shapes(c); physical = inputs.bindings.physical
    anchors = {
        "encoder.encoder.entity_continuous_projection.weight",
        "encoder.encoder.spatial_encoder.layers.0.self_attn.in_proj_weight",
        "encoder.encoder.temporal_encoder.layers.0.self_attn.in_proj_weight",
        "predictor.input_projection.weight",
    }
    return anchors <= set(physical) and not (set(physical) - expected.keys()) and all(
        tensor.shape == expected[name] and tensor.dtype in ("F32", "F16", "BF16")
        for name, tensor in physical.items()
    )
