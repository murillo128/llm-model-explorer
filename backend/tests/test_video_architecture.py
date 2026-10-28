"""Static conformance against the selected headers, never checkpoint acceptance."""

import copy
import hashlib

import numpy as np
import pytest

from llm_model_explorer.architecture_analysis import AnalysisInput
from llm_model_explorer.architecture_analysis import records as r
from llm_model_explorer.architecture_analysis.validation import BindingContext, NumericTensor
from llm_model_explorer.architecture_service import packaged_registry
from llm_model_explorer.video_profiles import PROFILES, declaration, inventory


def metadata(profile_key: str) -> AnalysisInput:
    profile = PROFILES[profile_key]
    physical: dict[str, r.ArchitectureStorage] = {}
    numeric: dict[str, NumericTensor] = {}
    for role, key in profile.components.items():
        for suffix, (dtype, dims) in inventory(key).items():
            name = f"video.{role}.{suffix}"
            physical[name] = r.ArchitectureStorage(name=name, dtype=dtype, shape=list(dims))
            if suffix.endswith(("scale_weight", "scale_input", "scaled_fp8")):
                continue
            identity = hashlib.sha256(name.encode()).hexdigest()
            numeric[identity] = NumericTensor(
                identity,
                name,
                dims,
                dtype,
                "fp8-e4m3-scaled" if dtype == "F8_E4M3" else "safetensors",
            )
    config = {
        "model_type": "video_inspection",
        "architectures": [profile.architecture],
        "video_inspection": declaration(profile),
    }
    if profile.family == "wan":
        config["quantization_config"] = {
            "quant_method": "fp8-e4m3-scaled",
            "format_version": 1,
            "components": {
                f"video.{role}": "legacy"
                for role in profile.components
                if role in ("high_noise", "low_noise", "text_encoder")
            },
        }
    return AnalysisInput(
        "synthetic-metadata-not-checkpoint-bytes", config, BindingContext(physical, numeric, True)
    )


@pytest.fixture(scope="module", params=list(PROFILES))
def graph(request: pytest.FixtureRequest) -> tuple[str, r.ArchitectureGraph]:
    result = packaged_registry().analyze(metadata(str(request.param)))
    assert result.graph is not None, result.diagnostics
    assert result.status == "complete"
    return str(request.param), result.graph


def attributes(node: r.ArchitectureNode) -> dict[str, object]:
    return {a.name: a.value for a in node.attributes}


def test_modalities_loop_and_native_provenance(graph: tuple[str, r.ArchitectureGraph]) -> None:
    key, document = graph
    assert document.scope == "video_generation"
    root = next(n for n in document.nodes if n.parent_id is None)
    assert {p.id for p in root.ports} == (
        {"prompt", "configuration", "video"}
        if key == "wan-t2v-a14b"
        else {"prompt", "configuration", "image", "video"}
    )
    assert next(p for p in root.ports if p.id == "prompt").value_type == "text"
    assert next(p for p in root.ports if p.id == "video").value_type == "RGB video frames"
    assert attributes(root)["image"] == (
        "absent"
        if key == "wan-t2v-a14b"
        else "required"
        if key == "wan-i2v-a14b"
        else "optional in image_to_video mode"
    )
    operations = {n.operation for n in document.nodes}
    assert {
        "local_tokenization",
        "embedding_lookup",
        "scheduler_step",
        "decode_when_schedule_complete",
        "latent_state",
        "solver_history_update",
    } <= operations
    assert "greedy_selection" not in operations
    assert any(edge.kind == "state" for edge in document.edges)
    assert all(p.source != "model-defined-json" for n in document.nodes for p in n.provenance)
    assert all(
        not p.name.endswith(
            ("scale_weight", "scale_input", "scaled_fp8", "weight_scale", "input_scale")
        )
        for p in document.parameters
    )


def test_expert_threshold_and_noninitial_bindings(graph: tuple[str, r.ArchitectureGraph]) -> None:
    key, document = graph
    selections = [n for n in document.nodes if n.operation == "timestep_expert_selection"]
    denoiser_repetitions = [
        rep for rep in document.repetitions if rep.instances[0].variant.startswith(("ltx-", "wan-"))
    ]
    if key in ("wan-t2v-a14b", "wan-i2v-a14b"):
        assert len(selections) == 1 and len(denoiser_repetitions) == 2
        assert attributes(selections[0])["boundary"] == (0.875 if key == "wan-t2v-a14b" else 0.9)
        assert attributes(selections[0])["num_train_timesteps"] == 1000
        assert ">=" in (selections[0].formula or "")
        names = {p.name: p for p in document.parameters}
        high = names["video.high_noise.blocks.1.self_attn.q.weight"]
        low = names["video.low_noise.blocks.1.self_attn.q.weight"]
        assert high.id != low.id and high.inspection.status == low.inspection.status == "available"
        assert high.inspection.tensor_id != low.inspection.tensor_id
        assert high.binding == low.binding == "quantized"
        assert all(rep.instances[-1].index == 39 for rep in denoiser_repetitions)
    else:
        assert not selections and len(denoiser_repetitions) == 1
        assert denoiser_repetitions[0].instances[-1].index == (27 if key.startswith("ltx") else 29)
    assert all(
        p.inspection.reason == "unsupported_rank"
        for p in document.parameters
        if len(p.logical_shape or []) > 2 and p.inspection.status == "unavailable"
    )


def test_source_specific_vae_and_text_semantics(graph: tuple[str, r.ArchitectureGraph]) -> None:
    key, document = graph
    operations = {n.operation for n in document.nodes}
    names = {p.name for p in document.parameters}
    assert "video.text_encoder.shared.weight" in names
    assert "video.text_encoder.encoder.block.23.layer.1.DenseReluDense.wo.weight" in names
    assert not any("lm_head" in name or "clip" in name.lower() for name in names)
    if key.startswith("ltx"):
        assert "classifier_free_guidance" not in operations
        assert "pixel_shuffle3d" in operations and "decode_noise_blend" in operations
        assert "video.vae.decoder.up_blocks.6.res_blocks.4.conv2.conv.weight" in names
        # LTX's pinned vae_encode samples the uniform-log-variance posterior and
        # normalizes channels using these exact stored statistics, reversed at decode.
        assert {"posterior_sample", "latent_normalization", "latent_denormalization"} <= operations
        statistics = {
            "video.vae.per_channel_statistics.mean-of-means",
            "video.vae.per_channel_statistics.std-of-means",
        }
        parameters = {p.id: p.name for p in document.parameters}
        for node in document.nodes:
            if node.operation in ("latent_normalization", "latent_denormalization"):
                assert {parameters[p] for p in node.parameter_ids} == statistics
            if node.operation == "pixel_norm":
                assert attributes(node)["eps"] == 1e-8
            if node.operation == "rms_norm" and any(
                parameters[p].endswith(("q_norm.weight", "k_norm.weight"))
                for p in node.parameter_ids
            ):
                assert attributes(node)["eps"] == 1e-5
        assert "zero_pad_text_context" not in operations
    else:
        assert "classifier_free_guidance" in operations
        assert "latent_denormalization" in operations
        # Wan T5EncoderModel trims padding before WanModel pads with zeros to 512
        # and projects every row; padded T5 hidden values must not reach cross attention.
        assert "trim_text_padding" in operations
        pads = [n for n in document.nodes if n.operation == "zero_pad_text_context"]
        assert len(pads) == (1 if key == "wan-ti2v-5b" else 2)
        for pad in pads:
            output = next(p for p in pad.ports if p.direction == "output")
            assert output.shape is not None
            assert output.shape[1] == r.ArchitectureConstantDimension(kind="constant", value=512)
        cross = {n.id for n in document.nodes if n.label == "Cross attention"}
        for node in document.nodes:
            if node.parent_id in cross and node.operation == "attention_scores":
                assert "mask" not in {p.id for p in node.ports}
                output = next(p for p in node.ports if p.direction == "output")
                assert output.shape is not None
                assert output.shape[-1] == r.ArchitectureConstantDimension(
                    kind="constant", value=512
                )
        if key == "wan-ti2v-5b":
            assert {"average_space_to_depth", "duplicate_depth_to_space"} <= operations
            assert "video.vae.decoder.upsamples.3.upsamples.2.residual.6.weight" in names
        else:
            assert "video.vae.decoder.upsamples.14.residual.6.weight" in names


def test_source_specific_positional_dependencies(graph: tuple[str, r.ArchitectureGraph]) -> None:
    key, document = graph
    operations = {n.operation for n in document.nodes}
    if not key.startswith("ltx"):
        assert "temporal_coordinates_seconds" not in operations
        assert "ltx_rotary_frequencies" not in operations
        return

    # Pinned LTX pipeline/vae_encode/transformer3d: frame rate is a real input,
    # not inferable from the grid. Both releases normalize seconds/pixels by
    # [20, 2048, 2048]; only 0.9.6 has the causal first-frame correction.
    by_id = {n.id: n for n in document.nodes}
    seconds_nodes = [n for n in document.nodes if n.operation == "temporal_coordinates_seconds"]
    assert len(seconds_nodes) == 1, "LTX positional encoding must depend on frame rate"
    seconds = seconds_nodes[0]
    grid = next(n for n in document.nodes if n.operation == "latent_grid_positions")
    pixels = next(n for n in document.nodes if n.operation == "latent_to_pixel_coordinates")
    normalized = next(n for n in document.nodes if n.operation == "positional_normalization")
    frequencies = next(n for n in document.nodes if n.operation == "ltx_rotary_frequencies")
    rate = next(n for n in document.nodes if attributes(n).get("field") == "frame_rate")
    denoiser = by_id[seconds.parent_id]  # type: ignore[index]
    loop = by_id[denoiser.parent_id]  # type: ignore[index]
    root = next(n for n in document.nodes if n.parent_id is None)
    edges = {
        (e.source.node_id, e.source.port_id, e.target.node_id, e.target.port_id)
        for e in document.edges
    }
    assert {
        (root.id, "configuration", loop.id, "configuration"),
        (loop.id, "configuration", denoiser.id, "configuration"),
        (denoiser.id, "configuration", rate.id, "configuration"),
        (denoiser.id, "latent", grid.id, "latent"),
        (grid.id, "out", pixels.id, "coords"),
        (pixels.id, "out", seconds.id, "pixels"),
        (rate.id, "out", seconds.id, "frame_rate"),
        (seconds.id, "out", normalized.id, "coordinates"),
        (normalized.id, "out", frequencies.id, "positions"),
    } <= edges
    assert attributes(pixels)["scale_factors"] == [8, 32, 32]
    assert attributes(pixels)["causal_fix"] is (key == "ltx-0.9.6")
    assert pixels.formula == (
        "out = coords * reshape(scale_factors, [1, 3, 1]); "
        "out[:,0,:] = max(out[:,0,:] + 1 - scale_factors[0], 0) if causal_fix else out[:,0,:]"
    )
    assert attributes(normalized)["max_pos"] == [20, 2048, 2048]
    assert normalized.formula == (
        "out = transpose(coordinates / reshape(max_pos, [1, 3, 1]), [0, 2, 1])"
    )
    assert attributes(frequencies)["theta"] == 10000
    assert attributes(frequencies)["frequency_count"] == 341
    assert attributes(frequencies)["identity_prefix"] == 2

    # The pinned pipeline's unpatchify takes latent geometry, independently of
    # its frame-rate-dependent RoPE. Its grid must describe T,H,W, not cos/sin.
    unpatchify = next(n for n in document.nodes if n.operation == "unpatchify_latents")
    grid_inputs = [
        e.source
        for e in document.edges
        if e.target.node_id == unpatchify.id and e.target.port_id == "grid"
    ]
    assert len(grid_inputs) == 1
    geometry = by_id[grid_inputs[0].node_id]
    assert geometry.operation == "latent_grid_shape", "Unpatchify needs geometry, not RoPE"
    assert geometry.formula == "out = shape(latent)[-3:]"
    assert attributes(geometry)["axes"] == "time,height,width"
    assert {p.id for p in geometry.ports if p.direction == "input"} == {"latent"}
    assert (denoiser.id, "latent", geometry.id, "latent") in edges
    dimensions = [r.ArchitectureConstantDimension(kind="constant", value=3)]
    assert next(p for p in geometry.ports if p.id == "out").shape == dimensions
    assert next(p for p in unpatchify.ports if p.id == "grid").shape == dimensions
    assert next(p for p in unpatchify.ports if p.id == "out").shape == [
        r.ArchitectureSymbolDimension(kind="symbol", name="B"),
        r.ArchitectureConstantDimension(kind="constant", value=128),
        *(r.ArchitectureSymbolDimension(kind="symbol", name=name) for name in ("T", "H", "W")),
    ]

    # Evaluate only the explicitly bounded displayed expression, not an inference
    # implementation. This guards division, axis/layout and unchanged spatial
    # coordinates with independent examples: frame 8 is .32 s at 25 fps, .16 at 50.
    assert seconds.formula == "out = pixels / reshape([frame_rate, 1, 1], [1, 3, 1])"
    expression = seconds.formula.removeprefix("out = ")
    for frame_rate, expected in ((25, 0.32), (50, 0.16)):
        actual = eval(
            expression,
            {"__builtins__": {}, "reshape": np.reshape},
            {"pixels": np.array([[[8.0], [32.0], [64.0]]]), "frame_rate": frame_rate},
        )
        np.testing.assert_allclose(actual, [[[expected], [32.0], [64.0]]], rtol=0, atol=1e-15)

    rotations = [n for n in document.nodes if n.operation == "rotary_embedding_3d"]
    assert len(rotations) == 56  # Q and K in each of 28 real blocks, including block 1.
    rotary_blocks = set()
    for rotation in rotations:
        attention = by_id[rotation.parent_id]  # type: ignore[index]
        block = by_id[attention.parent_id]  # type: ignore[index]
        rotary_blocks.add(block.id)
        assert {
            (frequencies.id, "out", block.id, "frequencies"),
            (block.id, "frequencies", attention.id, "frequencies"),
            (attention.id, "frequencies", rotation.id, "frequencies"),
        } <= edges
        assert rotation.formula == (
            "out = x * frequencies[...,0] + rotate_pairs(x) * frequencies[...,1]"
        )
        assert attributes(rotation)["rotate_pairs"] == "(a,b) -> (-b,a)"
        heads = [by_id[e.target.node_id] for e in document.edges if e.source.node_id == rotation.id]
        assert len(heads) == 1 and heads[0].operation == "split_heads"
    assert {
        (e.target.node_id, e.target.port_id)
        for e in document.edges
        if e.source.node_id == frequencies.id
    } == {(block, "frequencies") for block in rotary_blocks}


@pytest.mark.parametrize(
    "defect",
    ["missing_expert", "mixed_modality", "wrong_shape", "wrong_config", "wrong_role_provenance"],
)
def test_invalid_native_profile_cannot_fall_back_to_a_language_model(defect: str) -> None:
    inputs = metadata("wan-t2v-a14b")
    physical = dict(inputs.bindings.physical)
    config = copy.deepcopy(dict(inputs.configuration))
    if defect == "missing_expert":
        physical = {
            name: item for name, item in physical.items() if not name.startswith("video.low_noise.")
        }
    elif defect == "mixed_modality":
        name = "video.low_noise.patch_embedding.weight"
        physical[name] = physical[name].model_copy(update={"shape": [5120, 36, 1, 2, 2]})
    elif defect == "wrong_shape":
        name = "video.high_noise.blocks.1.self_attn.q.weight"
        physical[name] = physical[name].model_copy(update={"shape": [5120, 5119]})
    else:
        declaration_value = config["video_inspection"]
        assert isinstance(declaration_value, dict)
        if defect == "wrong_config":
            declaration_value["modalities"] = ["image_to_video"]
        else:
            components = declaration_value["components"]
            components["high_noise"]["sources"] = components["low_noise"]["sources"]
    result = packaged_registry().analyze(
        AnalysisInput(
            inputs.fingerprint, config, BindingContext(physical, inputs.bindings.numeric, True)
        )
    )
    assert result.status == "unavailable" and result.reason == "unsupported_architecture"


def test_reference_inventory_constants_are_not_constructor_defaults() -> None:
    assert inventory("wan-ti2v")["patch_embedding.weight"] == ("F16", (3072, 48, 1, 2, 2))
    assert inventory("wan-ti2v")["head.head.weight"] == ("F16", (192, 3072))
    assert inventory("wan-vae-2.2")["decoder.conv1.weight"] == ("F16", (1024, 48, 3, 3, 3))
    assert inventory("wan-i2v")["patch_embedding.weight"][1] == (5120, 36, 1, 2, 2)
    assert inventory("umt5")["shared.weight"][1] == (256384, 4096)
    assert inventory("t5")["shared.weight"][1] == (32128, 4096)
