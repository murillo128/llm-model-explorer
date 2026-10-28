"""Source-backed T5/UMT5 encoders and LTX/Wan denoisers; static dataflow only."""

from ..video_profiles import VideoProfile
from .dense import shape
from .video_graph import Value, VideoGraph


def attention(
    g: VideoGraph,
    key: str,
    parent: str,
    x: Value,
    context: Value,
    module: str,
    *,
    heads: int,
    family: str,
    cross: bool = False,
    positions: Value | None = None,
    bias: Value | None = None,
    mask: Value | None = None,
) -> Value:
    external = {"x": x, "context": context}
    if positions is not None:
        external["positions"] = positions
    if bias is not None:
        external["bias"] = bias
    if mask is not None:
        external["mask"] = mask
    ports = g.begin(
        key,
        parent,
        "Cross attention" if cross else "Self attention",
        external,
        semantic_role="attention",
        attention_axes="text"
        if family == "t5"
        else "latent queries and text keys"
        if cross
        else "joint temporal and spatial",
    )
    values: dict[str, Value] = {}
    context_axis: str | int = 512 if family == "wan" and cross else "S"
    for name in ("q", "k", "v"):
        source = ports["x"] if name == "q" else ports["context"]
        suffix = f"to_{name}" if family == "ltx" else name
        projected = g.linear(key + "." + name, key, source, module + "." + suffix)
        hidden = g.inputs.bindings.physical[module + "." + suffix + ".weight"].shape[0]
        if name != "v" and family != "t5":
            norm = (name + "_norm") if family == "ltx" else ("norm_" + name)
            projected = g.unary(
                key + "." + norm,
                key,
                "rms_norm",
                projected,
                weights=(module + "." + norm + ".weight",),
                eps=1e-5 if family == "ltx" else 1e-6,
                formula="out = x * rsqrt(mean(x^2, -1) + eps) * weight",
            )
        sequence = context_axis if family == "t5" or (cross and name != "q") else "N"
        projected = g.op(
            key + "." + name + ".heads",
            key,
            "split_heads",
            {"x": projected},
            shape("B", heads, sequence, hidden // heads),
            layout="B,H,L,D",
        )
        if positions is not None and name != "v" and not cross:
            projected = g.op(
                key + "." + name + ".rope",
                key,
                "rotary_embedding_3d",
                {"x": projected, "positions": ports["positions"]},
                projected.dims,
                theta=10000,
                axes="time,height,width",
            )
        values[name] = projected
    query_axis = "S" if family == "t5" else "N"
    key_axis = context_axis if family == "t5" or cross else "N"
    score_inputs = {"q": values["q"], "k": values["k"]}
    if bias is not None:
        score_inputs["bias"] = ports["bias"]
    if mask is not None:
        score_inputs["mask"] = ports["mask"]
    scores = g.op(
        key + ".scores",
        key,
        "attention_scores",
        score_inputs,
        shape("B", heads, query_axis, key_axis),
        formula="out = q @ k^T + bias + padding_mask(mask)"
        if family == "t5"
        else "out = (q @ k^T) / sqrt(head_dim) + padding_mask(mask)"
        if mask is not None
        else "out = (q @ k^T) / sqrt(head_dim)",
        head_dim=hidden // heads,
        scale_qk=family != "t5",
    )
    probabilities = g.unary(
        key + ".softmax", key, "softmax", scores, axis=-1, formula="out = softmax(x, -1)"
    )
    weighted = g.op(
        key + ".weighted",
        key,
        "matmul",
        {"p": probabilities, "v": values["v"]},
        shape("B", heads, query_axis, hidden // heads),
        formula="out = p @ v",
    )
    merged = g.op(
        key + ".merge", key, "merge_heads", {"x": weighted}, shape("B", query_axis, hidden)
    )
    output = g.linear(
        key + ".output", key, merged, module + (".to_out.0" if family == "ltx" else ".o")
    )
    return g.end(key, output)


def text_encoder(
    g: VideoGraph, parent: str, prompt: Value, profile: VideoProfile
) -> tuple[Value, Value]:
    key = parent + ".text_encoder"
    ports = g.begin(
        key,
        parent,
        "T5 text encoder" if profile.text == "t5" else "UMT5 text encoder",
        {"prompt": prompt},
    )
    tokens = g.op(
        key + ".tokenize",
        key,
        "local_tokenization",
        {"text": ports["prompt"]},
        shape("B", "S"),
        kind="context",
        available=g.inputs.bindings.tokenizer_available,
        max_length=512,
        output_type="integer token IDs",
    )
    mask = g.op(
        key + ".mask", key, "padding_mask", {"ids": tokens}, shape("B", "S"), padding_token_id=0
    )
    x = g.op(
        key + ".embed",
        key,
        "embedding_lookup",
        {"ids": tokens},
        shape("B", "S", 4096),
        weights=("video.text_encoder.shared.weight",),
        formula="out = weight[ids]",
    )
    bias: Value | None = None
    if profile.text == "t5":
        bias = g.op(
            key + ".bias",
            key,
            "relative_position_buckets",
            {"ids": tokens},
            shape(1, 64, "S", "S"),
            weights=(
                "video.text_encoder.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            ),
            bidirectional=True,
            buckets=32,
            max_distance=128,
        )
    blocks = []
    for index in range(24):
        block = f"{key}.block.{index}"
        module = f"video.text_encoder.encoder.block.{index}"
        ins = {"x": x, "mask": mask}
        if bias is not None:
            ins["bias"] = bias
        p = g.begin(
            block, key, f"Text block {index}", ins, layer_index=index, semantic_role="layer"
        )
        local_bias = p.get("bias")
        if local_bias is None:
            local_bias = g.op(
                block + ".bias",
                block,
                "relative_position_buckets",
                {"x": p["x"]},
                shape(1, 64, "S", "S"),
                weights=(module + ".layer.0.SelfAttention.relative_attention_bias.weight",),
                bidirectional=True,
                buckets=32,
                max_distance=128,
            )
        norm = g.unary(
            block + ".norm1",
            block,
            "rms_norm",
            p["x"],
            weights=(module + ".layer.0.layer_norm.weight",),
            eps=1e-6,
        )
        attended = attention(
            g,
            block + ".attention",
            block,
            norm,
            norm,
            module + ".layer.0.SelfAttention",
            heads=64,
            family="t5",
            bias=local_bias,
            mask=p["mask"],
        )
        residual = g.residual(block + ".residual1", block, p["x"], attended)
        mlp = block + ".mlp"
        m = g.begin(mlp, block, "Gated GELU MLP", {"x": residual}, semantic_role="mlp")
        norm = g.unary(
            mlp + ".norm",
            mlp,
            "rms_norm",
            m["x"],
            weights=(module + ".layer.1.layer_norm.weight",),
            eps=1e-6,
        )
        gate = g.linear(mlp + ".gate", mlp, norm, module + ".layer.1.DenseReluDense.wi_0")
        gate = g.unary(mlp + ".gelu", mlp, "gelu_tanh", gate)
        up = g.linear(mlp + ".up", mlp, norm, module + ".layer.1.DenseReluDense.wi_1")
        activated = g.op(
            mlp + ".multiply",
            mlp,
            "multiply",
            {"gate": gate, "up": up},
            up.dims,
            formula="out = gate * up",
        )
        down = g.linear(mlp + ".down", mlp, activated, module + ".layer.1.DenseReluDense.wo")
        output = g.end(mlp, down)
        x = g.end(block, g.residual(block + ".residual2", block, residual, output))
        blocks.append(block)
    g.repetition(key + ".blocks", key, blocks, profile.text)
    x = g.unary(
        key + ".final_norm",
        key,
        "rms_norm",
        x,
        weights=("video.text_encoder.encoder.final_layer_norm.weight",),
        eps=1e-6,
    )
    if profile.text == "umt5":
        x = g.op(
            key + ".trim",
            key,
            "trim_text_padding",
            {"x": x, "mask": mask},
            shape("B", "L", 4096),
            formula="out[b] = x[b, :sum(mask[b] > 0)]",
            batching="variable valid-token length per example",
        )
    return g.end(key, x, extra_outputs={"mask": mask}), Value(key, "mask", mask.dims)


def denoiser(
    g: VideoGraph,
    parent: str,
    role: str,
    latent: Value,
    context: Value,
    time: Value,
    profile: VideoProfile,
    text_mask: Value,
) -> Value:
    key, module = parent + "." + role, "video." + role
    ports = g.begin(
        key,
        parent,
        role.replace("_", " ") + " denoiser",
        {"latent": latent, "context": context, "time": time, "text_mask": text_mask},
        selection="t >= boundary"
        if role == "high_noise"
        else "t < boundary"
        if role == "low_noise"
        else "single denoiser",
    )
    ltx = profile.family == "ltx"
    if ltx:
        tokens = g.op(
            key + ".patchify",
            key,
            "patchify_latents",
            {"x": ports["latent"]},
            shape("B", "N", 128),
            patch_size=1,
        )
        x = g.linear(key + ".projection", key, tokens, module + ".patchify_proj")
    else:
        x = g.op(
            key + ".patchify",
            key,
            "conv3d_patch_embedding",
            {"x": ports["latent"]},
            shape("B", "N", profile.hidden),
            weights=(module + ".patch_embedding.weight", module + ".patch_embedding.bias"),
            kernel=[1, 2, 2],
            stride=[1, 2, 2],
            output_layout="flatten spatial-temporal axes to B,N,D",
        )
    positions = g.op(
        key + ".positions",
        key,
        "latent_grid_positions",
        {"latent": ports["latent"]},
        shape("B", "N", 3),
        axes="time,height,width",
        causal_temporal_positioning=ltx and profile.release == "0.9.6",
    )
    timestep = g.op(
        key + ".time_embedding",
        key,
        "sinusoidal_embedding",
        {"time": ports["time"]},
        shape("B", "N", 256),
        multiplier=1000 if ltx else 1,
        note="broadcast timestep to tokens; conditioned tokens may have their own timestep",
    )
    time_prefix = module + (".adaln_single.emb.timestep_embedder" if ltx else ".time_embedding")
    timestep = g.linear(
        key + ".time_linear1", key, timestep, time_prefix + (".linear_1" if ltx else ".0")
    )
    timestep = g.unary(key + ".time_silu", key, "silu", timestep)
    timestep = g.linear(
        key + ".time_linear2", key, timestep, time_prefix + (".linear_2" if ltx else ".2")
    )
    activated = g.unary(key + ".time_silu2", key, "silu", timestep)
    modulation = g.linear(
        key + ".time_projection",
        key,
        activated,
        module + (".adaln_single.linear" if ltx else ".time_projection.1"),
    )
    ctx_prefix = module + (".caption_projection" if ltx else ".text_embedding")
    ctx = ports["context"]
    if not ltx:
        ctx = g.op(
            key + ".text_padding",
            key,
            "zero_pad_text_context",
            {"context": ctx},
            shape("B", 512, 4096),
            length=512,
            formula="out[b] = concat(context[b], zero_rows_to_length_512, axis=0)",
            attention_mask="none after projection; all 512 projected rows are attended",
        )
    ctx = g.linear(key + ".text_linear1", key, ctx, ctx_prefix + (".linear_1" if ltx else ".0"))
    ctx = g.unary(key + ".text_gelu", key, "gelu_tanh", ctx)
    ctx = g.linear(key + ".text_linear2", key, ctx, ctx_prefix + (".linear_2" if ltx else ".2"))
    blocks = []
    for index in range(profile.layers):
        block = f"{key}.block.{index}"
        source = f"{module}.{'transformer_blocks' if ltx else 'blocks'}.{index}"
        p = g.begin(
            block,
            key,
            f"Block {index}",
            {
                "x": x,
                "context": ctx,
                "modulation": modulation,
                "positions": positions,
                "text_mask": ports["text_mask"],
            },
            layer_index=index,
            semantic_role="layer",
        )
        mod = g.op(
            block + ".modulation",
            block,
            "adaptive_modulation",
            {"time": p["modulation"]},
            shape("B", "N", 6, profile.hidden),
            weights=(source + (".scale_shift_table" if ltx else ".modulation"),),
            formula="out = reshape(time, B,N,6,D) + table",
            fields="shift_attention,scale_attention,gate_attention,shift_mlp,scale_mlp,gate_mlp",
        )
        norm = g.unary(
            block + ".norm1",
            block,
            "rms_norm" if ltx else "layer_norm",
            p["x"],
            eps=1e-6,
            affine=False,
        )
        norm = g.op(
            block + ".modulate_attention",
            block,
            "scale_shift",
            {"x": norm, "modulation": mod},
            norm.dims,
            formula="out = x * (1 + modulation[:,:,1,:]) + modulation[:,:,0,:]",
        )
        attn = attention(
            g,
            block + ".self_attention",
            block,
            norm,
            norm,
            source + (".attn1" if ltx else ".self_attn"),
            heads=profile.heads,
            family=profile.family,
            positions=p["positions"],
        )
        gated = g.op(
            block + ".gate_attention",
            block,
            "gate",
            {"x": attn, "modulation": mod},
            attn.dims,
            formula="out = x * modulation[:,:,2,:]",
        )
        x = g.residual(block + ".residual1", block, p["x"], gated)
        cross_input = (
            x
            if ltx
            else g.unary(
                block + ".norm3",
                block,
                "layer_norm",
                x,
                weights=(source + ".norm3.weight", source + ".norm3.bias"),
                eps=1e-6,
            )
        )
        cross = attention(
            g,
            block + ".cross_attention",
            block,
            cross_input,
            p["context"],
            source + (".attn2" if ltx else ".cross_attn"),
            heads=profile.heads,
            family=profile.family,
            cross=True,
            mask=p["text_mask"] if ltx else None,
        )
        x = g.residual(block + ".residual2", block, x, cross)
        mlp = block + ".mlp"
        m = g.begin(mlp, block, "GELU MLP", {"x": x, "modulation": mod}, semantic_role="mlp")
        norm = g.unary(
            mlp + ".norm", mlp, "rms_norm" if ltx else "layer_norm", m["x"], eps=1e-6, affine=False
        )
        norm = g.op(
            mlp + ".modulate",
            mlp,
            "scale_shift",
            {"x": norm, "modulation": m["modulation"]},
            norm.dims,
            formula="out = x * (1 + modulation[:,:,4,:]) + modulation[:,:,3,:]",
        )
        up = g.linear(mlp + ".up", mlp, norm, source + (".ff.net.0.proj" if ltx else ".ffn.0"))
        up = g.unary(mlp + ".gelu", mlp, "gelu_tanh", up)
        down = g.linear(mlp + ".down", mlp, up, source + (".ff.net.2" if ltx else ".ffn.2"))
        down = g.op(
            mlp + ".gate",
            mlp,
            "gate",
            {"x": down, "modulation": m["modulation"]},
            down.dims,
            formula="out = x * modulation[:,:,5,:]",
        )
        x = g.end(block, g.residual(block + ".residual3", block, x, g.end(mlp, down)))
        blocks.append(block)
    g.repetition(key + ".blocks", key, blocks, profile.denoiser)
    x = g.unary(key + ".head_norm", key, "layer_norm", x, affine=False, eps=1e-6)
    x = g.op(
        key + ".head_modulation",
        key,
        "scale_shift",
        {"x": x, "time": timestep},
        x.dims,
        weights=(module + (".scale_shift_table" if ltx else ".head.modulation"),),
        formula="(shift, scale) = table + time; out = x * (1 + scale) + shift",
    )
    x = g.linear(key + ".head", key, x, module + (".proj_out" if ltx else ".head.head"))
    x = g.op(
        key + ".unpatchify",
        key,
        "unpatchify_latents",
        {"x": x, "grid": positions},
        shape("B", profile.channels, "T", "H", "W"),
        patch=[1, 1, 1] if ltx else [1, 2, 2],
        layout="B,C,T,H,W",
    )
    return g.end(key, x)
