"""The selected causal video VAEs, with concrete learned modules and skip paths."""

from ..video_profiles import VideoProfile, reference
from . import records as r
from .dense import shape
from .video_graph import Value, VideoGraph


def dimension(
    d: r.ArchitectureDimension, factor: int, *, up: bool, temporal: bool
) -> r.ArchitectureDimension:
    if factor == 1:
        return d
    if d.kind == "constant":
        value = (
            factor * (d.value - 1) + 1
            if up and temporal
            else factor * d.value
            if up
            else (d.value - 1) // factor + 1
            if temporal
            else d.value // factor
        )
        return r.ArchitectureConstantDimension(kind="constant", value=value)
    text = d.name if d.kind == "symbol" else d.text if d.kind == "expression" else "unknown"
    symbols = [d.name] if d.kind == "symbol" else d.symbols if d.kind == "expression" else []
    expression = (
        f"{factor}*({text}-1)+1"
        if up and temporal
        else f"{factor}*({text})"
        if up
        else f"floor(({text}-1)/{factor})+1"
        if temporal
        else f"({text})/{factor}"
    )
    return r.ArchitectureExpressionDimension(kind="expression", text=expression, symbols=symbols)


def resized(
    x: Value, channels: int, factors: tuple[int, int, int] = (1, 1, 1), *, up: bool = False
) -> list[r.ArchitectureDimension]:
    assert x.dims is not None and len(x.dims) == 5
    return [
        x.dims[0],
        *shape(channels),
        *(
            dimension(d, f, up=up, temporal=i == 0)
            for i, (d, f) in enumerate(zip(x.dims[2:], factors, strict=True))
        ),
    ]


def conv(
    g: VideoGraph,
    key: str,
    parent: str,
    x: Value,
    module: str,
    *,
    causal: bool = True,
    stride: tuple[int, int, int] = (1, 1, 1),
) -> Value:
    physical = g.inputs.bindings.physical
    weight = physical[module + ".weight"]
    names = (module + ".weight",) + ((module + ".bias",) if module + ".bias" in physical else ())
    return g.op(
        key,
        parent,
        "conv3d" if len(weight.shape) == 5 else "conv2d_framewise",
        {"x": x},
        resized(x, weight.shape[0], stride),
        weights=names,
        kernel=weight.shape[2:],
        stride=list(stride),
        padding="right/bottom zero padding before spatial stride two"
        if stride[1] == 2
        else "causal temporal padding"
        if causal
        else "symmetric padding",
        causal=causal,
        formula="out = convolution(x, weight) + bias",
        layout="B,C,T,H,W",
    )


def channel_norm(
    g: VideoGraph,
    key: str,
    parent: str,
    x: Value,
    *,
    weights: tuple[str, ...] = (),
    ltx: bool = False,
) -> Value:
    assert x.dims is not None and x.dims[1].kind == "constant"
    return g.unary(
        key,
        parent,
        "pixel_norm" if ltx else "rms_norm_channels",
        x,
        weights=weights,
        axis=1,
        channels=x.dims[1].value,
        eps=1e-8 if ltx else 1e-12,
        formula="out = x / sqrt(mean(x^2, axis=1) + eps)"
        if ltx
        else "out = x / max(l2_norm(x, axis=1), eps) * sqrt(channels) * gamma",
    )


def resblock(
    g: VideoGraph,
    key: str,
    parent: str,
    x: Value,
    module: str,
    *,
    ltx: bool = False,
    time: Value | None = None,
    causal: bool = True,
) -> Value:
    inputs = {"x": x}
    if time is not None:
        inputs["time"] = time
    p = g.begin(key, parent, "Residual convolution block", inputs)
    y = p["x"]
    for index in range(2):
        weights = () if ltx else (module + f".residual.{0 if index == 0 else 3}.gamma",)
        y = channel_norm(
            g,
            key + f".norm{index}",
            key,
            y,
            weights=weights,
            ltx=ltx,
        )
        if time is not None:
            y = g.op(
                key + f".modulate{index}",
                key,
                "adaptive_scale_shift",
                {"x": y, "time": p["time"]},
                y.dims,
                weights=(module + ".scale_shift_table",),
                pair=index,
                formula=(
                    "modulation = table + reshape(time, B,4,C); "
                    "(shift,scale) = modulation[:,pair*2:pair*2+2]; out = x*(1+scale)+shift"
                ),
            )
        y = g.unary(key + f".silu{index}", key, "silu", y)
        y = conv(
            g,
            key + f".conv{index}",
            key,
            y,
            module + (f".conv{index + 1}.conv" if ltx else f".residual.{2 if index == 0 else 6}"),
            causal=causal,
        )
    shortcut = p["x"]
    for suffix in (".shortcut", ".conv_shortcut.conv"):
        if module + suffix + ".weight" in g.inputs.bindings.physical:
            shortcut = conv(g, key + ".shortcut", key, shortcut, module + suffix, causal=causal)
    return g.end(key, g.residual(key + ".add", key, shortcut, y))


def spatial_attention(g: VideoGraph, key: str, parent: str, x: Value, module: str) -> Value:
    p = g.begin(key, parent, "Frame-local spatial attention", {"x": x}, semantic_role="attention")
    norm = channel_norm(g, key + ".norm", key, p["x"], weights=(module + ".norm.gamma",))
    qkv = conv(g, key + ".qkv", key, norm, module + ".to_qkv")
    channels = g.inputs.bindings.physical[module + ".proj.weight"].shape[0]
    scores = g.op(
        key + ".scores",
        key,
        "spatial_attention_scores",
        {"qkv": qkv},
        None,
        formula="out = Q @ K^T / sqrt(channels)",
        channels=channels,
        value_layout="B*T,H*W,H*W; Q,K,V split from qkv",
    )
    probabilities = g.unary(key + ".softmax", key, "softmax", scores, axis=-1)
    weighted = g.op(
        key + ".weighted",
        key,
        "spatial_attention_values",
        {"p": probabilities, "qkv": qkv},
        p["x"].dims,
        formula="out = reshape(p @ V, B,C,T,H,W)",
    )
    projected = conv(g, key + ".projection", key, weighted, module + ".proj")
    return g.end(key, g.residual(key + ".residual", key, p["x"], projected))


def wan_resample(g: VideoGraph, key: str, parent: str, x: Value, module: str, *, up: bool) -> Value:
    p = g.begin(key, parent, "Upsample" if up else "Downsample", {"x": x})
    y = p["x"]
    temporal = module + ".time_conv.weight" in g.inputs.bindings.physical
    if up and temporal:
        channels = g.inputs.bindings.physical[module + ".time_conv.weight"].shape[0] // 2
        y = conv(g, key + ".time_conv", key, y, module + ".time_conv")
        y = g.op(
            key + ".interleave",
            key,
            "temporal_channel_interleave",
            {"x": y},
            resized(p["x"], channels, (2, 1, 1), up=True),
            first_frame="retain one initial frame",
        )
    channels = g.inputs.bindings.physical[module + ".resample.1.weight"].shape[0]
    if up:
        y = g.op(
            key + ".nearest",
            key,
            "nearest_exact_upsample",
            {"x": y},
            resized(
                y,
                g.inputs.bindings.physical[module + ".resample.1.weight"].shape[1],
                (1, 2, 2),
                up=True,
            ),
        )
    y = conv(
        g,
        key + ".spatial_conv",
        key,
        y,
        module + ".resample.1",
        stride=(1, 1, 1) if up else (1, 2, 2),
    )
    if not up:
        if temporal:
            y = conv(g, key + ".time_conv", key, y, module + ".time_conv", stride=(2, 1, 1))
    return g.end(key, y)


def wan_vae(g: VideoGraph, parent: str, x: Value, profile: VideoProfile, *, encode: bool) -> Value:
    direction = "encoder" if encode else "decoder"
    key, module = parent + "." + direction, "video.vae." + direction
    p = g.begin(key, parent, f"Wan {profile.vae[-3:]} VAE {direction}", {"x": x})
    y = p["x"]
    statistics = reference()["wan_latent_statistics"][profile.vae[-3:]]
    if not encode:
        y = g.unary(
            key + ".unscale",
            key,
            "latent_denormalization",
            y,
            formula="out = x * std + mean",
            **statistics,
        )
        y = conv(g, key + ".post_quant", key, y, "video.vae.conv2")
    elif profile.channels == 48:
        y = g.op(
            key + ".patchify",
            key,
            "space_to_depth",
            {"x": y},
            resized(y, 12, (1, 2, 2)),
            patch_size=2,
        )
    y = conv(g, key + ".conv_in", key, y, module + ".conv1")

    def middle(value: Value) -> Value:
        value = resblock(g, key + ".middle.0", key, value, module + ".middle.0")
        value = spatial_attention(g, key + ".middle.1", key, value, module + ".middle.1")
        return resblock(g, key + ".middle.2", key, value, module + ".middle.2")

    if not encode:
        y = middle(y)
    stack = "downsamples" if encode else "upsamples"
    if profile.channels == 48:
        for index in range(4):
            stage = f"{key}.stage.{index}"
            source = f"{module}.{stack}.{index}.{stack}"
            pstage = g.begin(
                stage, key, f"{'Compression' if encode else 'Expansion'} stage {index}", {"x": y}
            )
            y = pstage["x"]
            for block in range(2 if encode else 3):
                y = resblock(g, f"{stage}.residual.{block}", stage, y, f"{source}.{block}")
            if index < 3:
                y = wan_resample(
                    g,
                    stage + ".resample",
                    stage,
                    y,
                    source + (".2" if encode else ".3"),
                    up=not encode,
                )
            if encode or index < 3:
                skip = g.op(
                    stage + ".skip",
                    stage,
                    "average_space_to_depth" if encode else "duplicate_depth_to_space",
                    {"x": pstage["x"]},
                    y.dims,
                    temporal_factor=(2 if index in ((1, 2) if encode else (0, 1)) else 1),
                    spatial_factor=2 if index < 3 else 1,
                    first_frame="causal initial frame retained",
                    channel_groups="mean" if encode else "repeat",
                )
                y = g.residual(stage + ".add", stage, skip, y)
            y = g.end(stage, y)
    else:
        count, resamples = (11, {2, 5, 8}) if encode else (15, {3, 7, 11})
        for index in range(count):
            name = f"{module}.{stack}.{index}"
            node = f"{key}.stage.{index}"
            y = (
                wan_resample(g, node, key, y, name, up=not encode)
                if index in resamples
                else resblock(g, node, key, y, name)
            )
    if encode:
        y = middle(y)
    y = channel_norm(g, key + ".head_norm", key, y, weights=(module + ".head.0.gamma",))
    y = g.unary(key + ".head_silu", key, "silu", y)
    y = conv(g, key + ".head", key, y, module + ".head.2")
    if encode:
        y = conv(g, key + ".quant", key, y, "video.vae.conv1")
        y = g.op(
            key + ".mean",
            key,
            "posterior_mean",
            {"moments": y},
            resized(y, profile.channels),
            formula="out = moments[:, :latent_channels]",
            latent_channels=profile.channels,
            log_variance="not sampled in conditioning encoder",
        )
        y = g.unary(
            key + ".normalize",
            key,
            "latent_normalization",
            y,
            formula="out = (x - mean) / std",
            **statistics,
        )
    elif profile.channels == 48:
        y = g.op(
            key + ".unpatchify",
            key,
            "depth_to_space",
            {"x": y},
            resized(y, 3, (1, 2, 2), up=True),
            patch_size=2,
        )
    return g.end(key, y)


def ltx_vae(
    g: VideoGraph, parent: str, x: Value, *, encode: bool, time: Value | None = None
) -> Value:
    direction = "encoder" if encode else "decoder"
    key, module = parent + "." + direction, "video.vae." + direction
    inputs = {"x": x}
    if time is not None:
        inputs["time"] = time
    p = g.begin(key, parent, f"LTX causal VAE {direction}", inputs, causal=encode)
    y = p["x"]
    statistics = (
        "video.vae.per_channel_statistics.mean-of-means",
        "video.vae.per_channel_statistics.std-of-means",
    )
    if not encode:
        y = g.unary(
            key + ".unscale",
            key,
            "latent_denormalization",
            y,
            weights=statistics,
            formula="out = x * std_of_means + mean_of_means",
            broadcast="stored per-channel statistics over B,T,H,W",
        )
    if encode:
        y = g.op(
            key + ".patchify",
            key,
            "space_to_depth",
            {"x": y},
            resized(y, 48, (1, 4, 4)),
            patch_size=4,
        )
    y = conv(g, key + ".conv_in", key, y, module + ".conv_in.conv", causal=encode)
    counts = [4, 0, 6, 0, 6, 0, 2, 0, 2] if encode else [5, 0, 5, 0, 5, 0, 5]
    for index, count in enumerate(counts):
        stage = f"{key}.stage.{index}"
        source = f"{module}.{'down_blocks' if encode else 'up_blocks'}.{index}"
        external = {"x": y}
        if not encode:
            external["time"] = p["time"]
        ports = g.begin(
            stage, key, f"{'Compression' if encode else 'Expansion'} stage {index}", external
        )
        y = ports["x"]
        if count:
            embedded: Value | None = None
            if not encode:
                embedded = g.op(
                    stage + ".sinusoidal",
                    stage,
                    "sinusoidal_embedding",
                    {"time": ports["time"]},
                    shape("B", 256),
                    weights=(module + ".timestep_scale_multiplier",),
                )
                embedded = g.linear(
                    stage + ".time_linear1",
                    stage,
                    embedded,
                    source + ".time_embedder.timestep_embedder.linear_1",
                )
                embedded = g.unary(stage + ".time_silu", stage, "silu", embedded)
                embedded = g.linear(
                    stage + ".time_linear2",
                    stage,
                    embedded,
                    source + ".time_embedder.timestep_embedder.linear_2",
                )
            for block in range(count):
                y = resblock(
                    g,
                    f"{stage}.residual.{block}",
                    stage,
                    y,
                    f"{source}.res_blocks.{block}",
                    ltx=True,
                    time=embedded,
                    causal=encode,
                )
        else:
            factors = (
                (1, 2, 2)
                if encode and index == 1
                else (2, 1, 1)
                if encode and index == 3
                else (2, 2, 2)
            )
            convolved = conv(g, stage + ".conv", stage, y, source + ".conv.conv", causal=encode)
            weight = g.inputs.bindings.physical[source + ".conv.conv.weight"]
            channels = (
                weight.shape[0] * factors[0] * factors[1] * factors[2]
                if encode
                else weight.shape[0] // 8
            )
            out_dims = resized(y, channels, factors, up=not encode)
            branch = g.op(
                stage + ".shuffle",
                stage,
                "space_to_depth" if encode else "pixel_shuffle3d",
                {"x": convolved},
                out_dims,
                factors=list(factors),
                first_frame="prepend then rearrange"
                if encode
                else "remove first expanded temporal frame",
            )
            skip = g.op(
                stage + ".skip",
                stage,
                "space_to_depth_mean" if encode else "pixel_shuffle_repeat",
                {"x": y},
                out_dims,
                factors=list(factors),
            )
            y = g.residual(stage + ".add", stage, skip, branch)
        y = g.end(stage, y)
    y = channel_norm(g, key + ".norm_out", key, y, ltx=True)
    if not encode:
        embedded = g.op(
            key + ".time_sinusoidal",
            key,
            "sinusoidal_embedding",
            {"time": p["time"]},
            shape("B", 256),
            weights=(module + ".timestep_scale_multiplier",),
        )
        embedded = g.linear(
            key + ".time_linear1",
            key,
            embedded,
            module + ".last_time_embedder.timestep_embedder.linear_1",
        )
        embedded = g.unary(key + ".time_silu", key, "silu", embedded)
        embedded = g.linear(
            key + ".time_linear2",
            key,
            embedded,
            module + ".last_time_embedder.timestep_embedder.linear_2",
        )
        y = g.op(
            key + ".modulation",
            key,
            "adaptive_scale_shift",
            {"x": y, "time": embedded},
            y.dims,
            weights=(module + ".last_scale_shift_table",),
            formula="(shift,scale) = table + reshape(time, B,2,C); out = x*(1+scale)+shift",
        )
    y = g.unary(key + ".silu_out", key, "silu", y)
    y = conv(g, key + ".conv_out", key, y, module + ".conv_out.conv", causal=encode)
    y = (
        g.op(
            key + ".posterior",
            key,
            "posterior_sample",
            {"moments": y},
            resized(y, 128),
            latent_channels=128,
            log_variance="last encoder channel broadcast across the 128 latent channels",
            formula=(
                "out = moments[:,:128] + "
                "exp(0.5 * clamp(moments[:,128:129], -30, 20)) * normal_noise"
            ),
            executed=False,
        )
        if encode
        else g.op(
            key + ".unpatchify",
            key,
            "depth_to_space",
            {"x": y},
            resized(y, 3, (1, 4, 4), up=True),
            patch_size=4,
        )
    )
    if encode:
        y = g.unary(
            key + ".normalize",
            key,
            "latent_normalization",
            y,
            weights=statistics,
            formula="out = (x - mean_of_means) / std_of_means",
            broadcast="stored per-channel statistics over B,T,H,W",
        )
    return g.end(key, y)
