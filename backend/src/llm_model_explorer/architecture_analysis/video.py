"""Native static base pipelines for the selected LTX and Wan packages."""

from ..model_files import ModelError
from ..video_profiles import LTX_SOURCE, WAN_SOURCE, checked
from . import records as r
from .core import AnalysisInput, Description, DescriptionRegistry, GraphBuilder, Producer
from .dense import shape
from .video_graph import Value, VideoGraph
from .video_transformers import denoiser, text_encoder
from .video_vae import ltx_vae, wan_vae


def build(inputs: AnalysisInput, builder: GraphBuilder) -> None:
    profile = checked(inputs.configuration, inputs.bindings.physical)
    g = VideoGraph(inputs, builder)
    for name, meaning in {
        "B": "model batch, including conditional/unconditional copies where guidance applies",
        "S": "text token sequence length, at most 512",
        "L": "valid UMT5 token length per example after removing padding, at most 512",
        "N": "flattened latent patch sequence",
        "T": "latent temporal length",
        "H": "latent height",
        "W": "latent width",
        "Y": "input image height",
        "X": "input image width",
    }.items():
        builder.add_symbol(name, meaning)
    boundary: dict[str, r.ArchitectureShape] = {"prompt": None, "configuration": None}
    has_image = "image_to_video" in profile.modalities
    if has_image:
        boundary["image"] = shape("B", 3, 1, "Y", "X")
    root = "video"
    p = g.begin(
        root,
        None,
        f"{profile.family.upper()} {profile.variant} · base pipeline",
        boundary,
        modalities=list(profile.modalities),
        image="optional in image_to_video mode"
        if len(profile.modalities) == 2
        else "required"
        if has_image
        else "absent",
        execution="static description; no generation performed",
        release=profile.release,
    )
    prompt = p["prompt"]
    if profile.family == "wan":
        prompt = g.op(
            root + ".guidance_prompts",
            root,
            "guidance_conditioning_batch",
            {"prompt": prompt, "configuration": p["configuration"]},
            None,
            label="Conditional and negative text",
            source="selected Wan classifier-free guidance configuration",
        )
    context, text_mask = text_encoder(g, root, prompt, profile)
    latent_dims = shape("B", profile.channels, "T", "H", "W")
    noise = g.op(
        root + ".noise",
        root,
        "initial_latent_noise",
        {"configuration": p["configuration"]},
        latent_dims,
        distribution="normal",
        executed=False,
    )
    encoded: Value | None = None
    if has_image:
        condition_key = root + ".image_conditioning"
        c = g.begin(
            condition_key,
            root,
            "Initial image conditioning",
            {"image": p["image"], "configuration": p["configuration"]},
            condition="image_to_video mode",
            required=len(profile.modalities) == 1,
        )
        image = g.unary(
            condition_key + ".normalize",
            condition_key,
            "image_normalization",
            c["image"],
            formula="out = 2*x - 1",
            input_range="[0,1]",
            output_range="[-1,1]",
        )
        if profile.variant == "i2v-a14b":
            image = g.op(
                condition_key + ".reference_clip",
                condition_key,
                "initial_frame_zero_extension",
                {"image": image, "configuration": c["configuration"]},
                [
                    *shape("B", 3),
                    r.ArchitectureExpressionDimension(
                        kind="expression", text="4*(T-1)+1", symbols=["T"]
                    ),
                    *shape("Y", "X"),
                ],
                formula="out = concat(image, zeros_for_remaining_frames, axis=time)",
                convention=(
                    "encode the initial image plus zero frames before concatenating the latent mask"
                ),
            )
        image = (
            ltx_vae(g, condition_key, image, encode=True)
            if profile.family == "ltx"
            else wan_vae(g, condition_key, image, profile, encode=True)
        )
        image = g.op(
            condition_key + ".pack",
            condition_key,
            "image_conditioning_latents",
            {"image_latents": image, "configuration": c["configuration"]},
            shape("B", 20 if profile.variant == "i2v-a14b" else profile.channels, "T", "H", "W"),
            convention="4-channel temporal mask plus 16-channel encoded reference"
            if profile.variant == "i2v-a14b"
            else "clean initial-frame latent and conditioning mask",
            absent_image="branch inactive in text_to_video mode",
        )
        encoded = g.end(condition_key, image)
    initial = noise
    if encoded is not None and profile.variant != "i2v-a14b":
        initial = g.op(
            root + ".initialize",
            root,
            "condition_initial_latents",
            {"noise": noise, "image": encoded, "configuration": p["configuration"]},
            latent_dims,
            condition="only in image_to_video mode",
            convention="preserve conditioned first-frame latent; noise elsewhere",
        )
    loop = root + ".iteration"
    state = g.op(
        root + ".latent_state",
        root,
        "latent_state",
        {"initial": initial, "next": Value(loop, "out", latent_dims)},
        latent_dims,
        kind="state",
        initialization="use initial once; next on subsequent steps",
    )
    step = g.op(
        root + ".step_state",
        root,
        "iteration_index",
        {"next": Value(root + ".increment", "out", shape())},
        shape(),
        kind="state",
        initial=0,
    )
    schedule = g.op(
        root + ".schedule",
        root,
        "denoising_schedule",
        {"index": step, "configuration": p["configuration"]},
        shape("B", "N"),
        training_timesteps=1000,
        timestep_units="normalized rectified-flow t in [0,1]"
        if profile.family == "ltx"
        else "training timestep in [0,1000]",
        sampler="checkpoint distilled schedule"
        if profile.family == "ltx"
        else "UniPC / DPM++ as selected by reference configuration",
        boundary=profile.boundary,
        stochastic_sampling=profile.release == "0.9.6" if profile.family == "ltx" else False,
        decode_condition="index reaches configured schedule length",
    )
    history = g.op(
        root + ".solver_history",
        root,
        "symbolic_solver_history",
        {"next": Value(loop, "next_history", None)},
        None,
        kind="state",
        convention="prior/next solver state; no captured numeric history",
    )
    loop_inputs = {
        "latent": state,
        "text": context,
        "text_mask": text_mask,
        "time": schedule,
        "history": history,
        "configuration": p["configuration"],
    }
    if encoded is not None:
        loop_inputs["image"] = encoded
    lp = g.begin(
        loop,
        root,
        "One denoising iteration",
        loop_inputs,
        repetition="until schedule complete; same weights and state",
        executed=False,
    )
    latent = lp["latent"]
    if profile.variant == "i2v-a14b":
        latent = g.op(
            loop + ".concat",
            loop,
            "concat",
            {"latent": latent, "image": lp["image"]},
            shape("B", 36, "T", "H", "W"),
            axis=1,
            formula="out = concat(latent, image, axis=1)",
        )
    if profile.boundary is None:
        timestep = lp["time"]
        if "image" in lp:
            timestep = g.op(
                loop + ".conditioned_time",
                loop,
                "conditioned_token_timesteps",
                {"time": timestep, "image": lp["image"], "configuration": lp["configuration"]},
                timestep.dims,
                convention="zero on conditioned image tokens, current timestep on noisy tokens",
            )
        velocity = denoiser(
            g,
            loop,
            "denoiser",
            latent,
            lp["text"],
            timestep,
            profile,
            lp["text_mask"],
            configuration=lp["configuration"],
        )
    else:
        high = denoiser(
            g,
            loop,
            "high_noise",
            latent,
            lp["text"],
            lp["time"],
            profile,
            lp["text_mask"],
            configuration=lp["configuration"],
        )
        low = denoiser(
            g,
            loop,
            "low_noise",
            latent,
            lp["text"],
            lp["time"],
            profile,
            lp["text_mask"],
            configuration=lp["configuration"],
        )
        velocity = g.op(
            loop + ".expert_selection",
            loop,
            "timestep_expert_selection",
            {"time": lp["time"], "high_noise": high, "low_noise": low},
            latent_dims,
            formula="out = high_noise if time >= boundary*num_train_timesteps else low_noise",
            boundary=profile.boundary,
            num_train_timesteps=1000,
            condition="only the selected expert is evaluated",
            label="Noise-level expert choice",
        )
    if profile.family == "wan":
        velocity = g.op(
            loop + ".cfg",
            loop,
            "classifier_free_guidance",
            {"predictions": velocity, "time": lp["time"], "configuration": lp["configuration"]},
            latent_dims,
            formula="out = unconditional + scale*(conditional-unconditional)",
            scale_low=3.0
            if profile.variant == "t2v-a14b"
            else 3.5
            if profile.variant == "i2v-a14b"
            else 5.0,
            scale_high=4.0
            if profile.variant == "t2v-a14b"
            else 3.5
            if profile.variant == "i2v-a14b"
            else 5.0,
        )
    step_inputs = {
        "latent": lp["latent"],
        "velocity": velocity,
        "time": lp["time"],
        "history": lp["history"],
        "configuration": lp["configuration"],
    }
    if "image" in lp:
        step_inputs["image"] = lp["image"]
    updated = g.op(
        loop + ".update",
        loop,
        "scheduler_step",
        step_inputs,
        latent_dims,
        formula="out = solver_step(latent, velocity, time, history, configuration)",
        conditioning="preserve conditioned cells where the profile mask applies",
        solver_history="updated with each symbolic step",
    )
    next_history = g.op(
        loop + ".history_update",
        loop,
        "solver_history_update",
        {"previous": lp["history"], "velocity": velocity, "time": lp["time"]},
        None,
        convention="retain the solver's prior evaluations required for the next step",
    )
    updated = g.end(loop, updated, extra_outputs={"next_history": next_history})
    increment = g.op(
        root + ".increment",
        root,
        "next_iteration",
        {"index": step, "next_latent": updated},
        shape(),
        formula="out = index + 1",
    )
    final = g.op(
        root + ".termination",
        root,
        "decode_when_schedule_complete",
        {"latent": updated, "index": increment, "configuration": p["configuration"]},
        latent_dims,
        condition="index >= schedule length; otherwise repeat",
        label="Termination / final latents",
    )
    decoder = root + ".vae"
    d = g.begin(
        decoder, root, "VAE decoding", {"latents": final, "configuration": p["configuration"]}
    )
    if profile.family == "ltx":
        decode_time = g.op(
            decoder + ".decode_time",
            decoder,
            "decode_timestep",
            {"configuration": d["configuration"]},
            shape("B"),
            default=0.05,
        )
        noised = g.op(
            decoder + ".decode_noise",
            decoder,
            "decode_noise_blend",
            {"latents": d["latents"], "configuration": d["configuration"]},
            d["latents"].dims,
            formula="out = latents*(1-decode_noise_scale) + normal_noise*decode_noise_scale",
            decode_noise_scale=0.025,
            executed=False,
        )
        video = ltx_vae(g, decoder, noised, encode=False, time=decode_time)
    else:
        video = wan_vae(g, decoder, d["latents"], profile, encode=False)
    video = g.unary(
        decoder + ".output_range",
        decoder,
        "video_output_range",
        video,
        formula="out = clamp((x+1)/2, 0, 1)",
        output="RGB video frames; no playback or saved output",
    )
    g.end(root, g.end(decoder, video), output_name="video")


def register_video(registry: DescriptionRegistry) -> None:
    for family, architecture, source in (
        ("ltx", "LTXVideoInspection", LTX_SOURCE),
        ("wan", "WanVideoInspection", WAN_SOURCE),
    ):

        def supports(inputs: AnalysisInput, selected: str = family) -> bool:
            try:
                return checked(inputs.configuration, inputs.bindings.physical).family == selected
            except (ModelError, ValueError, KeyError, TypeError):
                return False

        registry.register(
            Description(
                Producer(f"{family}-video-base", "2" if family == "ltx" else "1", source),
                "video_generation",
                frozenset({"video_inspection"}),
                frozenset({architecture}),
                supports,
                build,
            )
        )
