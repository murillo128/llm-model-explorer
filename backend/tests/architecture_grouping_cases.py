"""Export observed producer graphs for the independent issue-119 TypeScript oracle.

Inputs reuse independently authored metadata. Correspondence records the builder's
explicit semantic keys, never labels or shape guesses. No weights are read.
"""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from dense_fixtures import small_config, small_storage
from test_dense_architecture import metadata as dense_metadata
from test_dense_architecture import registry as dense_registry
from test_qwen35_architecture import metadata as hybrid_metadata
from test_qwen35_architecture import registry as hybrid_registry
from test_vjepa2_architecture import metadata as visual_metadata
from test_vjepa2_architecture import registry as visual_registry

from llm_model_explorer.architecture_analysis import (
    AnalysisInput,
    DescriptionRegistry,
    GraphBuilder,
)


def cases() -> list[tuple[str, AnalysisInput, DescriptionRegistry]]:
    result = []
    for qwen in (False, True):
        data = dense_metadata(small_config(qwen), small_storage(qwen), tokenizer=True)
        result.append(("qwen3" if qwen else "llama", data, dense_registry()))
    config = small_config(False) | {"attention_bias": True, "mlp_bias": True}
    result.append(
        ("llama-bias", dense_metadata(config, small_storage(False, biases=True)), dense_registry())
    )
    partial = dense_metadata(small_config(False), small_storage(False)[:-1])
    result.append(("llama-partial", partial, dense_registry()))
    result.append(("hybrid", hybrid_metadata(), hybrid_registry()))
    visual = visual_metadata()
    result.append(("visual", visual, visual_registry()))
    config = copy.deepcopy(dict(visual.configuration)) | {"qkv_bias": False}
    physical = {
        k: v
        for k, v in visual.bindings.physical.items()
        if not any(k.endswith(f"attention.{role}.bias") for role in ("query", "key", "value"))
    }
    visual = replace(
        visual, configuration=config, bindings=replace(visual.bindings, physical=physical)
    )
    result.append(("visual-no-qkv-bias", visual, visual_registry()))
    return result


def export(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    original = GraphBuilder.record_id
    for name, inputs, registry in cases():
        names: dict[str, dict[str, str]] = {"nodes": {}, "parameters": {}, "repetitions": {}}

        def record(
            builder: GraphBuilder,
            kind: str,
            key: str,
            names: dict[str, dict[str, str]] = names,
        ) -> str:
            identity = original(builder, kind, key)
            if kind + "s" in names:
                names[kind + "s"][identity] = key
            return identity

        if name == "hybrid":
            # This historical oracle covers the retained cached single-pass component
            # used by Kev. Native generation is checked separately by export_families.
            from llm_model_explorer.architecture_analysis.qwen35 import build_single_pass

            description = registry.select(inputs)
            assert description is not None
            registry = DescriptionRegistry()
            registry.register(replace(description, build=build_single_pass))
        with patch.object(GraphBuilder, "record_id", record):
            result = registry.analyze(inputs)
        assert result.graph is not None, result
        (directory / f"{name}.json").write_text(
            json.dumps({"graph": result.graph.document(), "names": names})
        )


def export_families(directory: Path) -> None:
    """Frozen shipped semantic paths; real registered producers, inert inventories."""
    import tempfile

    from clm_fixtures import fixture as clm_fixture
    from kev_fixtures import fixture as kev_fixture
    from test_clm_export import export as export_clm
    from test_deepseek_v2_architecture import inputs as deepseek_inputs
    from test_deepseek_v2_architecture import native_configuration
    from test_glm4_moe_lite_architecture import inputs as glm_inputs
    from test_kev_export import export as export_kev
    from test_kimi_linear_architecture import compressed_expert_inputs
    from test_lora_architecture import make_qwen_native_and_gptq, make_smollm2_lora
    from test_qwen35_architecture import TINY
    from test_vjepa2_architecture import TINY as VISUAL_TINY

    from llm_model_explorer.architecture_service import packaged_registry
    from llm_model_explorer.models import ModelCatalogue

    directory.mkdir(parents=True, exist_ok=True)
    registry = packaged_registry()

    def save(name: str, data: AnalysisInput) -> None:
        result = registry.analyze(data)
        assert result.graph is not None, (name, result.diagnostics)
        assert result.graph.coverage == "complete", (name, result.diagnostics)
        (directory / f"{name}.json").write_text(json.dumps(result.graph.document()))

    for name, data, _ in cases():
        if name in {"llama", "qwen3"}:
            save(name, data)
    visual = copy.deepcopy(VISUAL_TINY)
    visual["configuration"]["pred_num_hidden_layers"] = 2
    visual["storage"].update(
        {
            key.replace("predictor.layer.0.", "predictor.layer.1."): value
            for key, value in list(visual["storage"].items())
            if key.startswith("predictor.layer.0.")
        }
    )
    save("visual", visual_metadata(visual))
    hybrid = copy.deepcopy(TINY)
    config = hybrid["configuration"]["text_config"]
    # Exact reference ordering, independently reduced widths; no weight reads.
    config["layer_types"] = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ] * 6
    config["num_hidden_layers"] = 24
    original = dict(hybrid["storage"])
    hybrid["storage"] = {k: v for k, v in original.items() if ".layers." not in k}
    for index, variant in enumerate(config["layer_types"]):
        source_index = 1 if variant == "full_attention" else 0
        hybrid["storage"].update(
            {
                k.replace(f".layers.{source_index}.", f".layers.{index}."): v
                for k, v in original.items()
                if f".layers.{source_index}." in k
            }
        )
    save("qwen35", hybrid_metadata(hybrid))
    save("deepseek", deepseek_inputs(native_configuration()))
    save("glm", glm_inputs())
    save("kimi", compressed_expert_inputs())
    with tempfile.TemporaryDirectory(prefix="architecture-families-") as scratch:
        root = Path(scratch)
        for name in ("clm", "kev"):
            model_root = root / name
            first, second, _ = (
                kev_fixture(model_root, repeated=3) if name == "kev" else clm_fixture(model_root)
            )
            destination = model_root / "published"
            (export_kev if name == "kev" else export_clm)(first, second, destination)
            entry = ModelCatalogue(model_root).inspect_directory(destination)
            save(name, AnalysisInput.from_source(entry.pin(), tokenizer_available=True))
        (root / "lora").mkdir()
        _, composition = make_smollm2_lora(root / "lora")
        save("lora", AnalysisInput.from_source(composition, tokenizer_available=False))
        (root / "qwen-lora").mkdir()
        model_id, adapter_id = make_qwen_native_and_gptq(root / "qwen-lora")
        entries = {
            entry.summary.id: entry for entry in ModelCatalogue(root / "qwen-lora").discover()
        }
        for name, suffix in [("qwen-lora", ""), ("qwen-gptq-lora", "@gptq-int4")]:
            source = entries[f"{model_id}{suffix}+peft-lora:{adapter_id}"].pin()
            save(name, AnalysisInput.from_source(source, tokenizer_available=False))
    from llm_model_explorer.architecture_analysis import BindingContext, NumericTensor
    from llm_model_explorer.architecture_analysis import records as r
    from llm_model_explorer.architecture_analysis.model_defined import (
        analyze_definition,
        parse_definition,
    )

    examples = Path(__file__).resolve().parents[2] / "examples/model-owned-architecture"
    for name, filename in [
        ("owned-linear", "architecture.json"),
        ("owned-shared", "shared-architecture.json"),
        ("owned-generation", "generation-architecture.json"),
    ]:
        declared = parse_definition((examples / filename).read_bytes())
        physical = {}
        for parameter in declared.parameters:
            assert parameter.shape is not None
            assert all(isinstance(d, r.ArchitectureConstantDimension) for d in parameter.shape)
            dimensions = [
                d.value for d in parameter.shape if isinstance(d, r.ArchitectureConstantDimension)
            ]
            physical[parameter.name] = r.ArchitectureStorage(
                name=parameter.name, dtype="F32", shape=dimensions
            )
        numeric = {
            str(index): NumericTensor(str(index), p.name, tuple(physical[p.name].shape), "F32")
            for index, p in enumerate(declared.parameters)
        }
        result = analyze_definition(
            declared, AnalysisInput(name, {}, BindingContext(physical, numeric, False))
        )
        assert result.graph is not None and result.graph.coverage == "complete", (
            name,
            result.diagnostics,
        )
        (directory / f"{name}.json").write_text(json.dumps(result.graph.document()))


if __name__ == "__main__":
    export_families(Path(sys.argv[1])) if "--families" in sys.argv else export(Path(sys.argv[1]))
