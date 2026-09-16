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

        with patch.object(GraphBuilder, "record_id", record):
            result = registry.analyze(inputs)
        assert result.graph is not None, result
        (directory / f"{name}.json").write_text(
            json.dumps({"graph": result.graph.document(), "names": names})
        )


if __name__ == "__main__":
    export(Path(sys.argv[1]))
