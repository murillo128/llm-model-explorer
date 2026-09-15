"""Small offline checkpoints joining the reviewed numeric and text fixtures."""

import json
import sys
from dataclasses import replace
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from acceptance.architecture_fixtures import generate as architecture_fixtures
from acceptance.architecture_fixtures import write_checkpoint


def packed_table(root: Path, family: str, *, unresolved: bool = False):
    # Reuse the reviewed scalar oracle, which does not import production decoders.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend/tests"))
    try:
        from quantized_oracles import Stored, gptq_fixture, nvfp4_fixture

        fixture = gptq_fixture() if family == "qwen3" else nvfp4_fixture()
    finally:
        sys.path.pop(0)
    prefix = "model.embed_tokens" if family == "qwen3" else "model.language_model.embed_tokens"
    fixture = replace(
        fixture,
        storage=tuple(
            replace(item, name=prefix + "." + item.name.rsplit(".", 1)[1])
            for item in fixture.storage
        ),
    )
    stored = fixture
    if unresolved:
        stored = replace(
            fixture, storage=(*fixture.storage, Stored("unknown.packed", "I32", (2,), bytes(8)))
        )
    directory = stored.write(root, split=True)
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config["architectures"] = [
        "Qwen3ForCausalLM" if family == "qwen3" else "Qwen3_5ForConditionalGeneration"
    ]
    dims = config if family == "qwen3" else config.setdefault("text_config", {})
    dims.update(vocab_size=fixture.shape[0], hidden_size=fixture.shape[1])
    path.write_text(json.dumps(config))
    return fixture


def generate(root: Path):
    architecture_fixtures(root)
    for family in ("qwen3", "qwen35"):
        config = json.loads((root / family / "config.json").read_text())
        config.pop("quantization_config")
        dims = config if family == "qwen3" else config["text_config"]
        name = (
            "model.embed_tokens.weight"
            if family == "qwen3"
            else "model.language_model.embed_tokens.weight"
        )
        write_checkpoint(
            root / (family + "-native"),
            config,
            {name: {"dtype": "F32", "shape": [dims["vocab_size"], dims["hidden_size"]]}},
        )
    # Six IDs fit every tiny text vocabulary. Repeated words produce repeated IDs
    # while preserving source sequence positions. No tokenizer is added to V-JEPA.
    tokenizer = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "<bos>": 1, "one": 2, "two": 3, "three": 4, "four": 5},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<bos>"
    )
    for family in ("smollm2", "qwen3", "qwen35", "qwen3-native", "qwen35-native"):
        wrapped.save_pretrained(root / family)
