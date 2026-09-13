"""Read a few samples from an operator-supplied SmolLM2 Base directory, offline."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def samples(directory: Path):
    config = json.loads((directory / "config.json").read_text())
    identity = config.get("_name_or_path", config.get("name_or_path", directory.name))
    if "instruct" in identity.lower() or "instruct" in directory.name.lower():
        raise ValueError("Reference acceptance requires SmolLM2-135M Base, not Instruct")
    if config.get("hidden_size") != 576 or config.get("num_hidden_layers") != 30:
        raise ValueError("Operator path does not describe the SmolLM2-135M architecture")
    selected = {}
    inventory = []
    endings = (
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.up_proj.weight",
        "embed_tokens.weight",
    )
    for file in sorted(directory.rglob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as weights:
            for name in weights.keys():  # noqa: SIM118 -- safetensors reader is not iterable
                view = weights.get_slice(name)
                shape = view.get_shape()
                inventory.append({"name": name, "shape": shape})
                ending = next((e for e in endings if name.endswith(e)), None)
                if ending and ending not in selected:
                    coordinates = (
                        [(0, 0), (0, shape[-1] - 1)]
                        if len(shape) == 1
                        else [
                            (0, 0),
                            (min(8, shape[0] - 1), min(8, shape[1] - 1)),
                            (shape[0] - 1, shape[1] - 1),
                        ]
                    )
                    selected[ending] = {
                        "name": name,
                        "shape": shape,
                        "samples": [
                            {
                                "row": row,
                                "column": column,
                                "value": float(view[column : column + 1].float()[0])
                                if len(shape) == 1
                                else float(view[row : row + 1, column : column + 1].float()[0, 0]),
                            }
                            for row, column in coordinates
                        ],
                    }
    if len(selected) != 4:
        raise ValueError("Reference checkpoint lacks one of the four required tensor roles")
    return {"inventory": inventory, "selected": list(selected.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    print(json.dumps(samples(parser.parse_args().model_directory)))
