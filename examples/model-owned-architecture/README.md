# A checkpoint that carries its own architecture

`architecture.json` is a complete, model-independent example of the version-1
[portable definition contract](../../docs/spec/backend/model-owned-architecture.md).
It declares a linear projection and GELU, explicit group ports and native weights.
It has no Python entry point, runtime tensor IDs, cached graph IDs or NFL coupling.

Generate a tiny untrained local checkpoint with the script below. Run it from the
repository root in an environment with PyTorch and Safetensors installed. Choose
an output directory that is an immediate child of the Explorer's model root.
The script refuses to overwrite an existing checkpoint directory.

```python
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

out = Path("/tmp/lmex-models/example-owned-architecture")
out.mkdir(parents=True, exist_ok=False)
generator = torch.Generator().manual_seed(17)
save_file(
    {
        "encoder.proj.weight": torch.randn(4, 3, generator=generator),
        "encoder.proj.bias": torch.zeros(4),
    },
    str(out / "model.safetensors"),
)
(out / "config.json").write_text(
    json.dumps({"model_type": "my_experimental_encoder"}) + "\n", encoding="utf-8"
)
shutil.copyfile("examples/model-owned-architecture/architecture.json", out / "architecture.json")
```

Restart the backend with `/tmp/lmex-models` as `--model-root` and a separate cache
root. The model does not need a recognized architecture class or tokenizer. The
graph is marked **model-supplied**, and both native parameters can be inspected.
These values are random initialization, not trained weights or scientific evidence.

An exporter in your own model repository can generate all repeated nodes and
connections from its configuration. Changing layers, branches, operation names or
weights requires only a new matching definition/checkpoint, not an Explorer PR.
Keep model/exporter correspondence tests in that repository. File-format changes
are different from model-architecture changes: only the former may need a new
Explorer reader.
