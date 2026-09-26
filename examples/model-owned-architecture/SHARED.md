# Model-owned Shared components

`shared-architecture.json` is a complete portable definition with two encoder
layers, two attention instances and two GELU MLP instances. It uses arbitrary
file-local node IDs and explicitly maps component roles; no model adapter or
parameter-name discovery is required. Q, K, V and the output projection have
independent tensors in each layer.

The graph's `repetitions` declares the layer window. Its `templates` declares the
attention and MLP Shared families. These are repeated **structures**, not tied
weights. All concrete nodes, ports, edges and parameters remain present.

Generate an inspectable checkpoint from the repository root, in an environment
with PyTorch and Safetensors. The output must be an immediate child of the
Explorer model root; this script refuses to overwrite an existing directory.

```python
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

out = Path("/tmp/lmex-models/example-shared-components")
out.mkdir(parents=True, exist_ok=False)
weights = {}
for layer in range(2):
    for projection in ("q", "k", "v", "o"):
        weights[f"blocks.{layer}.attn.{projection}.weight"] = torch.full(
            (4, 4), float(layer + 1)
        )
    weights[f"blocks.{layer}.mlp.up.weight"] = torch.full((8, 4), float(layer + 1))
    weights[f"blocks.{layer}.mlp.down.weight"] = torch.full((4, 8), float(layer + 1))
save_file(weights, str(out / "model.safetensors"))
(out / "config.json").write_text(
    json.dumps({"model_type": "example_shared_encoder"}) + "\n", encoding="utf-8"
)
shutil.copyfile("examples/model-owned-architecture/shared-architecture.json", out / "architecture.json")
```

Restart the backend, select the checkpoint, and open Architecture Explorer.
Under Model, navigate the encoder layer repetition. Under Shared, open Dense
attention or GELU MLP, inspect structure without selecting weights, then select
instance 1. Its matrices contain twos; instance 0 contains ones. This makes an
incorrect first-instance binding easy to detect. These are synthetic values,
not trained weights or evidence of inference correctness.

Every template instance has exhaustive `nodes`, `ports`, `edges`, and
`parameters` mappings. The same role names recur across instances; their target
IDs differ. A root group's `semantic_role` attribute must match the family's
`component_role`. All mapped formulas, shapes, operations and connections must
agree. A changed operation or incomplete mapping is rejected, not silently
accepted as equivalent. The graph remains marked model-supplied.

This optional extension retains `schema_version: 1`. Existing files without
`templates` need no migration; older Explorer readers reject files containing
the new field and must be updated. See the
[contract](../../docs/spec/backend/model-owned-architecture.md#declared-shared-structures)
for the trust and size boundaries.
