# NF4 reference comparison

The pinned small reference's `model.layers.0.self_attn.k_proj.weight`
([192, 576]) was compared at logical offset 0 for 4,096 values. The local
reference group matched the reported repository revision and its six companion
records are identified by shape and SHA-256 in
[`bnb-nf4-smollm2-reference.json`](bnb-nf4-smollm2-reference.json). No model
weights are committed.

Reproduce with a locally available copy of the pinned reference and an
environment containing the backend dependencies plus bitsandbytes 0.50.2:

```sh
PYTHONPATH=backend/src python backend/scripts/check_bnb_nf4_reference.py \
  --model-root /path/to/local/model-root \
  --model-id 'HuggingFaceTB/SmolLM2-135M@bnb-nf4-dq' \
  --output backend/evidence/bnb-nf4-smollm2-reference.json
```

The independent comparison uses bitsandbytes 0.50.2's NF4 codebook and CPU
nested-blockwise dequantizer with the high-first mapping in its pinned kernel
source. That CPU wheel does not register the `dequantize_4bit` kernel for CPU,
so the oracle composes those reference operations directly. The recorded run
matched all 4,096 float32 bit patterns; maximum absolute error was 0.0. This
checks one range in one actual stored matrix and does not claim full-checkpoint
acceptance.
