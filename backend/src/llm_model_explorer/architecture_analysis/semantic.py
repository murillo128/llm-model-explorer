"""Source-key annotations for the packaged reviewed descriptions, not pattern discovery."""

from . import records as r
from .core import Producer

# These suffixes are authored operation keys in dense.py/qwen35.py/vjepa2.py.
# They never select graph membership or infer computation from checkpoint names.
ROLES = {
    "q_proj": "query_projection",
    "k_proj": "key_projection",
    "v_proj": "value_projection",
    "query": "query_projection",
    "key": "key_projection",
    "value": "value_projection",
    "o_proj": "output_projection",
    "out_proj": "output_projection",
    "proj": "output_projection",
    "gate_proj": "gate_projection",
    "up_proj": "up_projection",
    "down_proj": "down_projection",
    "fc1": "up_projection",
    "fc2": "down_projection",
    "q_norm": "query_normalization",
    "k_norm": "key_normalization",
    "key_transpose": "key_transpose",
    "key_matrix_transpose": "key_transpose",
    "input_layernorm": "input_normalization",
    "post_attention_layernorm": "post_attention_normalization",
    "attention_residual": "attention_residual",
    "mlp_residual": "mlp_residual",
}


def source_key(producer: Producer, key: str) -> r.ArchitectureProvenance:
    return r.ArchitectureProvenance(
        kind="description",
        source=key,
        revision=producer.revision,
        rule="Semantic source key in the reviewed packaged description",
    )


def role_attribute(producer: Producer, role: str) -> r.ArchitectureAttribute:
    return r.ArchitectureAttribute(
        name="semantic_role", value=role, provenance=producer.provenance()
    )


def operation_role(key: str, operation: str) -> str:
    return ROLES.get(key.rsplit(".", 1)[-1], operation)
