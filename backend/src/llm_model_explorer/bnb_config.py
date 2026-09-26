"""Lightweight admission predicate for the reviewed BitsAndBytesConfig subset."""


def is_supported_config(quantization_config: object) -> bool:
    """Accept the reviewed Transformers BitsAndBytesConfig serialization only."""
    if not isinstance(quantization_config, dict):
        return False
    exact = {
        "_load_in_4bit": True,
        "_load_in_8bit": False,
        "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "llm_int8_enable_fp32_cpu_offload": False,
        "llm_int8_has_fp16_weight": False,
        "llm_int8_skip_modules": None,
        "llm_int8_threshold": 6.0,
        "load_in_4bit": True,
        "load_in_8bit": False,
        "quant_method": "bitsandbytes",
    }
    if set(quantization_config) != set(exact) | {"bnb_4bit_compute_dtype"}:
        return False
    if any(
        type(quantization_config[key]) is not type(value) or quantization_config[key] != value
        for key, value in exact.items()
    ):
        return False
    return quantization_config["bnb_4bit_compute_dtype"] in {
        "float16",
        "bfloat16",
        "float32",
    }
