"""Validate an explicit compute target without allocating tensors."""

import re


def validate_device(value: str) -> str:
    if value == "cpu":
        return value  # No CUDA import/probe is needed on the CPU startup path.
    if value == "cuda":
        value = "cuda:0"
    if not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value):
        raise ValueError("device must be cpu, cuda, or cuda:N (a nonnegative device index)")

    import torch

    index = int(value.split(":")[1])
    try:
        available = torch.cuda.is_available() and index < torch.cuda.device_count()
        if available:
            # Checks driver/device initialization without creating a CUDA tensor.
            torch.cuda.get_device_properties(index)
    except (RuntimeError, AssertionError) as exc:
        raise ValueError(f"requested device {value} is unavailable: {exc}") from exc
    if not available:
        raise ValueError(
            f"requested device {value} is unavailable; check PyTorch and the CUDA driver"
        )
    return value
