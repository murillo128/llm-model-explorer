"""Bounded observation at the lease's reusable post-product intervention seam."""

import torch

from ..model_files import ModelError
from .admission import NeuronInput
from .runtime import NeuronLease, numerical_failure


def capture_activations(lease: NeuronLease, selection: NeuronInput) -> torch.Tensor:
    """Copy only requested columns; never retain or modify the projection input."""
    selection.validate()
    module = next(m for m in selection.model.modules if m["module_id"] == selection.module_id)
    captured: torch.Tensor | None = None

    def observe(values: torch.Tensor) -> None:
        nonlocal captured
        if captured is not None or tuple(values.shape) != (
            1,
            len(selection.token_ids),
            module["intermediate_size"],
        ):
            raise ModelError("unsupported_representation", "Unexpected MLP activation shape.")
        # Advanced indexing makes an independent selected-column buffer on the
        # compute device. Only the bounded output is converted/copied to CPU F32.
        captured = (
            values.detach()[0, :, list(selection.neuron_indices)]
            .to(device="cpu", dtype=torch.float32, copy=True)
            .contiguous()
        )
        if not torch.isfinite(captured).all():
            raise numerical_failure()

    try:
        with lease.hook(selection.module_id, observe):
            lease.forward(selection.token_ids)
        if captured is None:
            raise ModelError("unsupported_representation", "MLP observation point was not reached.")
        return captured
    finally:
        # Includes failures after capture, such as logits validation/cancellation.
        captured = None
