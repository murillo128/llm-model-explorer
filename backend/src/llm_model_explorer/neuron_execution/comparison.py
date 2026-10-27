"""Same-prompt probability comparisons and independent bounded greedy branches."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from ..model_files import ModelError
from ..stream_metadata import NeuronComparisonOptions, NeuronCondition
from .admission import NeuronInput
from .runtime import NeuronLease, numerical_failure


@dataclass(frozen=True)
class ComparisonResult:
    probabilities: torch.Tensor
    candidate_token_ids: list[int]
    top_token_ids: list[list[int]]
    continuations: list[dict[str, Any]]


def compare_neurons(
    lease: NeuronLease,
    selection: NeuronInput,
    options: NeuronComparisonOptions,
    tokenizer: Any,
) -> ComparisonResult:
    """One lease covers all conditions. No state or hook survives a condition.

    Prefix recomputation starts with original activations on every forward. Its
    rows are absolute positions, including prompt rows during continuation.
    """
    prompt = selection.token_ids
    columns = list(selection.neuron_indices)
    vocabulary = selection.model.limits["vocabulary_size"]
    module = next(m for m in selection.model.modules if m["module_id"] == selection.module_id)
    distributions = []
    tops: list[list[int]] = []
    continuations: list[dict[str, Any]] = []

    def run_condition(condition: NeuronCondition) -> tuple[torch.Tensor, dict[str, Any]]:
        prefix = prompt
        seen = 0

        def intervene(values: torch.Tensor) -> torch.Tensor:
            nonlocal seen
            seen += 1
            if seen != 1 or tuple(values.shape) != (1, len(prefix), module["intermediate_size"]):
                raise ModelError("unsupported_representation", "Unexpected MLP activation shape.")
            assert condition.mode != "baseline"
            positions = {
                "prompt": slice(0, len(prompt)),
                "continuation": slice(len(prompt), None),
                "all": slice(None),
            }[condition.scope]
            # Advanced indexing copies selected original values. Never mutate
            # the forward's input storage or apply an operation twice to it.
            selected = values[:, positions, columns]
            if condition.mode == "zero":
                changed = torch.zeros_like(selected)
            elif condition.mode == "scale":
                changed = selected * torch.tensor(
                    condition.factor, dtype=values.dtype, device=values.device
                )
            else:
                changed = selected + torch.tensor(
                    condition.delta, dtype=values.dtype, device=values.device
                )
            replacement = values.clone()
            replacement[:, positions, columns] = changed
            return replacement

        def forward() -> tuple[torch.Tensor, int]:
            nonlocal seen
            seen = 0
            logits = lease.forward(prefix)
            if tuple(logits.shape) != (1, len(prefix), vocabulary) or (
                condition.mode != "baseline" and seen != 1
            ):
                raise ModelError("unsupported_representation", "Unexpected neuron forward result.")
            if not torch.isfinite(logits).all():
                raise numerical_failure()
            # Full-vocabulary stable softmax; expose F32 before ranking so
            # ties in the actual wire values resolve by ascending token ID.
            probabilities = torch.softmax(logits[0, -1].double(), dim=-1).float()
            if not torch.isfinite(probabilities).all():
                raise numerical_failure()
            # Greedy decoding uses the original logits. Rounding the exposed
            # probabilities can create ties between distinct finite logits.
            return probabilities, int(logits[0, -1].argmax().item())

        context = (
            nullcontext()
            if condition.mode == "baseline"
            else lease.hook(selection.module_id, intervene)
        )
        with context:
            probabilities, token = forward()
            shared_prompt = probabilities
            generated = []
            reason = "not_requested" if options.max_new_tokens == 0 else "length"
            for step in range(options.max_new_tokens):
                lease.cancellation.check()
                # Logit argmax returns the first maximum (lowest token ID).
                generated.append(token)
                if token in selection.model.limits["eos_token_ids"]:
                    reason = "eos"
                    break
                if step + 1 < options.max_new_tokens:
                    prefix = (*prefix, token)
                    probabilities, token = forward()
        lease.cancellation.check()
        text = (
            tokenizer.decode(
                generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if generated
            else ""
        )
        lease.cancellation.check()
        return shared_prompt, dict(
            condition_id=condition.id, token_ids=generated, text=text, finish_reason=reason
        )

    for condition in options.conditions:
        lease.cancellation.check()
        probabilities, continuation = run_condition(condition)
        distributions.append(probabilities)
        tops.append(
            torch.argsort(probabilities, descending=True, stable=True)[: options.top_k].tolist()
        )
        continuations.append(continuation)
    candidates = sorted({token for row in tops for token in row})
    # Gather probabilities from every full distribution, including entries
    # outside that condition's own top-k. Do not renormalize this union.
    table = torch.stack([row[candidates] for row in distributions]).contiguous()
    lease.cancellation.check()
    return ComparisonResult(table, candidates, tops, continuations)
