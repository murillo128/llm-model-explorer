"""Card vocabulary for reviewed primitive operations, shared by static producers.

This is explanatory text, never an evaluator. Unknown/composite operations keep
their description's formula; model-owned records do not use these defaults.
"""

from collections.abc import Iterable

from . import records as r


def primitive_formula(node: r.ArchitectureNode, parameters: Iterable[str]) -> str | None:
    inputs = [port.label for port in node.ports if port.direction == "input"]
    outputs = [port.label for port in node.ports if port.direction == "output"]
    if node.kind != "operation" or len(outputs) != 1:
        return None
    out = outputs[0]
    attrs = {attribute.name for attribute in node.attributes}
    names = {name.rsplit(".", 1)[-1] for name in parameters}
    operation = node.operation
    if operation == "linear" and len(inputs) == 1 and "weight" in names:
        return f"{out} = {inputs[0]} @ weightᵀ" + (" + bias" if "bias" in names else "")
    if operation in {"add", "multiply", "matmul", "add_mask"} and len(inputs) == 2:
        operator = {"add": "+", "add_mask": "+", "multiply": "*", "matmul": "@"}[operation]
        return f"{out} = {inputs[0]} {operator} {inputs[1]}"
    if len(inputs) == 1:
        x = inputs[0]
        if operation == "scale" and "factor" in attrs:
            return f"{out} = factor * {x}"
        if operation == "softmax":
            return f"{out} = softmax({x}" + (", axis=axis)" if "axis" in attrs else ")")
        if operation in {"reshape", "transpose"}:
            # The exact geometry remains in the port descriptors. A compact
            # signature needs no hidden axes/configuration variable on the card.
            return f"{out} = {operation}({x}, ...)"
    return None
