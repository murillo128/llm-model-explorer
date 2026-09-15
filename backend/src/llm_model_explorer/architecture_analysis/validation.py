"""Bounded graph validation against the admitted metadata, without tensor access."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from . import records as r

MAX_BYTES = 33_554_432
SAFE = 2**53 - 1


class GraphError(ValueError):
    """Only static, safe messages are exposed to callers."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphError("invalid_graph", message)


def bounded_chunks(chunks: Iterable[bytes], limit: int = MAX_BYTES) -> int:
    size = 0
    for chunk in chunks:
        size += len(chunk)
        if size > limit:
            raise GraphError("unsupported_size", "Architecture exceeds the supported byte limit.")
    return size


def preflight(value: object, limit: int = MAX_BYTES) -> None:
    """Bound raw or typed record trees before making serialization copies."""
    # Walk iteratively with iterators so wide collections do not grow a second stack.
    stack = [iter((value,))]
    cost = 0
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        cost += 1
        if cost > limit or len(stack) > 32:
            raise GraphError("unsupported_size", "Architecture exceeds structural limits.")
        if isinstance(item, str):
            if len(item) > 16_384:
                raise GraphError(
                    "unsupported_size", "Architecture text exceeds the supported limit."
                )
            cost += len(item.encode("utf-8"))
        elif isinstance(item, BaseModel):
            stack.append(iter(item.__dict__.values()))
        elif isinstance(item, dict):
            if len(item) > limit:
                raise GraphError("unsupported_size", "Architecture collection exceeds the limit.")
            stack.append(iter(part for pair in item.items() for part in pair))
        elif isinstance(item, (list, tuple)):
            if len(item) > limit:
                raise GraphError("unsupported_size", "Architecture collection exceeds the limit.")
            stack.append(iter(item))


def json_chunks(value: object, limit: int = MAX_BYTES) -> Iterable[bytes]:
    """Preflight bounds before encoding; never allocate a whole serialized graph."""
    preflight(value, limit)
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    size = 0
    for part in encoder.iterencode(value):
        data = part.encode("utf-8")
        size += len(data)
        if size > limit:
            raise GraphError("unsupported_size", "Architecture exceeds the supported byte limit.")
        yield data


def serialized_size(value: object, limit: int = MAX_BYTES) -> int:
    return bounded_chunks(json_chunks(value, limit), limit)


def product(shape: Iterable[int]) -> int:
    dims = tuple(shape)
    require(all(type(d) is int and 0 <= d <= SAFE for d in dims), "Unsafe dimension.")
    if 0 in dims:
        return 0
    total = 1
    for dim in dims:
        total *= dim
        require(total <= SAFE, "Unsafe dimension product.")
    return total


@dataclass(frozen=True)
class NumericTensor:
    id: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    storage_format: str = "safetensors"


@dataclass(frozen=True)
class BindingContext:
    """Path-free projection of already guarded checkpoint inventories."""

    physical: Mapping[str, r.ArchitectureStorage]
    numeric: Mapping[str, NumericTensor]
    tokenizer_available: bool


def unique(records: Iterable[Any], field: str = "id") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        key = getattr(record, field)
        require(key not in result, "Duplicate record identity.")
        result[key] = record
    return result


def terminals(records: Mapping[str, Any], link: str) -> dict[str, str]:
    """Linear-time closure/cycle checking, including long alias and parent chains."""
    resolved: dict[str, str] = {}
    for start in records:
        path: set[str] = set()
        key = start
        while key not in resolved:
            require(key in records, "Unresolved linked record.")
            require(key not in path, "Cyclic record linkage.")
            path.add(key)
            nxt = getattr(records[key], link, None)
            if nxt is None:
                break
            key = nxt
        end = resolved.get(key, key)
        for item in path:
            resolved[item] = end
    return resolved


def constants(shape: r.ArchitectureShape) -> tuple[int, ...] | None:
    if shape is None or not all(isinstance(d, r.ArchitectureConstantDimension) for d in shape):
        return None
    return tuple(d.value for d in shape if isinstance(d, r.ArchitectureConstantDimension))


def mismatch(left: r.ArchitectureShape, right: r.ArchitectureShape) -> bool:
    return (
        left is not None
        and right is not None
        and (
            len(left) != len(right)
            or any(
                isinstance(a, r.ArchitectureConstantDimension)
                and isinstance(b, r.ArchitectureConstantDimension)
                and a.value != b.value
                for a, b in zip(left, right, strict=False)
            )
        )
    )


def validate_packed_binding(
    parameter: r.ArchitectureParameter, tensor: NumericTensor, geometry: tuple[int, ...]
) -> None:
    """Verify complete supported logical geometry against observed companion storage."""
    require(
        parameter.name.endswith(".weight") and len(geometry) == 2 and min(geometry) > 0,
        "Unsupported packed logical parameter geometry.",
    )
    output, inputs = geometry
    prefix = parameter.name.removesuffix(".weight")
    expected: dict[str, tuple[str, tuple[int, ...]]]
    if tensor.storage_format == "gptq-int4":
        require(
            tensor.dtype == "I32" and inputs % 128 == 0 and output % 8 == 0,
            "Unsupported GPTQ logical representation.",
        )
        expected = {
            "qweight": ("I32", (inputs // 8, output)),
            "qzeros": ("I32", (inputs // 128, output // 8)),
            "scales": ("F16", (inputs // 128, output)),
            "g_idx": ("I32", (inputs,)),
        }
    elif tensor.storage_format == "nvfp4":
        require(tensor.dtype == "U8" and inputs % 16 == 0, "Unsupported NVFP4 representation.")
        expected = {
            "weight": ("U8", (output, inputs // 2)),
            "weight_scale": ("F8_E4M3", (output, inputs // 16)),
            "weight_scale_2": ("F32", ()),
            "input_scale": ("F32", ()),
        }
    else:
        raise GraphError("invalid_graph", "Unsupported packed numeric representation.")
    actual = {storage.name: (storage.dtype, tuple(storage.shape)) for storage in parameter.storage}
    require(
        len(parameter.storage) == len(expected)
        and actual == {prefix + "." + suffix: value for suffix, value in expected.items()},
        "Packed storage companions disagree with the complete logical tensor.",
    )


def validate_graph(graph: r.ArchitectureGraph, context: BindingContext) -> None:
    """Require schema-valid records; check all cross-record and inventory semantics."""
    nodes = unique(graph.nodes)
    params = unique(graph.parameters)
    unique([*graph.nodes, *graph.parameters, *graph.edges, *graph.repetitions])
    symbols = unique(graph.symbols, "name")
    ports = {node.id: unique(node.ports) for node in graph.nodes}
    child_positions: dict[str, dict[str, int]] = {}
    for d in graph.diagnostics:
        require(d.node_id is None or d.node_id in nodes, "Diagnostic node does not exist.")
        require(
            d.parameter_id is None or d.parameter_id in params,
            "Diagnostic parameter does not exist.",
        )

    def shape(dims: r.ArchitectureShape) -> None:
        if dims is None:
            return
        for dim in dims:
            if isinstance(dim, r.ArchitectureSymbolDimension):
                require(dim.name in symbols, "Undeclared shape symbol.")
            elif isinstance(dim, r.ArchitectureExpressionDimension):
                require(all(s in symbols for s in dim.symbols), "Undeclared expression symbol.")
        known = constants(dims)
        if known is not None:
            product(known)

    for node in graph.nodes:
        if node.parent_id is not None:
            parent = nodes.get(node.parent_id)
            require(isinstance(parent, r.ArchitectureGroupNode), "Parent must be a group.")
        if isinstance(node, r.ArchitectureGroupNode):
            require(len(set(node.children)) == len(node.children), "Duplicate child.")
            child_positions[node.id] = {key: pos for pos, key in enumerate(node.children)}
            for child in node.children:
                require(
                    child in nodes and nodes[child].parent_id == node.id,
                    "Child and parent disagree.",
                )
        require(all(key in params for key in node.parameter_ids), "Unknown node parameter.")
        for ref in node.references:
            if isinstance(ref, r.ArchitectureParameterReference):
                require(ref.parameter_id in params, "Unknown parameter resource.")
            elif isinstance(ref, r.ArchitectureTokenizerReference):
                require(context.tokenizer_available, "Tokenizer capability is unavailable.")
        for port in node.ports:
            shape(port.shape)
    for node in graph.nodes:
        if node.parent_id is not None:
            require(node.id in child_positions[node.parent_id], "Parent and child disagree.")
    terminals(nodes, "parent_id")
    aliases = terminals(params, "alias_of")
    for rep in graph.repetitions:
        require(rep.parent_id in child_positions, "Repetition parent must be a group.")
        unique(rep.instances, "node_id")
        indices = [i.index for i in rep.instances]
        require(indices == sorted(set(indices)), "Repetition indices must be strictly ordered.")
        positions = []
        for instance in rep.instances:
            instance_node = nodes.get(instance.node_id)
            require(
                isinstance(instance_node, r.ArchitectureGroupNode)
                and instance_node.parent_id == rep.parent_id,
                "Repetition instance must be a direct child group.",
            )
            positions.append(child_positions[rep.parent_id][instance.node_id])
        require(positions == sorted(positions), "Repetition disagrees with parent order.")

    for edge in graph.edges:
        for ep in (edge.source, edge.target):
            require(
                ep.node_id in ports and ep.port_id in ports[ep.node_id], "Unknown edge endpoint."
            )
        sn, tn = nodes[edge.source.node_id], nodes[edge.target.node_id]
        sp, tp = ports[sn.id][edge.source.port_id], ports[tn.id][edge.target.port_id]
        compatible = sp.direction == "output" and tp.direction == "input"
        compatible |= (
            sn.kind == "group" and tn.parent_id == sn.id and sp.direction == tp.direction == "input"
        )
        compatible |= (
            tn.kind == "group"
            and sn.parent_id == tn.id
            and sp.direction == tp.direction == "output"
        )
        require(compatible, "Incompatible port directions.")
        require(
            sn.parent_id == tn.parent_id or tn.parent_id == sn.id or sn.parent_id == tn.id,
            "Edge bypasses a group boundary.",
        )
        if mismatch(sp.shape, tp.shape):
            require(
                any(
                    d.code == "shape_mismatch" and d.node_id in (sn.id, tn.id)
                    for d in graph.diagnostics
                ),
                "Known connected dimensions disagree without a diagnostic.",
            )

    for parameter in graph.parameters:
        shape(parameter.logical_shape)
        for storage in parameter.storage:
            product(storage.shape)
            observed = context.physical.get(storage.name)
            require(
                observed is not None
                and observed.dtype == storage.dtype
                and observed.shape == storage.shape,
                "Unverified physical storage descriptor.",
            )
        if isinstance(parameter, r.ArchitectureFusedParameter):
            require(
                parameter.region.storage_name in {s.name for s in parameter.storage},
                "Fused region storage is unresolved.",
            )
        if parameter.binding == "native":
            geometry = constants(parameter.logical_shape)
            require(
                len(parameter.storage) == 1 and parameter.storage[0].name == parameter.name,
                "Native binding must name one complete tensor.",
            )
            require(
                geometry is not None and list(geometry) == parameter.storage[0].shape,
                "Native logical and physical geometry disagree.",
            )
        if isinstance(parameter, r.ArchitectureAliasParameter):
            target = params[aliases[parameter.id]]
            require(parameter.logical_shape == target.logical_shape, "Alias geometry disagrees.")
        if parameter.inspection.status != "available":
            continue
        inspection = parameter.inspection
        require(
            parameter.binding in ("native", "quantized", "alias"),
            "Inspection requires a complete logical parameter.",
        )
        native = params[aliases[parameter.id]]
        geometry = constants(parameter.logical_shape)
        require(geometry is not None and len(geometry) in (1, 2), "Unsupported inspection rank.")
        assert geometry is not None
        require(
            native.binding in ("native", "quantized")
            and native.logical_shape == parameter.logical_shape
            and native.inspection.status == "available"
            and native.inspection.tensor_id == inspection.tensor_id,
            "Alias inspection must resolve to the same complete logical tensor.",
        )
        tensor = context.numeric.get(inspection.tensor_id)
        require(
            tensor is not None and tensor.name == native.name and tensor.shape == geometry,
            "Inspection tensor is outside the admitted numeric inventory or geometry.",
        )
        assert tensor is not None
        if native.binding == "quantized":
            validate_packed_binding(native, tensor, geometry)
            continue
        require(
            any(
                s.name == tensor.name
                and s.dtype == tensor.dtype
                and s.dtype in ("F32", "F16", "BF16", "float32", "float16", "bfloat16")
                and s.role not in ("scales", "packed", "packed_data")
                for s in native.storage
            ),
            "Inspection storage identity is not native.",
        )
