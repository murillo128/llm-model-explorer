"""Construction vocabulary local to the reviewed static video descriptions."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from . import records as r
from .core import AnalysisInput, GraphBuilder
from .dense import shape
from .semantic import source_key
from .validation import require


@dataclass(frozen=True)
class Value:
    node: str
    port: str
    dims: r.ArchitectureShape


class VideoGraph:
    def __init__(self, inputs: AnalysisInput, builder: GraphBuilder):
        self.inputs, self.b = inputs, builder
        self.children: dict[str, list[str]] = {}
        self.groups: dict[
            str, tuple[str | None, str, dict[str, r.ArchitectureShape], dict[str, Any]]
        ] = {}
        self.parameters: dict[str, str] = {}
        self.numeric = {t.name: t for t in inputs.bindings.numeric.values()}

    def nid(self, key: str) -> str:
        return self.b.record_id("node", key)

    def attrs(self, values: dict[str, Any]) -> list[r.ArchitectureAttribute]:
        return [
            r.ArchitectureAttribute(name=k, value=v, provenance=self.b.producer.provenance())
            for k, v in values.items()
        ]

    def link(
        self, source: Value, target: Value, kind: Literal["data", "state", "context"] = "data"
    ) -> None:
        self.b.add_edge(
            r.ArchitectureEdge(
                id=self.b.record_id(
                    "edge", f"{source.node}:{source.port}->{target.node}:{target.port}"
                ),
                source=r.ArchitectureEndpoint(node_id=self.nid(source.node), port_id=source.port),
                target=r.ArchitectureEndpoint(node_id=self.nid(target.node), port_id=target.port),
                kind=kind,
                provenance=self.b.producer.provenance(),
            )
        )

    def begin(
        self,
        key: str,
        parent: str | None,
        label: str,
        inputs: Mapping[str, Value | r.ArchitectureShape],
        **attributes: Any,
    ) -> dict[str, Value]:
        dims = {
            name: value.dims if isinstance(value, Value) else value
            for name, value in inputs.items()
        }
        self.groups[key] = (parent, label, dims, attributes)
        self.children[key] = []
        if parent is not None:
            self.children[parent].append(self.nid(key))
        result = {name: Value(key, name, value) for name, value in dims.items()}
        for name, value in inputs.items():
            if isinstance(value, Value):
                self.link(value, result[name])
        return result

    def end(
        self,
        key: str,
        output: Value,
        *,
        output_name: str = "out",
        extra_outputs: Mapping[str, Value] | None = None,
    ) -> Value:
        parent, label, inputs, attributes = self.groups[key]
        result = Value(key, output_name, output.dims)
        self.link(output, result)
        ports = [
            r.ArchitecturePort(id=k, label=k, direction="input", shape=v) for k, v in inputs.items()
        ]
        ports.append(
            r.ArchitecturePort(
                id=output_name, label=output_name, direction="output", shape=output.dims
            )
        )
        for name, value in (extra_outputs or {}).items():
            self.link(value, Value(key, name, value.dims))
            ports.append(
                r.ArchitecturePort(id=name, label=name, direction="output", shape=value.dims)
            )
        if parent is None:
            types = {
                "prompt": "text",
                "configuration": (
                    "mode, frame count, image size, frame rate, schedule and guidance configuration"
                ),
                "image": "optional initial RGB image when mode=image_to_video"
                if len(attributes.get("modalities", [])) == 2
                else "required initial RGB image",
                "video": "RGB video frames",
            }
            ports = [
                port.model_copy(update={"value_type": types[port.id]}) if port.id in types else port
                for port in ports
            ]
        kwargs = {} if parent is None else {"parent_id": self.nid(parent)}
        self.b.add_node(
            r.ArchitectureGroupNode(
                id=self.nid(key),
                kind="group",
                label=label,
                children=self.children[key],
                ports=ports,
                parameter_ids=[],
                references=[],
                attributes=self.attrs(attributes),
                provenance=self.b.producer.provenance() + [source_key(self.b.producer, key)],
                **kwargs,
            ),
            semantic_key=key,
        )
        return result

    def parameter(self, name: str) -> str:
        if name in self.parameters:
            return self.parameters[name]
        physical = self.inputs.bindings.physical
        require(name in physical, "Missing required video parameter.")
        storage = physical[name]
        numeric = self.numeric.get(name)
        if storage.dtype != "F8_E4M3":
            result = self.b.native_parameter(
                name, name, shape(*storage.shape), self.b.producer.provenance()
            )
        else:
            require(
                numeric is not None and numeric.storage_format == "fp8-e4m3-scaled",
                "Missing admitted FP8 video parameter.",
            )
            assert numeric is not None
            prefix = name.removesuffix(".weight")
            companions = [
                physical[prefix + suffix]
                for suffix in (".scale_weight", ".scale_input", ".weight_scale", ".input_scale")
                if prefix + suffix in physical
            ]
            result = self.b.record_id("parameter", name)
            self.b.add_parameter(
                r.ArchitectureDirectParameter(
                    id=result,
                    name=name,
                    logical_shape=shape(*storage.shape),
                    binding="quantized",
                    storage=[storage.model_copy(update={"role": "packed_data"})]
                    + [
                        s.model_copy(
                            update={
                                "role": "activation_scale"
                                if s.name.endswith((".scale_input", ".input_scale"))
                                else "scales"
                            }
                        )
                        for s in companions
                    ],
                    inspection=r.ArchitectureAvailableInspection(
                        status="available", tensor_id=numeric.id
                    ),
                    provenance=self.b.producer.provenance(),
                )
            )
        self.parameters[name] = result
        return result

    def op(
        self,
        key: str,
        parent: str,
        operation: str,
        inputs: dict[str, Value],
        dims: r.ArchitectureShape,
        *,
        weights: tuple[str, ...] = (),
        formula: str | None = None,
        label: str | None = None,
        kind: Literal["operation", "state", "context"] = "operation",
        **attributes: Any,
    ) -> Value:
        self.children[parent].append(self.nid(key))
        params = [self.parameter(name) for name in weights]
        references: list[r.ArchitectureReference] = [
            r.ArchitectureParameterReference(kind="parameter", parameter_id=p) for p in params
        ]
        if operation == "local_tokenization" and self.inputs.bindings.tokenizer_available:
            references.append(r.ArchitectureTokenizerReference(kind="tokenizer"))
        ports = [
            r.ArchitecturePort(id=k, label=k, direction="input", shape=v.dims)
            for k, v in inputs.items()
        ]
        ports.append(r.ArchitecturePort(id="out", label="out", direction="output", shape=dims))
        kwargs = {} if formula is None else {"formula": formula}
        self.b.add_node(
            r.ArchitectureLeafNode(
                id=self.nid(key),
                parent_id=self.nid(parent),
                kind=kind,
                label=label or operation.replace("_", " "),
                operation=operation,
                ports=ports,
                parameter_ids=params,
                references=references,
                attributes=self.attrs(attributes),
                provenance=self.b.producer.provenance() + [source_key(self.b.producer, key)],
                **kwargs,
            ),
            semantic_key=key,
        )
        for port, value in inputs.items():
            self.link(value, Value(key, port, value.dims), "state" if kind == "state" else "data")
        return Value(key, "out", dims)

    def linear(self, key: str, parent: str, x: Value, module: str) -> Value:
        storage = self.inputs.bindings.physical
        weight = storage[module + ".weight"]
        require(len(weight.shape) == 2, "Video linear requires a matrix.")
        dims = [*(x.dims or [])[:-1], *shape(weight.shape[0])]
        names = (module + ".weight",) + ((module + ".bias",) if module + ".bias" in storage else ())
        return self.op(
            key,
            parent,
            "linear",
            {"x": x},
            dims,
            weights=names,
            formula="out = x @ W^T + b" if len(names) == 2 else "out = x @ W^T",
            module=module,
        )

    def unary(self, key: str, parent: str, operation: str, x: Value, **attributes: Any) -> Value:
        return self.op(key, parent, operation, {"x": x}, x.dims, **attributes)

    def residual(self, key: str, parent: str, x: Value, branch: Value) -> Value:
        return self.op(
            key, parent, "add", {"x": x, "branch": branch}, x.dims, formula="out = x + branch"
        )

    def repetition(self, key: str, parent: str, instances: list[str], variant: str) -> None:
        self.b.add_repetition(
            r.ArchitectureRepetition(
                id=self.b.record_id("repetition", key),
                parent_id=self.nid(parent),
                label="Blocks",
                instances=[
                    r.ArchitectureRepetitionInstance(
                        node_id=self.nid(node), index=index, variant=variant
                    )
                    for index, node in enumerate(instances)
                ],
            )
        )
