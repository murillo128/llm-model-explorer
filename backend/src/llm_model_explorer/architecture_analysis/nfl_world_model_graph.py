"""Graph helpers for the reviewed BDB-2025 NFL world-model description."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from . import records as r
from .core import GraphBuilder, Producer
from .nfl_world_model_config import Config, parameter_shapes
from .semantic import operation_role, role_attribute, source_key


def shape(*dims: int | str) -> r.ArchitectureShape:
    return [
        r.ArchitectureConstantDimension(kind="constant", value=x)
        if isinstance(x, int) else r.ArchitectureSymbolDimension(kind="symbol", name=x)
        for x in dims
    ]


def expression(text: str, *symbols: str) -> r.ArchitectureExpressionDimension:
    return r.ArchitectureExpressionDimension(kind="expression", text=text, symbols=list(symbols))


@dataclass(frozen=True)
class Value:
    node: str
    port: str
    shape: r.ArchitectureShape


class Graph:
    def __init__(self, builder: GraphBuilder, config: Config, producer: Producer) -> None:
        self.b, self.c, self.producer = builder, config, producer
        self.children: dict[str, list[str]] = {}
        self.parameters = {
            name: builder.native_parameter(
                name, name, shape(*dims),
                [*producer.provenance(), r.ArchitectureProvenance(kind="storage", source=name)],
            )
            for name, dims in parameter_shapes(config).items()
        }

    def nid(self, key: str) -> str:
        return self.b.record_id("node", key)

    def port(self, name: str, dims: r.ArchitectureShape, output: bool = False) -> r.ArchitecturePort:
        return r.ArchitecturePort(id=name, direction="output" if output else "input", label=name, shape=dims)

    def link(self, value: Value, node: str, port: str, kind: Literal["data", "context"] = "data") -> None:
        self.b.add_edge(r.ArchitectureEdge(
            id=self.b.record_id("edge", f"{value.node}:{value.port}->{node}:{port}:{kind}"),
            source=r.ArchitectureEndpoint(node_id=self.nid(value.node), port_id=value.port),
            target=r.ArchitectureEndpoint(node_id=self.nid(node), port_id=port),
            kind=kind, provenance=self.producer.provenance(),
        ))

    def node(self, key: str, operation: str, inputs: dict[str, Value], outputs: dict[str, r.ArchitectureShape],
             *, parent: str | None = None, kind: Literal["operation", "input", "output", "context"] = "operation",
             parameters: tuple[str, ...] = (), formula: str | None = None,
             attributes: dict[str, Any] | None = None) -> dict[str, Value]:
        args: dict[str, Any] = {}
        if parent is not None:
            args["parent_id"] = self.nid(parent); self.children[parent].append(self.nid(key))
        if formula is not None: args["formula"] = formula
        pids = [self.parameters[p] for p in parameters]
        attrs = [r.ArchitectureAttribute(name=n, value=v, provenance=self.producer.provenance()) for n, v in (attributes or {}).items()]
        attrs.append(role_attribute(self.producer, operation_role(key, operation)))
        self.b.add_node(r.ArchitectureLeafNode(
            id=self.nid(key), kind=kind, label=operation_role(key, operation).replace("_", " "), operation=operation,
            ports=[self.port(n, v.shape) for n, v in inputs.items()] + [self.port(n, dims, True) for n, dims in outputs.items()],
            parameter_ids=pids,
            references=[r.ArchitectureParameterReference(kind="parameter", parameter_id=p) for p in pids],
            attributes=attrs, provenance=self.producer.provenance() + [source_key(self.producer, key)], **args,
        ), semantic_key=key)
        for n, value in inputs.items(): self.link(value, key, n, "context" if value.node.startswith("training.") else "data")
        return {n: Value(key, n, dims) for n, dims in outputs.items()}

    def op(self, key: str, operation: str, inputs: dict[str, Value], output: r.ArchitectureShape, **kwargs: Any) -> Value:
        return self.node(key, operation, inputs, {"out": output}, **kwargs)["out"]

    def input(self, key: str, output: r.ArchitectureShape, context: bool = False) -> Value:
        return self.op(key, "training_context" if context else "structured_input", {}, output, kind="context" if context else "input")

    def group(self, key: str, inputs: dict[str, Value]) -> dict[str, Value]:
        self.children[key] = []
        for n, value in inputs.items(): self.link(value, key, n, "context" if value.node.startswith("training.") else "data")
        return {n: Value(key, n, value.shape) for n, value in inputs.items()}

    def end_group(self, key: str, inputs: dict[str, Value], output: Value, *, parent: str | None = None,
                  label: str | None = None, role: str | None = None, attributes: dict[str, Any] | None = None,
                  module_reference: bool = True) -> Value:
        self.link(output, key, "out")
        args: dict[str, Any] = {}
        if parent is not None:
            args["parent_id"] = self.nid(parent); self.children[parent].append(self.nid(key))
        attrs = [r.ArchitectureAttribute(name=n, value=v, provenance=self.producer.provenance()) for n, v in (attributes or {}).items()]
        if role: attrs.append(role_attribute(self.producer, role))
        self.b.add_node(r.ArchitectureGroupNode(
            id=self.nid(key), kind="group", label=label or key,
            ports=[self.port(n, v.shape) for n, v in inputs.items()] + [self.port("out", output.shape, True)],
            children=self.children[key], parameter_ids=[],
            references=[r.ArchitectureModuleReference(kind="module", name=key)] if module_reference else [],
            attributes=attrs, provenance=self.producer.provenance() + [source_key(self.producer, key)], **args,
        ), semantic_key=key)
        return Value(key, "out", output.shape)

    def affine(self, key: str, x: Value, output: r.ArchitectureShape, parent: str, *, weight: str, bias: str,
               operation: str = "linear", attributes: dict[str, Any] | None = None) -> Value:
        return self.op(key, operation, {"x": x}, output, parent=parent, parameters=(weight, bias),
                       formula="LayerNorm(x; weight, bias, epsilon)" if operation == "layer_norm" else "x W^T + bias",
                       attributes=attributes)

    def transformer(self, stack: str, x: Value, mask: Value, *, layers: int,
                    batch: r.ArchitectureDimension, sequence: r.ArchitectureDimension, causal: bool) -> Value:
        d, h, hd = self.c.d, self.c.heads, self.c.head_dim
        hidden = [batch, sequence, *shape(d)]; heads = [batch, *shape(h), sequence, *shape(hd)]
        scores = [batch, *shape(h), sequence, sequence]; instances = []
        for i in range(layers):
            key = f"{stack}.layers.{i}"; ins = self.group(key, {"x": x, "padding_mask": mask}); residual = ins["x"]
            attn_key = key + ".self_attn"; attn_in = self.group(attn_key, {"x": ins["x"], "padding_mask": ins["padding_mask"]})
            fused = self.affine(attn_key + ".in_proj", attn_in["x"], [batch, sequence, *shape(3*d)], attn_key,
                                weight=key + ".self_attn.in_proj_weight", bias=key + ".self_attn.in_proj_bias")
            qkv = self.node(attn_key + ".split_qkv_heads", "split_qkv_heads", {"x": fused},
                            {"query": heads, "key": heads, "value": heads}, parent=attn_key,
                            attributes={"heads": h, "head_dim": hd})
            score = self.op(attn_key + ".scores", "scaled_query_key_product",
                            {"query": qkv["query"], "key": qkv["key"], "padding_mask": attn_in["padding_mask"]},
                            scores, parent=attn_key, formula="masked(Q K^T / sqrt(head_dim))",
                            attributes={"causal": causal, "padding_mask": True})
            prob = self.op(attn_key + ".softmax", "softmax", {"scores": score}, scores, parent=attn_key, attributes={"axis": -1})
            weighted = self.op(attn_key + ".weighted_values", "attention_value_product", {"probabilities": prob, "value": qkv["value"]}, heads, parent=attn_key)
            merged = self.op(attn_key + ".merge_heads", "transpose_merge_heads", {"x": weighted}, hidden, parent=attn_key)
            attn = self.affine(attn_key + ".out_proj", merged, hidden, attn_key,
                               weight=key + ".self_attn.out_proj.weight", bias=key + ".self_attn.out_proj.bias")
            attn = self.end_group(attn_key, attn_in, attn, parent=key, label="Attention", role="attention", attributes={"causal": causal})
            residual = self.op(key + ".attention_residual", "residual_add", {"skip": residual, "branch": attn}, hidden, parent=key)
            norm = self.affine(key + ".norm1", residual, hidden, key, weight=key + ".norm1.weight", bias=key + ".norm1.bias",
                               operation="layer_norm", attributes={"epsilon": 1e-5})
            mlp_key = key + ".mlp"; mlp_in = self.group(mlp_key, {"x": norm})
            up = self.affine(key + ".linear1", mlp_in["x"], [batch, sequence, *shape(self.c.ff)], mlp_key,
                             weight=key + ".linear1.weight", bias=key + ".linear1.bias")
            active = self.op(key + ".gelu", "gelu", {"x": up}, up.shape, parent=mlp_key)
            down = self.affine(key + ".linear2", active, hidden, mlp_key, weight=key + ".linear2.weight", bias=key + ".linear2.bias")
            down = self.end_group(mlp_key, mlp_in, down, parent=key, label="MLP", role="mlp", module_reference=False)
            residual = self.op(key + ".mlp_residual", "residual_add", {"skip": norm, "branch": down}, hidden, parent=key)
            x = self.affine(key + ".norm2", residual, hidden, key, weight=key + ".norm2.weight", bias=key + ".norm2.bias",
                            operation="layer_norm", attributes={"epsilon": 1e-5})
            x = self.end_group(key, ins, x, parent=stack, label=f"Layer {i}", attributes={"post_norm": True, "dropout": self.c.encoder_dropout})
            instances.append(r.ArchitectureRepetitionInstance(node_id=self.nid(key), index=i, variant="torch_transformer_encoder_layer_post_norm"))
        self.b.add_repetition(r.ArchitectureRepetition(
            id=self.b.record_id("repetition", stack), parent_id=self.nid(stack),
            label="Temporal layers" if causal else "Spatial layers", instances=instances,
        ))
        return x
