"""Static graph construction and explicit registration of trusted packaged descriptions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol

from . import records as r
from .templates import ComponentTemplates
from .validation import (
    MAX_BYTES,
    BindingContext,
    GraphError,
    NumericTensor,
    constants,
    json_chunks,
    preflight,
    require,
    serialized_size,
    validate_graph,
)

if TYPE_CHECKING:
    from ..tensor_source import PeftLoraComposition, PhysicalTensor, TensorDescriptor

ANALYZER_REVISION = "static-graph-core-3"
Scope = Literal["language_model", "visual_encoder_predictor"]


class MetadataSource(Protocol):
    """Satisfied by ModelSource. No path, tensor reader, or execution capability is passed on."""

    @property
    def fingerprint(self) -> str: ...
    def configuration(self) -> dict[str, object]: ...
    def physical_tensors(self) -> tuple[PhysicalTensor, ...]: ...
    def tensors(self) -> tuple[TensorDescriptor, ...]: ...
    def check_unchanged(self, *, rehash: bool = False) -> None: ...

    lora_composition: PeftLoraComposition | None


@dataclass(frozen=True)
class AnalysisInput:
    fingerprint: str
    configuration: Mapping[str, object]
    bindings: BindingContext
    lora_composition: PeftLoraComposition | None = None

    @classmethod
    def from_source(cls, source: MetadataSource, *, tokenizer_available: bool) -> AnalysisInput:
        source.check_unchanged()
        config = source.configuration()
        physical = {
            tensor.name: r.ArchitectureStorage(
                name=tensor.name, dtype=tensor.dtype, shape=list(tensor.shape)
            )
            for tensor in source.physical_tensors()
        }
        numeric = {
            tensor.id: NumericTensor(
                tensor.id, tensor.name, tensor.shape, tensor.storage_dtype, tensor.storage_format
            )
            for tensor in source.tensors()
        }
        source.check_unchanged()
        return cls(
            source.fingerprint,
            MappingProxyType(config),
            BindingContext(
                physical=MappingProxyType(physical),
                numeric=MappingProxyType(numeric),
                tokenizer_available=tokenizer_available,
                adapter_tensor_storage=MappingProxyType(
                    {
                        name: storage
                        for target in (
                            source.lora_composition.targets if source.lora_composition else ()
                        )
                        for name, storage in (
                            (target.a_tensor_name, target.a_storage_name),
                            (target.b_tensor_name, target.b_storage_name),
                        )
                    }
                ),
            ),
            source.lora_composition,
        )


def identity(namespace: str, *parts: str) -> str:
    # Length-delimited JSON avoids collisions between differently split semantic keys.
    raw = json.dumps([namespace, *parts], ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Producer:
    description: str
    revision: str
    source_revision: str
    analyzer_revision: str = ANALYZER_REVISION
    schema_revision: str = r.SCHEMA_REVISION

    def graph_id(self, fingerprint: str, scope: Scope) -> str:
        return identity(
            "architecture-graph",
            fingerprint,
            scope,
            self.description,
            self.revision,
            self.source_revision,
            self.analyzer_revision,
            self.schema_revision,
        )

    def provenance(self) -> list[r.ArchitectureProvenance]:
        return [
            r.ArchitectureProvenance(
                kind="description",
                source=self.description,
                revision=self.revision,
                rule=f"Reviewed implementation revision: {self.source_revision}",
            )
        ]


@dataclass(frozen=True)
class AnalysisResult:
    """Domain result; HTTP/model/session/cache envelopes belong to their own consumers."""

    graph: r.ArchitectureGraph | None
    reason: Literal["unsupported_architecture", "analysis_failed", "unsupported_size"] | None
    diagnostics: tuple[r.ArchitectureDiagnostic, ...] = ()

    @property
    def status(self) -> Literal["complete", "partial", "unavailable"]:
        return self.graph.coverage if self.graph is not None else "unavailable"


def unavailable(
    reason: Literal["unsupported_architecture", "analysis_failed", "unsupported_size"],
    code: str,
    message: str,
) -> AnalysisResult:
    return AnalysisResult(None, reason, (r.ArchitectureDiagnostic(code=code, message=message),))


class GraphBuilder:
    """Append full records under an incremental byte budget, then validate atomically.

    Keys identify semantic instances, never their insertion order or display labels.
    The caller supplies a reduced budget if its response envelope needs reserved bytes.
    """

    def __init__(
        self,
        inputs: AnalysisInput,
        producer: Producer,
        scope: Scope,
        *,
        byte_limit: int = MAX_BYTES,
    ) -> None:
        require(0 < byte_limit <= MAX_BYTES, "Invalid architecture byte budget.")
        self.inputs = inputs
        self.producer = producer
        self.scope = scope
        self.graph_id = producer.graph_id(inputs.fingerprint, scope)
        self.byte_limit = byte_limit
        self._numeric_by_name = {t.name: t for t in inputs.bindings.numeric.values()}
        self._used = 0
        self._ids: set[str] = set()
        self._nodes: list[r.ArchitectureNode] = []
        self._edges: list[r.ArchitectureEdge] = []
        self._parameters: list[r.ArchitectureParameter] = []
        self._repetitions: list[r.ArchitectureRepetition] = []
        self._symbols: list[r.ArchitectureSymbol] = []
        self._diagnostics: list[r.ArchitectureDiagnostic] = []
        self._partial = False
        self.templates = ComponentTemplates()
        self._parameter_by_id: dict[str, r.ArchitectureParameter] = {}

    def record_id(self, kind: str, key: str) -> str:
        return identity("architecture-record", self.graph_id, kind, key)

    def _append(self, collection: list[Any], record: r.Record) -> None:
        # Round-trip catches mutated nested lists and excludes caller ownership after insertion.
        preflight(record, self.byte_limit - self._used)
        size = serialized_size(record.document(), self.byte_limit - self._used)
        copy = type(record).model_validate(record.document())
        key = getattr(copy, "id", None)
        require(key is None or key not in self._ids, "Duplicate record identity.")
        self._used += size
        if key is not None:
            self._ids.add(key)
        collection.append(copy)

    def begin_template(
        self, key: str, base: str, family: str, role: Literal["attention", "mlp"]
    ) -> None:
        self.templates.begin(self.record_id("node", key), base, family, role)

    def add_node(self, node: r.ArchitectureNode, *, semantic_key: str | None = None) -> str:
        self._append(self._nodes, node)
        self.templates.observe(node, semantic_key, self._parameter_by_id)
        return node.id

    def add_edge(self, edge: r.ArchitectureEdge) -> str:
        self._append(self._edges, edge)
        return edge.id

    def add_parameter(self, parameter: r.ArchitectureParameter) -> str:
        self._append(self._parameters, parameter)
        self._parameter_by_id[parameter.id] = self._parameters[-1]
        return parameter.id

    def add_repetition(self, repetition: r.ArchitectureRepetition) -> str:
        self._append(self._repetitions, repetition)
        return repetition.id

    def add_symbol(self, name: str, meaning: str) -> None:
        require(all(s.name != name for s in self._symbols), "Duplicate shape symbol.")
        self._append(self._symbols, r.ArchitectureSymbol(name=name, meaning=meaning))

    def diagnose(self, diagnostic: r.ArchitectureDiagnostic, *, partial: bool = True) -> None:
        self._append(self._diagnostics, diagnostic)
        self._partial |= partial

    def unknown(
        self,
        key: str,
        label: str,
        reason: str,
        *,
        parent_id: str | None = None,
        ports: Sequence[r.ArchitecturePort] = (),
    ) -> str:
        """An explicitly opaque region, with no guessed operation or implicit edges."""
        args: dict[str, Any] = {} if parent_id is None else {"parent_id": parent_id}
        node_id = self.record_id("node", key)
        self.add_node(
            r.ArchitectureLeafNode(
                id=node_id,
                kind="context",
                label=label,
                ports=list(ports),
                parameter_ids=[],
                references=[],
                attributes=[],
                provenance=self.producer.provenance(),
                **args,
            )
        )
        self.diagnose(
            r.ArchitectureDiagnostic(code="unknown_region", message=reason, node_id=node_id)
        )
        return node_id

    def native_parameter(
        self,
        key: str,
        name: str,
        logical_shape: r.ArchitectureShape,
        provenance: Sequence[r.ArchitectureProvenance],
    ) -> str:
        """Bind only exact native geometry; absent storage remains explicitly unresolved."""
        storage = self.inputs.bindings.physical.get(name)
        inspection: r.ArchitectureInspection
        binding: Literal["native", "unresolved"] = "native" if storage is not None else "unresolved"
        geometry = constants(logical_shape)
        if storage is None:
            inspection = r.ArchitectureUnavailableInspection(
                status="unavailable",
                reason="unresolved_binding",
                message="Parameter storage is absent.",
            )
        else:
            require(
                geometry is not None and list(geometry) == storage.shape,
                "Native logical and physical geometry disagree.",
            )
            tensor = self._numeric_by_name.get(name)
            require(
                tensor is None or (tensor.shape == geometry and tensor.dtype == storage.dtype),
                "Numeric and physical inventory geometry disagree.",
            )
            if len(storage.shape) not in (1, 2):
                inspection = r.ArchitectureUnavailableInspection(
                    status="unavailable",
                    reason="unsupported_rank",
                    message="Numeric inspection supports complete rank-1/rank-2 tensors.",
                )
            elif tensor is None:
                inspection = r.ArchitectureUnavailableInspection(
                    status="unavailable",
                    reason="unsupported_representation",
                    message="No admitted complete native numeric view exists.",
                )
            else:
                inspection = r.ArchitectureAvailableInspection(
                    status="available", tensor_id=tensor.id
                )
        parameter_id = self.record_id("parameter", key)
        self.add_parameter(
            r.ArchitectureDirectParameter(
                id=parameter_id,
                name=name,
                logical_shape=logical_shape,
                binding=binding,
                storage=[] if storage is None else [storage],
                inspection=inspection,
                provenance=list(provenance),
            )
        )
        if storage is None:
            self.diagnose(
                r.ArchitectureDiagnostic(
                    code="unresolved_binding",
                    message="Required parameter storage is absent.",
                    parameter_id=parameter_id,
                )
            )
        return parameter_id

    def adapter_parameter(
        self,
        key: str,
        logical_name: str,
        storage_name: str,
        logical_shape: r.ArchitectureShape,
        provenance: Sequence[r.ArchitectureProvenance],
    ) -> str:
        """Bind a composite logical factor ID to its verified native adapter storage."""
        storage = self.inputs.bindings.physical.get(storage_name)
        matches = [t for t in self.inputs.bindings.numeric.values() if t.name == logical_name]
        if storage is None or len(matches) != 1:
            raise GraphError("lora_missing_factor", "A required LoRA factor binding is missing.")
        tensor = matches[0]
        geometry = constants(logical_shape)
        if (
            self.inputs.bindings.adapter_tensor_storage.get(logical_name) != storage_name
            or geometry is None
            or tuple(storage.shape) != geometry
            or tensor.shape != geometry
            or tensor.dtype != storage.dtype
            or storage.dtype not in {"F32", "F16", "BF16"}
        ):
            raise GraphError(
                "lora_factor_geometry", "LoRA factor storage disagrees with its logical tensor."
            )
        parameter_id = self.record_id("parameter", key)
        self.add_parameter(
            r.ArchitectureDirectParameter(
                id=parameter_id,
                name=logical_name,
                logical_shape=logical_shape,
                binding="native",
                storage=[
                    r.ArchitectureStorage(
                        name=storage.name,
                        dtype=storage.dtype,
                        shape=list(storage.shape),
                        role="adapter_factor",
                    )
                ],
                inspection=r.ArchitectureAvailableInspection(
                    status="available", tensor_id=tensor.id
                ),
                provenance=list(provenance)
                + [r.ArchitectureProvenance(kind="storage", source=storage.name)],
            )
        )
        return parameter_id

    def finish(self) -> r.ArchitectureGraph:
        graph = r.ArchitectureGraph(
            graph_id=self.graph_id,
            scope=self.scope,
            coverage="partial" if self._partial else "complete",
            symbols=self._symbols,
            nodes=self._nodes,
            edges=self._edges,
            repetitions=self._repetitions,
            parameters=self._parameters,
            diagnostics=self._diagnostics,
        )
        serialized_size(graph.document(), self.byte_limit)
        validate_graph(graph, self.inputs.bindings)
        graph = self.templates.annotate(graph, self)
        serialized_size(graph.document(), self.byte_limit)
        validate_graph(graph, self.inputs.bindings)
        return graph


def parse_graph(
    document: object, context: BindingContext, *, byte_limit: int = MAX_BYTES
) -> r.ArchitectureGraph:
    """Validate a pre-read document at the metadata boundary; never echo parser errors."""
    try:
        serialized_size(document, byte_limit)
        graph = r.ArchitectureGraph.model_validate(document)
        validate_graph(graph, context)
        return graph
    except GraphError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise GraphError("invalid_graph", "Malformed architecture records.") from exc


def serialize_graph(graph: r.ArchitectureGraph, *, byte_limit: int = MAX_BYTES) -> bytes:
    preflight(graph, byte_limit)
    return b"".join(json_chunks(graph.document(), byte_limit))


@dataclass(frozen=True)
class Description:
    producer: Producer
    scope: Scope
    model_types: frozenset[str]
    architectures: frozenset[str]
    # The reviewed selector must check relevant options, layer variants and storage.
    # Returning False leaves the checkpoint unsupported; it must not guess a fallback.
    supports: Callable[[AnalysisInput], bool]
    build: Callable[[AnalysisInput, GraphBuilder], None]


class DescriptionRegistry:
    def __init__(self) -> None:
        self._descriptions: dict[str, Description] = {}

    def register(self, description: Description) -> None:
        name = description.producer.description
        require(name not in self._descriptions, "Duplicate packaged description.")
        require(
            bool(description.model_types) and bool(description.architectures),
            "Description requires explicit configuration discriminators.",
        )
        # Validate provenance limits before accepting a producer.
        description.producer.provenance()
        self._descriptions[name] = description

    def select(self, inputs: AnalysisInput) -> Description | None:
        """Resolve the semantic producer before graph construction or cache lookup."""
        model_type = inputs.configuration.get("model_type")
        architectures = inputs.configuration.get("architectures")
        candidates = []
        for description in self._descriptions.values():
            if (
                isinstance(model_type, str)
                and model_type in description.model_types
                and isinstance(architectures, list)
                and architectures
                and all(
                    isinstance(a, str) and a in description.architectures for a in architectures
                )
                and description.supports(inputs)
            ):
                candidates.append(description)
        return candidates[0] if len(candidates) == 1 else None

    def analyze(self, inputs: AnalysisInput, *, byte_limit: int = MAX_BYTES) -> AnalysisResult:
        try:
            selected = self.select(inputs)
            if selected is None:
                return unavailable(
                    "unsupported_architecture",
                    "description_selection",
                    "No unique verified architecture description matches this metadata.",
                )
            builder = GraphBuilder(inputs, selected.producer, selected.scope, byte_limit=byte_limit)
            selected.build(inputs, builder)
            graph = builder.finish()
            return AnalysisResult(graph, None)
        except GraphError as exc:
            if exc.code.startswith("lora_"):
                return unavailable("analysis_failed", exc.code, str(exc))
            reason: Literal["unsupported_size", "analysis_failed"] = (
                "unsupported_size" if exc.code == "unsupported_size" else "analysis_failed"
            )
            return unavailable(
                reason,
                reason if reason == "unsupported_size" else "invalid_graph",
                "Architecture exceeds the supported size."
                if reason == "unsupported_size"
                else "Architecture validation failed.",
            )
        except Exception:
            # A broken packaged description must not leak a traceback or checkpoint text.
            return unavailable(
                "analysis_failed",
                "description_failed",
                "The packaged architecture description could not produce a valid graph.",
            )
