"""Strict Python representations of the accepted LMEX control schemas."""

import re
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

SAFE_INTEGER = 2**53 - 1
Size = Annotated[int, Field(ge=0, le=SAFE_INTEGER)]
Text = Annotated[str, Field(min_length=1)]


def _product(shape: list[int]) -> int:
    # A later zero makes the product zero even after very large dimensions.
    if 0 in shape:
        return 0
    result = 1
    for dimension in shape:
        result *= dimension
        if result > SAFE_INTEGER:
            raise ValueError("unsafe shape product")
    return result


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class TensorMetadata(Control):
    kind: Literal["tensor"]
    tensor_id: Text
    name: Text
    shape: list[Size]
    dtype: Literal["float32"]
    byte_order: Literal["little"]
    layout: Literal["c"]
    byte_length: Size

    @model_validator(mode="after")
    def geometry(self) -> Self:
        if self.byte_length != 4 * _product(self.shape):
            raise ValueError("tensor byte length disagrees with shape")
        return self


class Percentiles(Control):
    p01: float | None
    p05: float | None
    p50: float | None
    p95: float | None
    p99: float | None


class StatisticsFields(Control):
    count: Size
    finite_count: Size
    non_finite_count: Size
    minimum: float | None
    maximum: float | None
    mean: float | None
    stddev: Annotated[float, Field(ge=0)] | None
    percentiles: Percentiles
    byte_length: Annotated[Size, Field(le=0)]

    @model_validator(mode="after")
    def statistics(self) -> Self:
        if self.count != self.finite_count + self.non_finite_count:
            raise ValueError("statistics count sum")
        values = [self.minimum, self.maximum, self.mean, self.stddev]
        quantiles = list(self.percentiles.model_dump().values())
        if self.finite_count == 0:
            if any(v is not None for v in [*values, *quantiles]):
                raise ValueError("empty statistics must use null scalars")
        else:
            if any(v is None for v in [*values, *quantiles]):
                raise ValueError("finite statistics must use numeric scalars")
            assert self.minimum is not None and self.maximum is not None and self.mean is not None
            ordered = [self.minimum, *quantiles, self.maximum]
            if ordered != sorted(ordered) or not self.minimum <= self.mean <= self.maximum:
                raise ValueError("statistics range or percentile order")
        return self


class TensorStatisticsMetadata(StatisticsFields):
    kind: Literal["tensor_statistics"]
    tensor_id: Text


class DistributionSection(Control):
    name: Literal["row_counts", "column_counts"]
    shape: Annotated[list[Size], Field(min_length=2, max_length=2)]
    offset: Size
    byte_length: Size


class DistributionsFields(Control):
    rows: Size
    columns: Size
    bin_count: Annotated[Size, Field(ge=100, le=100)]
    binning: Literal["linear-full-range"]
    domain_minimum: float | None
    domain_maximum: float | None
    dtype: Literal["uint32"]
    byte_order: Literal["little"]
    sections: Annotated[list[DistributionSection], Field(min_length=2, max_length=2)]
    byte_length: Size

    @model_validator(mode="after")
    def geometry(self) -> Self:
        if self.rows * self.columns > SAFE_INTEGER:
            raise ValueError("unsafe source size")
        offset = 0
        for section, name, shape in zip(
            self.sections,
            ("row_counts", "column_counts"),
            ([self.rows, 100], [100, self.columns]),
            strict=True,
        ):
            length = 4 * _product(shape)
            if (
                section.name != name
                or section.shape != shape
                or section.offset != offset
                or section.byte_length != length
            ):
                raise ValueError("distribution section geometry")
            offset += length
        if offset != self.byte_length:
            raise ValueError("distribution section total")
        lo, hi = self.domain_minimum, self.domain_maximum
        if (lo is None) != (hi is None) or (lo is not None and hi is not None and lo > hi):
            raise ValueError("distribution domain")
        return self


class TensorDistributionsMetadata(DistributionsFields):
    kind: Literal["tensor_distributions"]
    tensor_id: Text


class InputEmbeddingsMetadata(Control):
    kind: Literal["input_embeddings"]
    token_ids: list[Size]
    shape: Annotated[list[Size], Field(min_length=2, max_length=2)]
    dtype: Literal["float32"]
    byte_order: Literal["little"]
    layout: Literal["c"]
    byte_length: Size

    @model_validator(mode="after")
    def geometry(self) -> Self:
        if (
            self.shape[0] != len(self.token_ids)
            or self.shape[1] == 0
            or self.byte_length != 4 * _product(self.shape)
        ):
            raise ValueError("input embedding geometry")
        return self


class InputEmbeddingsStatisticsMetadata(StatisticsFields):
    kind: Literal["input_embeddings_statistics"]
    token_ids: list[Size]
    shape: Annotated[list[Size], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def embedding_geometry(self) -> Self:
        if (
            self.shape[0] != len(self.token_ids)
            or self.shape[1] == 0
            or self.count != _product(self.shape)
        ):
            raise ValueError("input embedding statistics geometry")
        return self


class InputEmbeddingsDistributionsMetadata(DistributionsFields):
    kind: Literal["input_embeddings_distributions"]
    token_ids: list[Size]

    @model_validator(mode="after")
    def embedding_geometry(self) -> Self:
        if self.rows != len(self.token_ids) or self.columns == 0:
            raise ValueError("input embedding distribution geometry")
        if self.rows == 0 and self.domain_minimum is not None:
            raise ValueError("empty input embeddings require a null domain")
        return self


NeuronId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._~-]+$")]
ModelRevision = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
PositiveSize = Annotated[Size, Field(ge=1)]


class NeuronExecution(Control):
    family: Literal["smollm2"]
    adapter: Literal["smollm2_native"]
    adapter_version: Annotated[Text, Field(max_length=128)]
    runtime_version: Annotated[Text, Field(max_length=128)]
    torch_version: Annotated[Text, Field(max_length=128)]
    device: Annotated[str, Field(pattern=r"^(cpu|cuda:[0-9]+)$")]
    dtype: Literal["float32"]


class NeuronLimits(Control):
    max_modules_per_request: Literal[1]
    max_neurons: Literal[8]
    max_input_tokens: Annotated[PositiveSize, Field(le=512)]
    max_new_tokens: Annotated[Size, Field(le=64)]
    max_conditions: Literal[4]
    max_top_k: Annotated[PositiveSize, Field(le=20)]
    context_length: PositiveSize
    vocabulary_size: PositiveSize
    eos_token_ids: Annotated[list[Size], Field(max_length=64)]

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if (
            self.max_input_tokens != min(512, self.context_length)
            or self.max_new_tokens != min(64, self.context_length - 1)
            or self.max_top_k != min(20, self.vocabulary_size)
            or self.eos_token_ids != sorted(set(self.eos_token_ids))
            or any(i >= self.vocabulary_size for i in self.eos_token_ids)
        ):
            raise ValueError("neuron limits")
        return self


class NeuronWeightBinding(Control):
    tensor_id: Annotated[Text, Field(max_length=256)]
    axis: Literal["row", "column"]
    shape: Annotated[list[Size], Field(min_length=2, max_length=2)]


class NeuronWeights(Control):
    gate: NeuronWeightBinding
    up: NeuronWeightBinding
    down: NeuronWeightBinding


class NeuronModuleReference(Control):
    kind: Literal["module"]
    name: Annotated[Text, Field(max_length=1024)]


class NeuronGraphBinding(Control):
    graph_id: Annotated[
        Text, Field(max_length=256, pattern=re.compile(r"^(?!\.{1,2}$)[A-Za-z0-9._~-]+$"))
    ]
    node_id: Annotated[
        Text, Field(max_length=256, pattern=re.compile(r"^(?!\.{1,2}$)[A-Za-z0-9._~-]+$"))
    ]


class NeuronModule(Control):
    module_id: NeuronId
    module: NeuronModuleReference
    layer_index: Annotated[Size, Field(le=63)]
    intermediate_size: PositiveSize
    hidden_size: PositiveSize
    point: Literal["post_activation_product_pre_down_proj"]
    weights: NeuronWeights
    graph_binding: NeuronGraphBinding | None = None

    @model_validator(mode="after")
    def bindings(self) -> Self:
        if "graph_binding" in self.model_fields_set and self.graph_binding is None:
            raise ValueError("null graph binding")
        width, hidden = self.intermediate_size, self.hidden_size
        ids = []
        for role in ("gate", "up", "down"):
            binding = getattr(self.weights, role)
            if binding.axis != ("column" if role == "down" else "row") or binding.shape != (
                [hidden, width] if role == "down" else [width, hidden]
            ):
                raise ValueError("neuron weight geometry")
            ids.append(binding.tensor_id)
        if len(set(ids)) != 3 or width * hidden * 4 > SAFE_INTEGER:
            raise ValueError("neuron weight identity or size")
        return self


class NeuronSection(Control):
    name: Literal["activations", "probabilities"]
    shape: Annotated[list[Size], Field(min_length=2, max_length=2)]
    offset: Size
    byte_length: Size


class NeuronBaselineCondition(Control):
    id: Literal["baseline"]
    mode: Literal["baseline"]


class NeuronIntervention(Control):
    id: NeuronId
    scope: Literal["prompt", "continuation", "all"]

    @model_validator(mode="after")
    def identity(self) -> Self:
        if self.id == "baseline":
            raise ValueError("reserved baseline identity")
        return self


class NeuronZeroCondition(NeuronIntervention):
    mode: Literal["zero"]


Float32Parameter = Annotated[float, Field(ge=-3.4028234663852886e38, le=3.4028234663852886e38)]


class NeuronScaleCondition(NeuronIntervention):
    mode: Literal["scale"]
    factor: Float32Parameter


class NeuronAddCondition(NeuronIntervention):
    mode: Literal["add"]
    delta: Float32Parameter


NeuronCondition = Annotated[
    NeuronBaselineCondition | NeuronZeroCondition | NeuronScaleCondition | NeuronAddCondition,
    Field(discriminator="mode"),
]


class NeuronComparisonOptions(Control):
    conditions: list[NeuronCondition]
    top_k: PositiveSize
    max_new_tokens: Size
    decoding: Literal["greedy"]

    @model_validator(mode="after")
    def condition_order(self) -> Self:
        if (
            not self.conditions
            or self.conditions[0].mode != "baseline"
            or any(c.mode == "baseline" for c in self.conditions[1:])
            or len({c.id for c in self.conditions}) != len(self.conditions)
        ):
            raise ValueError("comparison requires one initial baseline and unique condition IDs")
        return self

    def check_limits(self, limits: NeuronLimits, prompt_length: int) -> None:
        if (
            len(self.conditions) > limits.max_conditions
            or self.top_k > limits.max_top_k
            or self.max_new_tokens > limits.max_new_tokens
            or prompt_length + self.max_new_tokens > limits.context_length
        ):
            raise ValueError("comparison exceeds effective limits")


class NeuronMetadata(Control):
    session_id: Text
    model_id: Text
    model_revision: ModelRevision
    execution: NeuronExecution
    limits: NeuronLimits
    module: NeuronModule
    token_ids: Annotated[list[Size], Field(min_length=1, max_length=512)]
    neuron_indices: Annotated[list[Size], Field(min_length=1, max_length=8)]
    dtype: Literal["float32"]
    byte_order: Literal["little"]
    layout: Literal["c"]
    sections: Annotated[list[NeuronSection], Field(min_length=1, max_length=1)]
    byte_length: Size

    @model_validator(mode="after")
    def selection(self) -> Self:
        UUID(self.session_id)
        if (
            len(self.token_ids) > self.limits.max_input_tokens
            or any(i >= self.limits.vocabulary_size for i in self.token_ids)
            or len(set(self.neuron_indices)) != len(self.neuron_indices)
            or any(i >= self.module.intermediate_size for i in self.neuron_indices)
        ):
            raise ValueError("neuron selection")
        return self

    def check_section(self, name: str, shape: list[int]) -> None:
        section = self.sections[0]
        if (
            section.name != name
            or section.shape != shape
            or section.offset != 0
            or section.byte_length != 4 * _product(shape)
            or section.byte_length != self.byte_length
        ):
            raise ValueError("neuron section geometry")


class NeuronActivationsMetadata(NeuronMetadata):
    kind: Literal["neuron_activations"]

    @model_validator(mode="after")
    def geometry(self) -> Self:
        self.check_section("activations", [len(self.token_ids), len(self.neuron_indices)])
        return self


class NeuronContinuation(Control):
    condition_id: NeuronId
    token_ids: Annotated[list[Size], Field(max_length=64)]
    text: Annotated[str, Field(max_length=1048576)]
    finish_reason: Literal["not_requested", "eos", "length"]


class NeuronComparisonMetadata(NeuronMetadata, NeuronComparisonOptions):
    kind: Literal["neuron_comparison"]
    candidate_token_ids: Annotated[list[Size], Field(min_length=1, max_length=80)]
    top_token_ids: Annotated[list[list[Size]], Field(min_length=1, max_length=4)]
    continuations: Annotated[list[NeuronContinuation], Field(min_length=1, max_length=4)]

    @model_validator(mode="after")
    def comparison(self) -> Self:
        self.check_limits(self.limits, len(self.token_ids))
        candidates, tops = self.candidate_token_ids, self.top_token_ids
        count = len(self.conditions)
        if (
            candidates != sorted(set(candidates))
            or any(i >= self.limits.vocabulary_size for i in candidates)
            or len(tops) != count
            or any(len(row) != self.top_k or len(set(row)) != self.top_k for row in tops)
            or candidates != sorted({i for row in tops for i in row})
            or len(self.continuations) != count
        ):
            raise ValueError("comparison candidates or condition correspondence")
        eos = self.limits.eos_token_ids
        for condition, continuation in zip(self.conditions, self.continuations, strict=True):
            ids, reason = continuation.token_ids, continuation.finish_reason
            if (
                continuation.condition_id != condition.id
                or len(ids) > self.max_new_tokens
                or any(i >= self.limits.vocabulary_size for i in ids)
            ):
                raise ValueError("continuation identity or bounds")
            if self.max_new_tokens == 0:
                if reason != "not_requested" or ids or continuation.text:
                    raise ValueError("generation not requested")
            elif reason == "eos":
                if not ids or ids[-1] not in eos or any(i in eos for i in ids[:-1]):
                    raise ValueError("EOS stopping")
            elif (
                reason != "length" or len(ids) != self.max_new_tokens or any(i in eos for i in ids)
            ):
                raise ValueError("length stopping")
        self.check_section("probabilities", [count, len(candidates)])
        return self


Metadata = (
    TensorMetadata
    | InputEmbeddingsMetadata
    | TensorStatisticsMetadata
    | TensorDistributionsMetadata
    | InputEmbeddingsStatisticsMetadata
    | InputEmbeddingsDistributionsMetadata
    | NeuronActivationsMetadata
    | NeuronComparisonMetadata
)
METADATA: TypeAdapter[Metadata] = TypeAdapter(Annotated[Metadata, Field(discriminator="kind")])


class StreamProgress(Control):
    completed: Size
    total: Annotated[int, Field(ge=1, le=SAFE_INTEGER)] | None = None
    unit: Text

    @model_validator(mode="after")
    def counters(self) -> Self:
        if "total" in self.model_fields_set and (self.total is None or self.completed > self.total):
            raise ValueError("progress total")
        return self


class StreamError(Control):
    code: Literal[
        "malformed_json",
        "model_not_found",
        "session_not_found",
        "tensor_not_found",
        "model_content_changed",
        "validation_error",
        "unsupported_representation",
        "unsupported_rank",
        "unsupported_size",
        "resource_exhausted",
        "internal_error",
    ]
    message: Text
    details: dict[str, object] = Field(default_factory=dict)
