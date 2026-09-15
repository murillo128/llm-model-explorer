"""Strict Python representations of the accepted LMEX control schemas."""

from typing import Annotated, Literal, Self

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


Metadata = (
    TensorMetadata
    | InputEmbeddingsMetadata
    | TensorStatisticsMetadata
    | TensorDistributionsMetadata
    | InputEmbeddingsStatisticsMetadata
    | InputEmbeddingsDistributionsMetadata
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
