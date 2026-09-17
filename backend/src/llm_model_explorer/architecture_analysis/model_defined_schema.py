"""Portable, data-only architecture definition schema owned by checkpoint authors."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from . import records as r

MAX_DEFINITION_BYTES = 8 * 1024 * 1024


class DefinitionAttribute(r.Record):
    name: r.ArchitectureName
    value: str | float | bool | None | list[str | float | bool | None]


class DefinitionNodeFields(r.Record):
    id: r.ArchitectureId
    label: r.ArchitectureName
    ports: list[r.ArchitecturePort] = Field(default_factory=list)
    parameter_ids: list[r.ArchitectureId] = Field(default_factory=list)
    references: list[r.ArchitectureReference] = Field(default_factory=list)
    attributes: list[DefinitionAttribute] = Field(default_factory=list)
    parent_id: r.ArchitectureId | None = None
    operation: r.ArchitectureName | None = None
    description: r.ArchitectureText | None = None
    formula: r.ArchitectureText | None = None


class DefinitionLeaf(DefinitionNodeFields):
    kind: Literal["operation", "input", "output", "context", "state"]

    @model_validator(mode="after")
    def named_operation(self) -> DefinitionLeaf:
        if self.kind == "operation" and self.operation is None:
            raise ValueError("operation nodes require an operation name")
        return self


class DefinitionGroup(DefinitionNodeFields):
    kind: Literal["group"]
    children: list[r.ArchitectureId]


DefinitionNode = Annotated[DefinitionLeaf | DefinitionGroup, Field(discriminator="kind")]


class DefinitionEdge(r.Record):
    id: r.ArchitectureId
    source: r.ArchitectureEndpoint
    target: r.ArchitectureEndpoint
    kind: Literal["data", "state", "context"] = "data"
    label: r.ArchitectureName | None = None


class DefinitionParameter(r.Record):
    id: r.ArchitectureId
    name: r.ArchitectureName
    shape: r.ArchitectureShape


class ModelDefinition(r.Record):
    """Version 1 uses explicit instances, local IDs and complete native tensor names.

    The file cannot supply runtime tensor IDs, storage descriptors, verified
    provenance, executable expressions, includes, or schema references to fetch.
    """

    schema_version: Literal[1]
    architecture_revision: r.ArchitectureName
    name: r.ArchitectureName
    scope: r.ArchitectureName
    coverage: Literal["complete", "partial"] = "complete"
    incomplete_reason: r.ArchitectureText | None = None
    symbols: list[r.ArchitectureSymbol] = Field(default_factory=list)
    nodes: Annotated[list[DefinitionNode], Field(min_length=1)]
    edges: list[DefinitionEdge] = Field(default_factory=list)
    parameters: list[DefinitionParameter] = Field(default_factory=list)
    repetitions: list[r.ArchitectureRepetition] = Field(default_factory=list)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def explicit_incompleteness(self) -> ModelDefinition:
        if self.coverage == "partial" and not self.incomplete_reason:
            raise ValueError("partial definitions require an incomplete_reason")
        if self.coverage == "complete" and self.incomplete_reason is not None:
            raise ValueError("complete definitions cannot declare incompleteness")
        return self


def definition_schema() -> dict[str, Any]:
    """Export an offline JSON Schema; mirror Record's absent-not-null optionals."""
    schema = ModelDefinition.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "Model-owned architecture definition v1"
    definitions = [schema, *schema.get("$defs", {}).values()]
    for record in definitions:
        required = record.get("required", [])
        for name, field in record.get("properties", {}).items():
            if name in required:
                continue
            alternatives = field.get("anyOf", [])
            without_null = [item for item in alternatives if item != {"type": "null"}]
            if len(without_null) != len(alternatives):
                field["anyOf"] = without_null
                field.pop("default", None)
    schema["x-max-decoded-bytes"] = MAX_DEFINITION_BYTES
    schema["$defs"]["DefinitionLeaf"]["allOf"] = [
        {
            "if": {"properties": {"kind": {"const": "operation"}}, "required": ["kind"]},
            "then": {"required": ["operation"]},
        }
    ]
    schema["allOf"] = [
        {
            "if": {"properties": {"coverage": {"const": "partial"}}, "required": ["coverage"]},
            "then": {"required": ["incomplete_reason"]},
            "else": {"not": {"required": ["incomplete_reason"]}},
        }
    ]
    return schema


if __name__ == "__main__":
    print(json.dumps(definition_schema(), indent=2, sort_keys=True))
