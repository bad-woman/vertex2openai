"""Defensive validation for client-supplied JSON Schemas."""

import json
from typing import Any


MAX_SCHEMA_DEPTH = 32
MAX_SCHEMA_NODES = 10_000
MAX_SCHEMA_SERIALIZED_BYTES = 1_000_000
_JSON_SCHEMA_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_SCHEMA_MAP_KEYS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
_SCHEMA_SINGLE_KEYS = {"items", "additionalProperties", "not", "contains", "propertyNames", "if", "then", "else"}
_SCHEMA_LIST_KEYS = {"oneOf", "anyOf", "allOf", "prefixItems"}


class SchemaValidationError(ValueError):
    """A client schema is malformed or exceeds defensive complexity limits."""


def validate_json_schema(schema: Any, *, label: str = "schema") -> None:
    if not isinstance(schema, dict):
        raise SchemaValidationError(f"{label} must be a JSON object")
    try:
        serialized = json.dumps(schema, ensure_ascii=False, separators=(",", ":"), check_circular=True)
    except (TypeError, ValueError, RecursionError) as exc:
        raise SchemaValidationError(f"{label} is not a valid JSON Schema object: {exc}") from None
    if len(serialized.encode("utf-8")) > MAX_SCHEMA_SERIALIZED_BYTES:
        raise SchemaValidationError(
            f"{label} exceeds the {MAX_SCHEMA_SERIALIZED_BYTES}-byte serialized-size limit")

    nodes = 0
    active: set[int] = set()

    def visit(node: Any, depth: int, path: str) -> None:
        nonlocal nodes
        if depth > MAX_SCHEMA_DEPTH:
            raise SchemaValidationError(f"{label} exceeds maximum depth {MAX_SCHEMA_DEPTH} at {path}")
        nodes += 1
        if nodes > MAX_SCHEMA_NODES:
            raise SchemaValidationError(f"{label} exceeds maximum node count {MAX_SCHEMA_NODES}")
        if not isinstance(node, dict):
            raise SchemaValidationError(f"{path} must be a JSON object")
        ident = id(node)
        if ident in active:
            raise SchemaValidationError(f"{label} contains a circular reference at {path}")
        active.add(ident)
        try:
            raw_type = node.get("type")
            types = raw_type if isinstance(raw_type, list) else [raw_type]
            if raw_type is not None:
                if not types or any(not isinstance(t, str) or t.lower() not in _JSON_SCHEMA_TYPES for t in types):
                    raise SchemaValidationError(f"{path}.type contains an unsupported JSON Schema type")
            required = node.get("required")
            if required is not None and (not isinstance(required, list)
                                         or any(not isinstance(name, str) for name in required)):
                raise SchemaValidationError(f"{path}.required must be an array of strings")

            for key in _SCHEMA_MAP_KEYS:
                value = node.get(key)
                if value is None:
                    continue
                if not isinstance(value, dict):
                    raise SchemaValidationError(f"{path}.{key} must be an object")
                for name, child in value.items():
                    if not isinstance(name, str):
                        raise SchemaValidationError(f"{path}.{key} keys must be strings")
                    visit(child, depth + 1, f"{path}.{key}.{name}")

            for key in _SCHEMA_SINGLE_KEYS:
                value = node.get(key)
                if value is None or isinstance(value, bool):
                    continue
                if key == "items" and isinstance(value, list):
                    for index, child in enumerate(value):
                        visit(child, depth + 1, f"{path}.{key}[{index}]")
                else:
                    visit(value, depth + 1, f"{path}.{key}")

            for key in _SCHEMA_LIST_KEYS:
                value = node.get(key)
                if value is None:
                    continue
                if not isinstance(value, list):
                    raise SchemaValidationError(f"{path}.{key} must be an array")
                for index, child in enumerate(value):
                    visit(child, depth + 1, f"{path}.{key}[{index}]")
        finally:
            active.remove(ident)

    visit(schema, 0, label)


def validate_request_schemas(request: Any) -> None:
    for index, tool in enumerate(getattr(request, "tools", None) or []):
        if not isinstance(tool, dict) or (tool.get("type") != "function" and "function" not in tool):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            raise SchemaValidationError(f"tools[{index}].function must be an object")
        parameters = function.get("parameters")
        if parameters is not None:
            validate_json_schema(parameters, label=f"tools[{index}].function.parameters")

    response_format = getattr(request, "response_format", None)
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            raise SchemaValidationError("response_format.json_schema must be an object")
        schema = json_schema.get("schema")
        validate_json_schema(schema, label="response_format.json_schema.schema")
