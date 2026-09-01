"""Fail-closed validation for offline-produced Ascend quant artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "vllm-hust.ascend-quant-artifact/v1"
REQUIRED_FIELDS = (
    "schema_version",
    "weight_packing",
    "weight_signedness",
    "weight_scale_granularity",
    "activation_scale_granularity",
    "zero_point_semantics",
    "supported_shapes",
    "operator_name",
)


class ArtifactContractError(ValueError):
    """The artifact cannot be safely consumed by the runtime."""


class ArtifactContractValidator:
    """Validate metadata only; this class does not load models or devices."""

    @staticmethod
    def validate(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ArtifactContractError("artifact contract must be an object")
        unexpected = sorted(set(value) - set(REQUIRED_FIELDS))
        if unexpected:
            raise ArtifactContractError(
                "artifact contract contains unsupported fields: "
                + ", ".join(unexpected)
            )
        missing = [field for field in REQUIRED_FIELDS if not value.get(field)]
        if missing:
            raise ArtifactContractError(
                "artifact contract is incomplete: " + ", ".join(missing)
            )
        if value["schema_version"] != SCHEMA_VERSION:
            raise ArtifactContractError(
                f"unsupported artifact schema: {value['schema_version']!r}"
            )
        scalar_fields = set(REQUIRED_FIELDS) - {"supported_shapes"}
        malformed = sorted(
            field
            for field in scalar_fields
            if not isinstance(value[field], str) or not value[field].strip()
        )
        if malformed:
            raise ArtifactContractError(
                "artifact contract fields must be non-empty strings: "
                + ", ".join(malformed)
            )
        shapes = value["supported_shapes"]
        if not isinstance(shapes, list) or not all(
            isinstance(shape, str) and shape for shape in shapes
        ):
            raise ArtifactContractError(
                "supported_shapes must be a non-empty string array"
            )
        if len(shapes) != len(set(shapes)):
            raise ArtifactContractError("supported_shapes must not contain duplicates")
        return dict(value)
