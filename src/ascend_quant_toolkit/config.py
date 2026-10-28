"""Stable, dependency-free data contracts used by the quantization plugin."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PrecisionGroup:
    """One declared precision group inside a possibly mixed-precision artifact.

    ``scope`` is documentation for humans; it is not matched against tensor
    names. The verifier compares ``producer_quant_type`` against the artifact's
    own quantization description.
    """

    scope: str
    producer_quant_type: str
    weight_dtype: str
    activation_dtype: str

    def as_dict(self) -> dict[str, str]:
        return {
            "scope": self.scope,
            "producer_quant_type": self.producer_quant_type,
            "weight_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
        }


@dataclass(frozen=True)
class Recipe:
    """Description of one reproducible offline quantization recipe.

    The first ten fields describe a single-scheme artifact. Recipes that emit
    several quantization types in one checkpoint additionally populate
    ``precision_groups`` and set ``mixed_precision``; such artifacts can be
    produced and verified, but they are not expressible in artifact contract
    1.1.0, so runtime metadata migration and contract export refuse them.
    """

    name: str
    description: str
    model_type: str
    config_resource: str
    producer_quant_type: str
    runtime_quant_type: str
    calibration_dataset: str
    weight_dtype: str
    activation_dtype: str
    quant_scheme: str
    precision_groups: tuple[PrecisionGroup, ...] = ()
    expected_float_patterns: tuple[str, ...] = ()
    expected_shards: int | None = None
    expected_weight_bytes: int | None = None
    forbid_w8a16: bool = False
    require_fused_linear_attn: bool = False
    mixed_precision: bool = False

    def config_path(self) -> Path:
        resource = files("ascend_quant_toolkit").joinpath(self.config_resource)
        return Path(str(resource))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type,
            "config_resource": self.config_resource,
            "producer_quant_type": self.producer_quant_type,
            "runtime_quant_type": self.runtime_quant_type,
            "calibration_dataset": self.calibration_dataset,
            "weight_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "quant_scheme": self.quant_scheme,
            "precision_groups": [group.as_dict() for group in self.precision_groups],
            "expected_float_patterns": list(self.expected_float_patterns),
            "expected_shards": self.expected_shards,
            "expected_weight_bytes": self.expected_weight_bytes,
            "forbid_w8a16": self.forbid_w8a16,
            "require_fused_linear_attn": self.require_fused_linear_attn,
            "mixed_precision": self.mixed_precision,
        }
