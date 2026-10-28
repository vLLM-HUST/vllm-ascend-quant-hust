"""Fail-closed verification of a recipe's declared precision scope.

The toolkit can produce artifacts that mix several quantization types in one
checkpoint. Artifact contract 1.1.0 is single-scheme, so such artifacts cannot
be exported as a runtime contract; they can still be verified against the
recipe that produced them. This module implements that verification from the
artifact's own ``quant_model_description.json`` and the rollup file sizes.

Nothing here imports torch, msmodelslim, or vLLM: verification is offline and
runs on any machine that can read the artifact.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import Recipe

DESCRIPTION = "quant_model_description.json"
FLOAT = "FLOAT"
# Top-level metadata written by the saver; never tensor entries.
METADATA_KEYS = frozenset({"model_quant_type", "version", "metadata", "group_size", "optional"})
QUANT_PREFIXES = ("W", "INT", "FAK", "ASCEND_QUANT_")
FUSED_PROJECTIONS = ("in_proj_qkv", "in_proj_z")


def is_quant_type(value: Any) -> bool:
    """True for ``FLOAT`` and for any recognized quantized-type string."""

    return isinstance(value, str) and (value == FLOAT or value.startswith(QUANT_PREFIXES))


def tensor_types(description: Any) -> dict[str, str]:
    """Return tensor-name -> quantization type, rejecting unrecognized entries."""

    if not isinstance(description, dict):
        raise ValueError(f"{DESCRIPTION} must be a JSON object")
    tensors: dict[str, str] = {}
    for name, value in description.items():
        if name in METADATA_KEYS:
            continue
        if not is_quant_type(value):
            raise ValueError(f"Unrecognized quantization entry {name}={value!r}")
        tensors[name] = value
    return tensors


def non_float_types(description: Any) -> set[str]:
    """Quantization types actually applied to at least one tensor."""

    return {value for value in tensor_types(description).values() if value != FLOAT}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _check_declared_types(tensors: dict[str, str], recipe: Recipe) -> dict[str, Any]:
    declared = sorted({group.producer_quant_type for group in recipe.precision_groups})
    observed = sorted({value for value in tensors.values() if value != FLOAT})
    return _check(
        "declared_precision_types",
        observed == declared,
        f"declared={declared} observed={observed}",
    )


def _check_no_w8a16(tensors: dict[str, str]) -> dict[str, Any]:
    count = sum(1 for value in tensors.values() if value == "W8A16")
    return _check("no_w8a16", count == 0, f"W8A16 tensors={count}")


def _check_fused_projections(tensors: dict[str, str], require: bool) -> dict[str, Any]:
    """GDN fuses in_proj_qkv and in_proj_z at load time.

    vLLM merges the two weights into one projection when it builds the model,
    so a checkpoint that quantizes only one of them cannot be served.
    """

    mismatched: list[str] = []
    missing: list[str] = []
    pairs = 0
    for name, quant_type in tensors.items():
        if not name.endswith(f".linear_attn.{FUSED_PROJECTIONS[0]}.weight"):
            continue
        siblings = [name.replace(f".{FUSED_PROJECTIONS[0]}.", f".{other}.") for other in FUSED_PROJECTIONS]
        if any(sibling not in tensors for sibling in siblings):
            missing.append(name)
            continue
        pairs += 1
        if len({tensors[sibling] for sibling in siblings}) != 1:
            mismatched.append(name)
    ok = not mismatched and not missing and (pairs > 0 or not require)
    detail = f"pairs={pairs}"
    if mismatched:
        detail += f" mismatched={mismatched[:3]}"
    if missing:
        detail += f" missing_counterpart={missing[:3]}"
    return _check("fused_linear_attn_projections", ok, detail)


def _check_float_scopes(tensors: dict[str, str], patterns: tuple[str, ...]) -> dict[str, Any]:
    problems: list[str] = []
    unmatched: list[str] = []
    for pattern in patterns:
        matched = [name for name in tensors if pattern in name]
        if not matched:
            unmatched.append(pattern)
            continue
        problems.extend(name for name in matched if tensors[name] != FLOAT)
    ok = not problems and not unmatched
    detail = f"patterns={len(patterns)} violations={problems[:3]}"
    if unmatched:
        detail += f" unmatched_patterns={unmatched}"
    return _check("float_scopes", ok, detail)


def verify_artifact(
    model_path: Path,
    recipe: Recipe,
    *,
    hash_shards: bool = False,
) -> dict[str, Any]:
    """Verify an artifact against the precision scope a recipe declares."""

    description_path = model_path / DESCRIPTION
    if not description_path.is_file():
        raise ValueError(f"Missing {DESCRIPTION}: {description_path}")
    description = json.loads(description_path.read_text(encoding="utf-8"))
    tensors = tensor_types(description)
    histogram = Counter(tensors.values())

    checks: list[dict[str, Any]] = []
    if recipe.precision_groups:
        checks.append(_check_declared_types(tensors, recipe))
    if recipe.forbid_w8a16:
        checks.append(_check_no_w8a16(tensors))
    if recipe.require_fused_linear_attn or any(".linear_attn." in name for name in tensors):
        checks.append(_check_fused_projections(tensors, recipe.require_fused_linear_attn))
    if recipe.expected_float_patterns:
        checks.append(_check_float_scopes(tensors, recipe.expected_float_patterns))

    shards = sorted(model_path.glob("quant_model_weights-*.safetensors"))
    if not shards:
        checks.append(_check("shard_inventory", False, "no quant_model_weights-*.safetensors found"))
    elif recipe.expected_shards is not None or recipe.expected_weight_bytes is not None:
        total = sum(path.stat().st_size for path in shards)
        problems = []
        if recipe.expected_shards is not None and len(shards) != recipe.expected_shards:
            problems.append(f"shards={len(shards)} expected={recipe.expected_shards}")
        if recipe.expected_weight_bytes is not None and total != recipe.expected_weight_bytes:
            problems.append(f"bytes={total} expected={recipe.expected_weight_bytes}")
        checks.append(
            _check(
                "shard_inventory",
                not problems,
                f"shards={len(shards)} bytes={total}" + (f" {'; '.join(problems)}" if problems else ""),
            )
        )

    result: dict[str, Any] = {
        "valid": all(check["ok"] for check in checks),
        "artifact": str(model_path.resolve()),
        "recipe": recipe.name,
        "mixed_precision": recipe.mixed_precision,
        "tensors": len(tensors),
        "histogram": dict(sorted(histogram.items())),
        "checks": checks,
    }
    if hash_shards:
        result["shard_sha256"] = {path.name: _sha256(path) for path in shards}
    return result
