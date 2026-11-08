import json
from pathlib import Path

import pytest

from ascend_quant_toolkit.artifact import build_runtime_contract, prepare_runtime_metadata
from ascend_quant_toolkit.config import PrecisionGroup, Recipe
from ascend_quant_toolkit.registry import get_recipe
from ascend_quant_toolkit.scope import non_float_types, tensor_types, verify_artifact

DESCRIPTION = {
    "model_quant_type": "W8A8_DYNAMIC",
    "version": "1.0.0",
    "group_size": 0,
    "optional": {},
    "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC",
    "model.language_model.layers.0.mlp.experts.0.up_proj.weight": "W4A8_DYNAMIC",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "W8A8_DYNAMIC",
    "model.language_model.layers.0.linear_attn.in_proj_z.weight": "W8A8_DYNAMIC",
    "model.language_model.layers.0.linear_attn.out_proj.weight": "W8A8_DYNAMIC",
    "model.language_model.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
    "model.language_model.layers.0.self_attn.o_proj.weight": "FLOAT",
    "model.language_model.layers.0.linear_attn.in_proj_a.weight": "FLOAT",
    "model.language_model.layers.0.linear_attn.in_proj_b.weight": "FLOAT",
    "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": "FLOAT",
    "lm_head.weight": "FLOAT",
    "model.visual.blocks.0.norm1.weight": "FLOAT",
    "mtp.fc.weight": "FLOAT",
}


def _recipe(**overrides) -> Recipe:
    fields = {
        "name": "fixture",
        "description": "fixture",
        "model_type": "Qwen3.5-35B-A3B",
        "config_resource": "recipes/yaml/qwen35-35b-a3b-w4a8-gdnw8a8.yaml",
        "producer_quant_type": "W4A8_DYNAMIC",
        "runtime_quant_type": "ASCEND_QUANT_W4A8",
        "calibration_dataset": "calibImages",
        "weight_dtype": "int4",
        "activation_dtype": "int8",
        "quant_scheme": "W4A8",
        "precision_groups": (
            PrecisionGroup("mlp.experts", "W4A8_DYNAMIC", "int4", "int8"),
            PrecisionGroup("linear_attn+self_attn", "W8A8_DYNAMIC", "int8", "int8"),
        ),
        "expected_float_patterns": (
            "o_proj",
            "in_proj_a",
            "in_proj_b",
            "shared_expert",
            "lm_head",
            "visual",
            "mtp",
        ),
        "forbid_w8a16": True,
        "require_fused_linear_attn": True,
        "mixed_precision": True,
    }
    fields.update(overrides)
    return Recipe(**fields)


def _artifact(root: Path, description: dict, shards: int = 1, shard_bytes: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForCausalLM"]})
    )
    (root / "quant_model_description.json").write_text(json.dumps(description))
    for index in range(shards):
        (root / f"quant_model_weights-{index}.safetensors").write_bytes(b"x" * shard_bytes)
    return root


def _checks(result: dict) -> dict[str, dict]:
    return {check["name"]: check for check in result["checks"]}


def test_tensor_types_excludes_metadata_keys():
    tensors = tensor_types(DESCRIPTION)

    assert "model_quant_type" not in tensors
    assert "version" not in tensors
    assert len(tensors) == len(DESCRIPTION) - 4


def test_non_float_types_reports_every_quantized_type():
    assert non_float_types(DESCRIPTION) == {"W4A8_DYNAMIC", "W8A8_DYNAMIC"}


def test_mixed_artifact_passes_every_check(tmp_path: Path):
    result = verify_artifact(_artifact(tmp_path / "artifact", DESCRIPTION), _recipe())

    assert result["valid"] is True
    assert result["mixed_precision"] is True
    assert result["histogram"] == {"FLOAT": 7, "W4A8_DYNAMIC": 2, "W8A8_DYNAMIC": 4}
    assert all(check["ok"] for check in result["checks"])


def test_unrecognized_entry_is_rejected(tmp_path: Path):
    description = dict(DESCRIPTION)
    description["model.language_model.layers.0.mlp.experts.0.down_proj.weight"] = "BF16"

    with pytest.raises(ValueError, match="Unrecognized quantization entry"):
        verify_artifact(_artifact(tmp_path / "artifact", description), _recipe())


def test_w8a16_is_rejected(tmp_path: Path):
    description = dict(DESCRIPTION)
    description["model.language_model.layers.0.self_attn.k_proj.weight"] = "W8A16"

    result = verify_artifact(_artifact(tmp_path / "artifact", description), _recipe())

    assert result["valid"] is False
    assert _checks(result)["no_w8a16"]["ok"] is False
    assert _checks(result)["declared_precision_types"]["ok"] is False


def test_fused_projection_mismatch_is_rejected(tmp_path: Path):
    # vLLM fuses in_proj_qkv and in_proj_z at load, so mixed types are unserveable.
    description = dict(DESCRIPTION)
    description["model.language_model.layers.0.linear_attn.in_proj_z.weight"] = "W4A8_DYNAMIC"

    result = verify_artifact(_artifact(tmp_path / "artifact", description), _recipe())

    assert result["valid"] is False
    check = _checks(result)["fused_linear_attn_projections"]
    assert check["ok"] is False
    assert "pairs=1" in check["detail"]


def test_missing_fused_counterpart_is_rejected(tmp_path: Path):
    description = dict(DESCRIPTION)
    del description["model.language_model.layers.0.linear_attn.in_proj_z.weight"]

    result = verify_artifact(_artifact(tmp_path / "artifact", description), _recipe())

    check = _checks(result)["fused_linear_attn_projections"]
    assert check["ok"] is False
    assert "missing_counterpart" in check["detail"]


def test_float_scope_violation_is_rejected(tmp_path: Path):
    description = dict(DESCRIPTION)
    description["model.language_model.layers.0.self_attn.o_proj.weight"] = "W8A8_DYNAMIC"

    result = verify_artifact(_artifact(tmp_path / "artifact", description), _recipe())

    assert result["valid"] is False
    check = _checks(result)["float_scopes"]
    assert check["ok"] is False
    assert "o_proj" in check["detail"]


def test_unmatched_float_pattern_is_rejected(tmp_path: Path):
    result = verify_artifact(
        _artifact(tmp_path / "artifact", DESCRIPTION),
        _recipe(expected_float_patterns=("does_not_exist",)),
    )

    assert result["valid"] is False
    assert "does_not_exist" in _checks(result)["float_scopes"]["detail"]


def test_shard_inventory_is_pinned(tmp_path: Path):
    artifact = _artifact(tmp_path / "artifact", DESCRIPTION, shards=2, shard_bytes=8)

    result = verify_artifact(
        artifact,
        _recipe(expected_shards=6, expected_weight_bytes=48),
    )

    check = _checks(result)["shard_inventory"]
    assert check["ok"] is False
    assert "shards=2 expected=6" in check["detail"]


def test_shard_hashes_are_optional(tmp_path: Path):
    artifact = _artifact(tmp_path / "artifact", DESCRIPTION)

    plain = verify_artifact(artifact, _recipe())
    hashed = verify_artifact(artifact, _recipe(), hash_shards=True)

    assert "shard_sha256" not in plain
    digest = hashed["shard_sha256"]["quant_model_weights-0.safetensors"]
    assert len(digest) == 64
    int(digest, 16)


def test_recipe_without_declarations_skips_scope_checks(tmp_path: Path):
    artifact = _artifact(tmp_path / "artifact", {"layer.weight": "W8A8_MIX"})

    result = verify_artifact(artifact, get_recipe("qwen25-w8a8"))

    assert result["checks"] == []
    assert result["valid"] is True


def test_linear_attn_pairing_is_checked_even_without_a_declaration(tmp_path: Path):
    # A GDN checkpoint with mismatched fused projections is unserveable
    # whatever recipe is passed, so the check is not opt-in.
    description = {
        "layer.linear_attn.in_proj_qkv.weight": "W8A8_DYNAMIC",
        "layer.linear_attn.in_proj_z.weight": "W4A8_DYNAMIC",
    }

    result = verify_artifact(_artifact(tmp_path / "artifact", description), get_recipe("qwen25-w8a8"))

    assert result["valid"] is False
    assert _checks(result)["fused_linear_attn_projections"]["ok"] is False


def test_mixed_artifact_cannot_be_migrated_or_exported(tmp_path: Path):
    artifact = _artifact(tmp_path / "artifact", DESCRIPTION, shards=6, shard_bytes=8)
    recipe = _recipe(expected_shards=6, expected_weight_bytes=48)

    with pytest.raises(ValueError, match="single-scheme"):
        prepare_runtime_metadata(artifact, recipe)
    with pytest.raises(ValueError, match="single-scheme"):
        build_runtime_contract(artifact, recipe)
    assert not (artifact / "quant_model_description.modelslim.json").exists()


def test_multi_type_description_is_refused_even_for_single_scheme_recipe(tmp_path: Path):
    artifact = _artifact(tmp_path / "artifact", DESCRIPTION)

    with pytest.raises(ValueError, match="single-scheme"):
        prepare_runtime_metadata(artifact, get_recipe("qwen25-w8a8"))


def test_single_scheme_artifact_still_migrates(tmp_path: Path):
    artifact = _artifact(
        tmp_path / "artifact",
        {"model_quant_type": "W8A8_MIX", "layer.weight": "W8A8_MIX"},
    )

    result = prepare_runtime_metadata(artifact, get_recipe("qwen25-w8a8"))

    assert result["runtime_quant_type"] == "ASCEND_QUANT_W8A8"
    assert result["replacements"] == 2
