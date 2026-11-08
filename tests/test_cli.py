import argparse
import json
from pathlib import Path

from ascend_quant_toolkit import cli
from ascend_quant_toolkit.config import PrecisionGroup, Recipe

DESCRIPTION = {
    "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": "W4A8_DYNAMIC",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "W8A8_DYNAMIC",
    "model.language_model.layers.0.linear_attn.in_proj_z.weight": "W8A8_DYNAMIC",
    "model.language_model.layers.0.self_attn.o_proj.weight": "FLOAT",
}


def _recipe(mixed: bool) -> Recipe:
    groups = (
        (
            PrecisionGroup("mlp.experts", "W4A8_DYNAMIC", "int4", "int8"),
            PrecisionGroup("linear_attn", "W8A8_DYNAMIC", "int8", "int8"),
        )
        if mixed
        else ()
    )
    return Recipe(
        name="fixture",
        description="fixture",
        model_type="Qwen3.5-35B-A3B",
        config_resource="recipes/yaml/qwen25-w8a8.yaml",
        producer_quant_type="W4A8_DYNAMIC",
        runtime_quant_type="ASCEND_QUANT_W4A8",
        calibration_dataset="calibImages",
        weight_dtype="int4",
        activation_dtype="int8",
        quant_scheme="W4A8",
        precision_groups=groups,
        forbid_w8a16=True,
        require_fused_linear_attn=True,
        mixed_precision=mixed,
    )


def _artifact(root: Path, description: dict) -> Path:
    root.mkdir(parents=True)
    config = {"model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForCausalLM"]}
    (root / "config.json").write_text(json.dumps(config))
    (root / "quant_model_description.json").write_text(json.dumps(description))
    (root / "quant_model_weights-0.safetensors").write_bytes(b"weights")
    return root


def _args(artifact: Path, tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        recipe="fixture",
        model=str(tmp_path / "source-model"),
        output=str(artifact),
        device="npu:0",
        executable="msmodelslim",
        trust_remote_code=False,
    )


def test_quantize_verifies_a_mixed_recipe_and_skips_the_contract(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path / "artifact", DESCRIPTION)
    monkeypatch.setattr(cli, "get_recipe", lambda name: _recipe(mixed=True))
    monkeypatch.setattr(cli, "run_quantization", lambda request: 0)

    assert cli.command_quantize(_args(artifact, tmp_path)) == 0

    assert (artifact / "ascend_quant_manifest.json").is_file()
    # A mixed checkpoint must not be stamped with a single-scheme contract.
    assert not (artifact / "ascend_quant_artifact.json").exists()
    assert not (artifact / "quant_model_description.modelslim.json").exists()


def test_quantize_fails_when_the_mixed_scope_does_not_match(tmp_path, monkeypatch):
    broken = dict(DESCRIPTION)
    broken["model.language_model.layers.0.linear_attn.in_proj_z.weight"] = "W4A8_DYNAMIC"
    artifact = _artifact(tmp_path / "artifact", broken)
    monkeypatch.setattr(cli, "get_recipe", lambda name: _recipe(mixed=True))
    monkeypatch.setattr(cli, "run_quantization", lambda request: 0)

    assert cli.command_quantize(_args(artifact, tmp_path)) == 2


def test_quantize_keeps_the_contract_for_a_single_scheme_recipe(tmp_path, monkeypatch):
    artifact = _artifact(
        tmp_path / "artifact",
        {"model_quant_type": "W8A8_MIX", "layer.weight": "W8A8_MIX"},
    )
    monkeypatch.setattr(
        cli,
        "get_recipe",
        lambda name: Recipe(
            name="fixture",
            description="fixture",
            model_type="Qwen2.5-14B-Instruct",
            config_resource="recipes/yaml/qwen25-w8a8.yaml",
            producer_quant_type="W8A8_MIX",
            runtime_quant_type="ASCEND_QUANT_W8A8",
            calibration_dataset="c4",
            weight_dtype="int8",
            activation_dtype="int8",
            quant_scheme="W8A8",
        ),
    )
    monkeypatch.setattr(cli, "run_quantization", lambda request: 0)

    assert cli.command_quantize(_args(artifact, tmp_path)) == 0

    assert (artifact / "ascend_quant_artifact.json").is_file()
    assert (artifact / "quant_model_description.modelslim.json").is_file()
