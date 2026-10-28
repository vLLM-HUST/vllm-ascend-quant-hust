"""Validated Qwen3.5-35B-A3B W4A8 experts + W8A8 GDN recipe.

This is the recipe behind the published 2026-10-23 single-card curve: int4
weights on the 40 x 256 MLP experts, int8 weights on the 30 GDN
``linear_attn`` projections and the 10 ``self_attn`` q/k/v projections, and
FLOAT everywhere else.

It is a *mixed-precision* checkpoint: one artifact carries two quantization
types. That is producible and verifiable, but artifact contract 1.1.0 is
single-scheme, so runtime metadata migration and contract export refuse it.
The single-scheme fields below (``producer_quant_type``, ``runtime_quant_type``,
``quant_scheme``) are therefore nominal: they name the dominant scheme for
listings and logs only.
"""

from ascend_quant_toolkit.config import PrecisionGroup, Recipe


def get_recipe() -> Recipe:
    return Recipe(
        name="qwen35-35b-a3b-w4a8-gdnw8a8",
        description=(
            "Qwen3.5-35B-A3B experts W4A8_DYNAMIC + GDN linear_attn and "
            "self_attn q/k/v W8A8_DYNAMIC, no rotation, calibImages calibration"
        ),
        model_type="Qwen3.5-35B-A3B",
        config_resource="recipes/yaml/qwen35-35b-a3b-w4a8-gdnw8a8.yaml",
        producer_quant_type="W4A8_DYNAMIC",
        runtime_quant_type="ASCEND_QUANT_W4A8",
        calibration_dataset="calibImages",
        weight_dtype="int4",
        activation_dtype="int8",
        quant_scheme="W4A8",
        precision_groups=(
            PrecisionGroup(
                scope="mlp.experts",
                producer_quant_type="W4A8_DYNAMIC",
                weight_dtype="int4",
                activation_dtype="int8",
            ),
            PrecisionGroup(
                scope="linear_attn.in_proj_qkv|in_proj_z|out_proj",
                producer_quant_type="W8A8_DYNAMIC",
                weight_dtype="int8",
                activation_dtype="int8",
            ),
            PrecisionGroup(
                scope="self_attn.q|k|v",
                producer_quant_type="W8A8_DYNAMIC",
                weight_dtype="int8",
                activation_dtype="int8",
            ),
        ),
        expected_float_patterns=(
            "o_proj",
            "in_proj_a",
            "in_proj_b",
            "shared_expert",
            "lm_head",
            "visual",
            "mtp",
        ),
        expected_shards=6,
        expected_weight_bytes=24_047_130_272,
        forbid_w8a16=True,
        require_fused_linear_attn=True,
        mixed_precision=True,
    )
