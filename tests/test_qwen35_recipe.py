import hashlib

from ascend_quant_toolkit.registry import get_recipe

# The YAML decides what a 24 GB checkpoint contains, so its bytes are pinned.
# Update this digest only when the recipe body changes on purpose, and record
# the new artifact in docs/evidence/ with the release that ships it.
CONFIG_SHA256 = "0bcef0f56ddebea8f34841127e6ac5b369ffee20853e5a7d148c344387fd9384"


def test_recipe_is_discoverable_with_expected_fields():
    recipe = get_recipe("qwen35-35b-a3b-w4a8-gdnw8a8")

    assert recipe.model_type == "Qwen3.5-35B-A3B"
    assert recipe.quant_scheme == "W4A8"
    assert recipe.calibration_dataset == "calibImages"
    assert recipe.weight_dtype == "int4"
    assert recipe.activation_dtype == "int8"
    assert recipe.mixed_precision is True
    assert recipe.forbid_w8a16 is True
    assert recipe.require_fused_linear_attn is True
    assert recipe.expected_shards == 6
    assert recipe.expected_weight_bytes == 24_047_130_272


def test_recipe_declares_both_precision_groups():
    recipe = get_recipe("qwen35-35b-a3b-w4a8-gdnw8a8")
    scopes = [group.scope for group in recipe.precision_groups]
    types = {group.producer_quant_type for group in recipe.precision_groups}

    assert types == {"W4A8_DYNAMIC", "W8A8_DYNAMIC"}
    assert "mlp.experts" in scopes
    assert any("linear_attn" in scope for scope in scopes)
    assert any("self_attn" in scope for scope in scopes)
    experts = next(group for group in recipe.precision_groups if group.scope == "mlp.experts")
    assert experts.weight_dtype == "int4"


def test_recipe_keeps_every_unquantized_scope_float():
    recipe = get_recipe("qwen35-35b-a3b-w4a8-gdnw8a8")

    assert set(recipe.expected_float_patterns) == {
        "o_proj",
        "in_proj_a",
        "in_proj_b",
        "shared_expert",
        "lm_head",
        "visual",
        "mtp",
    }


def test_recipe_config_resource_exists_and_is_pinned():
    recipe = get_recipe("qwen35-35b-a3b-w4a8-gdnw8a8")
    config = recipe.config_path()

    assert config.is_file()
    assert hashlib.sha256(config.read_bytes()).hexdigest() == CONFIG_SHA256


def test_recipe_config_keeps_producer_config_id():
    recipe = get_recipe("qwen35-35b-a3b-w4a8-gdnw8a8")

    text = recipe.config_path().read_text(encoding="utf-8")
    # The published checkpoint records this config_id in its own description.
    assert "config_id: Qwen3_5_moe_w8a8_norot" in text
    assert '"*mlp.experts*"' in text
    assert '"*linear_attn.in_proj_qkv"' in text
