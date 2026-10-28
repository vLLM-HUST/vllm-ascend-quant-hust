from ascend_quant_toolkit.registry import get_recipe, list_recipes


def test_builtin_recipe_is_discoverable():
    names = {recipe.name for recipe in list_recipes()}
    assert {"qwen25-w8a8", "qwen35-35b-a3b-w4a8-gdnw8a8"} <= names


def test_builtin_qwen25_recipe_keeps_its_contract():
    recipe = get_recipe("qwen25-w8a8")
    assert recipe.producer_quant_type == "W8A8_MIX"
    assert recipe.runtime_quant_type == "ASCEND_QUANT_W8A8"
    assert recipe.quant_scheme == "W8A8"
    assert recipe.mixed_precision is False


def test_builtin_qwen35_recipe_is_mixed_precision():
    recipe = get_recipe("qwen35-35b-a3b-w4a8-gdnw8a8")
    assert recipe.mixed_precision is True
    assert len(recipe.precision_groups) == 3
