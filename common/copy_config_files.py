import glob
import os

from common.security.path import (
    get_valid_read_path,
    get_valid_write_path,
    json_safe_dump,
    json_safe_load,
    safe_copy_file,
    set_file_stat,
)

W4A8_FORMAT_FIELDS = (
    "weight_packing",
    "weight_signedness",
    "weight_scale_granularity",
    "activation_scale_granularity",
    "zero_point_semantics",
    "supported_shapes",
    "operator_name",
)


def validate_w4a8_format_contract(quant_config):
    """Reject W4A8 conversion until its artifact contract is explicit."""
    quant_type = str(quant_config.model_quant_type.value).strip().lower()
    if not (quant_type == "w4a8" or quant_type.startswith("w4a8_")):
        return

    contract = getattr(quant_config, "w4a8_format_contract", None)
    if not isinstance(contract, dict):
        raise TypeError("W4A8 conversion requires an explicit w4a8_format_contract.")

    missing = []
    for field in W4A8_FORMAT_FIELDS:
        value = contract.get(field)
        if value is None or value == "" or value == []:
            missing.append(field)
    if missing:
        raise ValueError(
            "W4A8 format contract is incomplete: " + ", ".join(missing)
        )

    if not isinstance(contract["supported_shapes"], list):
        raise TypeError("W4A8 supported_shapes must be a non-empty list.")


def copy_json(src_path: str, dst_path: str, quant_config, mindie_format: bool):
    safe_copy_file(src_path, dst_path)
    set_file_stat(dst_path, "600")


def modify_config_json(src_path: str, dst_path: str, quant_config, mindie_format: bool, custom_hook=None):
    validate_w4a8_format_contract(quant_config)
    model_config = json_safe_load(src_path)
    model_config["quantize"] = str(quant_config.model_quant_type.value).lower()

    config_dir = os.path.dirname(dst_path)
    if mindie_format:
        candidates = glob.glob(os.path.join(config_dir, "quant_model_description*.json"))
        if not candidates:
            raise FileNotFoundError("No quant_model_description*.json found in save directory.")
        dest_quant_description_filepath = candidates[0]
    else:
        dest_quant_description_filepath = os.path.join(config_dir, "quant_model_description.json")

    dest_quant_description_filepath = get_valid_write_path(dest_quant_description_filepath, is_dir=False)
    quant_description_data = json_safe_load(dest_quant_description_filepath, check_user_stat=True)
    quantization_config = {} if mindie_format else quant_description_data
    quantization_config.update(
        {
            "kv_quant_type": "C8" if quant_config.use_kvcache_quant else None,
            "fa_quant_type": "FAQuant" if quant_config.use_fa_quant else None,
            "group_size": max(0, quant_config.group_size),
        }
    )

    if mindie_format:
        model_config["quantization_config"] = quantization_config
    else:
        json_safe_dump(quantization_config, dest_quant_description_filepath, indent=4)

    if custom_hook:
        custom_hook(model_config)

    json_safe_dump(model_config, dst_path, indent=4)


EXCLUDING_SUBFIX_LIST = ("index.json",)
FILE_HOOKS = {"config.json": modify_config_json}
DEFAULT_FILE_HOOKS = copy_json


def copy_config_files(input_path, output_path, quant_config, mindie_format=None, custom_hooks=None):
    # Validate before copying any file so a rejected W4A8 conversion cannot
    # leave a partially materialized output directory.
    validate_w4a8_format_contract(quant_config)
    for file in os.listdir(input_path):
        if not file.endswith((".json", ".py")):
            continue
        if any(file.endswith(subfix) for subfix in EXCLUDING_SUBFIX_LIST):
            continue

        src_path = get_valid_read_path(os.path.join(input_path, file), extensions=[".json", ".py"])
        dst_path = get_valid_write_path(os.path.join(output_path, file))
        hook = custom_hooks[file] if custom_hooks and file in custom_hooks else FILE_HOOKS.get(file, DEFAULT_FILE_HOOKS)
        hook(src_path, dst_path, quant_config, mindie_format)
