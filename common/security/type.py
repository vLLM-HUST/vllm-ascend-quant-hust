import importlib
import re
from typing import Mapping

try:
    from msmodelslim import NEW_PACKAGE_NAME, OLD_PACKAGE_NAME
except ImportError:
    OLD_PACKAGE_NAME = "modelslim"
    NEW_PACKAGE_NAME = "msmodelslim"


STR_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9\"'><=\[\])(,}{: /.~-]")


def type_to_str(value_type):
    return " or ".join([item.__name__ for item in value_type]) if isinstance(value_type, tuple) else value_type.__name__


def check_type(value, value_type, param_name="value", additional_check_func=None, additional_msg=None):
    is_modelslim_import = (
        value.__class__.__module__.startswith(OLD_PACKAGE_NAME)
        and getattr(value_type, "__module__", "").startswith(NEW_PACKAGE_NAME)
    )
    if is_modelslim_import:
        original_module_name = value_type.__module__
        redirect_module_name = original_module_name.replace(NEW_PACKAGE_NAME, OLD_PACKAGE_NAME)
        module = importlib.import_module(redirect_module_name)
        value_type = getattr(module, value_type.__qualname__)

    if not isinstance(value, value_type):
        raise TypeError(f"{param_name} must be {type_to_str(value_type)}, not {type(value).__name__}.")
    if additional_check_func is not None:
        additional_msg = (" " + additional_msg) if additional_msg else ""
        if isinstance(value, (list, tuple)):
            if not all(map(additional_check_func, value)):
                raise ValueError(f"Element in {param_name} is invalid." + additional_msg)
        elif not additional_check_func(value):
            raise ValueError(f"Value of {param_name} is invalid." + additional_msg)
    is_int_when_tuple = isinstance(value_type, tuple) and int in value_type and bool not in value_type
    if value_type == int or is_int_when_tuple:
        if isinstance(value, bool):
            raise TypeError(f"{param_name} must be {type_to_str(value_type)}, not bool.")


def check_number(value, value_type=(int, float), min_value=None, max_value=None, param_name="value"):
    check_type(value, value_type, param_name=param_name)
    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name} = {value} is larger than {max_value}.")
    if min_value is not None and value < min_value:
        raise ValueError(f"{param_name} = {value} is smaller than {min_value}.")


def check_character(value, param_name="value"):
    max_depth = 100

    def check_character_recursion(inner_value, depth=0):
        if isinstance(inner_value, str):
            if re.search(STR_WHITE_LIST_REGEX, inner_value):
                raise ValueError(f"{param_name} contains invalid characters.")
        elif isinstance(inner_value, (list, tuple)):
            if depth > max_depth:
                raise ValueError(f"Recursion depth of {param_name} exceeds limitation.")
            for sub_value in inner_value:
                check_character_recursion(sub_value, depth=depth + 1)

    check_character_recursion(value)


def check_dict_character(dict_value, key_max_len=512, param_name="dict"):
    max_depth = 100

    def check_dict_character_recursion(inner_dict_value, depth=0):
        check_type(inner_dict_value, dict, param_name=param_name)
        for key, value in inner_dict_value.items():
            key = str(key)
            check_character(key, param_name=f"{param_name} key")
            if key_max_len > 0 and len(key) > key_max_len:
                raise ValueError(f"Length of {param_name} key exceeds limitation {key_max_len}.")
            if isinstance(value, dict):
                if depth > max_depth:
                    raise ValueError(f"Recursion depth of {param_name} exceeds limitation.")
                check_dict_character_recursion(value, depth=depth + 1)
            else:
                check_character(value, param_name=param_name)

    check_dict_character_recursion(dict_value)


def check_mapping_element(mapping_value, value_type, param_name="dict", additional_msg=None):
    check_type(mapping_value, Mapping, param_name=param_name)
    additional_msg = (" " + additional_msg) if additional_msg else ""
    for key in mapping_value:
        value = mapping_value[key]
        if not isinstance(value, value_type):
            raise ValueError(
                f"Param of dict {param_name}[{key}] should be {type_to_str(value_type)}, " + additional_msg
            )
