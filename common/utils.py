import argparse
import json
import os
import shutil

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    from msmodelslim import logger

    logger.warning("torch_npu is not available, if you are using NPU, please install torch_npu")

from common.security.path import get_valid_read_path, get_valid_write_path


class SafeGenerator:
    @staticmethod
    def get_config_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=True)
        try:
            return AutoConfig.from_pretrained(model_path, local_files_only=True, **kwargs)
        except EnvironmentError as env_err:
            raise EnvironmentError(
                "Get config from pretrained failed, please check files in the model path. "
                f"Original error: {env_err}"
            ) from env_err
        except Exception as err:
            raise ValueError(
                "Get config from pretrained failed, please check files in the model path. "
                f"Original error: {err}"
            ) from err

    @staticmethod
    def get_model_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=True)
        try:
            return AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **kwargs)
        except EnvironmentError as env_err:
            raise EnvironmentError(
                "Get model from pretrained failed, please check model weights files in the model path. "
                f"Original error: {env_err}"
            ) from env_err
        except Exception as err:
            raise ValueError(
                "Get model from pretrained failed, please check model weights files in the model path. "
                f"Original error: {err}"
            ) from err

    @staticmethod
    def get_tokenizer_from_pretrained(model_path, **kwargs):
        model_path = get_valid_read_path(model_path, is_dir=True, check_user_stat=True)
        try:
            return AutoTokenizer.from_pretrained(model_path, local_files_only=True, **kwargs)
        except EnvironmentError as env_err:
            raise EnvironmentError(
                "Get tokenizer from pretrained failed, please check tokenizer files in the model path. "
                f"Original error: {env_err}"
            ) from env_err
        except Exception as err:
            raise ValueError(
                "Get tokenizer from pretrained failed, please check tokenizer files in the model path. "
                f"Original error: {err}"
            ) from err

    @staticmethod
    def copy_tokenizer_files(model_dir, dest_dir):
        model_dir = get_valid_read_path(model_dir, is_dir=True, check_user_stat=True)
        dest_dir = get_valid_write_path(dest_dir, is_dir=True) if os.path.exists(dest_dir) else dest_dir
        os.makedirs(dest_dir, mode=0o750, exist_ok=True)
        filenames = os.listdir(model_dir)
        max_file_num = 1024
        if len(filenames) > max_file_num:
            raise argparse.ArgumentTypeError(f"The file num in dir is {len(filenames)}, exceeds {max_file_num}.")
        for filename in filenames:
            if any(name in filename for name in ["tokenizer", "tokenization", "special_token_map", "generation", "configuration", "tiktoken"]):
                src_filepath = os.path.join(model_dir, filename)
                dest_filepath = os.path.join(dest_dir, filename)
                if os.path.isfile(src_filepath):
                    shutil.copyfile(src_filepath, dest_filepath)
                    os.chmod(dest_filepath, 0o600)

    @staticmethod
    def load_jsonl(dataset_path, key_name="inputs_pretokenized"):
        dataset = []
        with os.fdopen(os.open(dataset_path, os.O_RDONLY, 0o600), "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                dataset.append(data.get(key_name, line))
        return dataset


def cmd_bool(cmd_arg):
    if cmd_arg == "True":
        return True
    if cmd_arg == "False":
        return False
    raise ValueError(f"{cmd_arg} should be True or False")
