import json
import os
import re
import shutil
import stat
import sys

from msmodelslim import logger
from msmodelslim.utils.logging import LOGGER_FUNC

from common.security.type import check_dict_character, check_type


PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.:\\ -]")
MAX_READ_FILE_SIZE_4G = 4294967296
WRITE_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH | stat.S_IROTH | stat.S_IXOTH
READ_FILE_NOT_PERMITTED_STAT = stat.S_IWGRP | stat.S_IWOTH


def is_endswith_extensions(path, extensions):
    if isinstance(extensions, (list, tuple)):
        return any(path.endswith(extension) for extension in extensions)
    if isinstance(extensions, str):
        return path.endswith(extensions)
    return False


def get_valid_path(path, extensions=None):
    check_type(path, str, "path")
    if not path:
        raise ValueError("The value of the path cannot be empty.")
    if PATH_WHITE_LIST_REGEX.search(path):
        raise ValueError("Input path contains invalid characters.")
    if os.path.islink(os.path.abspath(path)):
        raise ValueError(f"The value of the path cannot be soft link: {path}.")

    real_path = os.path.realpath(path)
    file_name = os.path.split(real_path)[1]
    if len(file_name) > 255:
        raise ValueError("The length of filename should be less than 256.")
    if len(real_path) > 4096:
        raise ValueError("The length of file path should be less than 4096.")
    if real_path != path and PATH_WHITE_LIST_REGEX.search(real_path):
        raise ValueError("Input path contains invalid characters.")
    if extensions and not is_endswith_extensions(path, extensions):
        raise ValueError(f'The filename {path} does not end with "{extensions}".')
    return real_path


def is_belong_to_user_or_group(file_stat):
    if sys.platform.startswith("win"):
        return True
    return file_stat.st_uid == os.getuid() or file_stat.st_gid in os.getgroups()


def check_others_not_writable(path):
    if sys.platform.startswith("win"):
        return
    dir_stat = os.stat(path)
    is_writable = bool(dir_stat.st_mode & stat.S_IWGRP) or bool(dir_stat.st_mode & stat.S_IWOTH)
    if is_writable:
        logger.warning("The file path %r may be insecure because it can be written by others.", path)


def check_path_owner_consistent(path):
    if sys.platform.startswith("win"):
        return
    file_owner = os.stat(path).st_uid
    if file_owner != os.getuid() and os.getuid() != 0:
        logger.warning("The file path %r may be insecure because it does not belong to you.", path)


def check_dirpath_before_read(path):
    dirpath = os.path.dirname(os.path.realpath(path))
    check_others_not_writable(dirpath)
    check_path_owner_consistent(dirpath)


def get_valid_read_path(path, extensions=None, size_max=MAX_READ_FILE_SIZE_4G, check_user_stat=True, is_dir=False):
    check_dirpath_before_read(path)
    real_path = get_valid_path(path, extensions)
    if not is_dir and not os.path.isfile(real_path):
        raise ValueError(f"The path {path} does not exist or is not a file.")
    if is_dir and not os.path.isdir(real_path):
        raise ValueError(f"The path {path} does not exist or is not a directory.")

    file_stat = os.stat(real_path)
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(file_stat):
        if os.geteuid() == 0:
            logger.warning("The file %r does not belong to the current user or group; current user is root.", path)
        else:
            raise ValueError(f"The file {path} does not belong to the current user or group.")
    if not sys.platform.startswith("win") and check_user_stat and os.stat(path).st_mode & READ_FILE_NOT_PERMITTED_STAT:
        raise ValueError(f"The file {path} is group writable, or is others writable.")
    if not os.access(real_path, os.R_OK):
        raise ValueError(f"Current user does not have read permission to the file {path}.")
    if not is_dir and size_max > 0 and file_stat.st_size > size_max:
        raise ValueError(f"The file {path} exceeds size limitation of {size_max}.")
    return real_path


def check_write_directory(dir_name, check_user_stat=True):
    real_dir_name = get_valid_path(dir_name)
    if not os.path.isdir(real_dir_name):
        raise ValueError(f"The file write directory {dir_name} does not exist.")
    if not os.access(real_dir_name, os.W_OK):
        raise ValueError(f"Current user does not have write permission to directory {dir_name}.")
    if check_user_stat and not sys.platform.startswith("win") and not is_belong_to_user_or_group(os.stat(real_dir_name)):
        if os.geteuid() == 0:
            logger.warning("The directory %r does not belong to the current user or group; current user is root.", dir_name)
        else:
            raise ValueError(f"The directory {dir_name} does not belong to the current user or group.")


def get_write_directory(dir_name, write_mode=0o750):
    real_dir_name = get_valid_path(dir_name)
    if os.path.exists(real_dir_name):
        logger.info("write directory exists, write file to directory %r", dir_name)
    else:
        logger.warning("write directory does not exist, creating directory %r", dir_name)
        os.makedirs(name=real_dir_name, mode=write_mode, exist_ok=True)
    return real_dir_name


def get_valid_write_path(path, extensions=None, check_user_stat=True, is_dir=False, warn_exists=True):
    real_path = get_valid_path(path, extensions)
    real_path_dir = real_path if is_dir else os.path.dirname(real_path)
    check_write_directory(real_path_dir, check_user_stat=check_user_stat)
    if not is_dir and os.path.exists(real_path):
        if os.path.isdir(real_path):
            raise ValueError(f"The file {path} exists and is a directory.")
        if not os.access(real_path, os.W_OK):
            raise ValueError(f"The file {path} exists and is not writable.")
        if not sys.platform.startswith("win") and check_user_stat:
            file_stat = os.stat(real_path)
            if file_stat.st_uid != os.getuid():
                raise ValueError(f"The file {path} does not belong to the current user.")
            if file_stat.st_mode & WRITE_FILE_NOT_PERMITTED_STAT:
                raise ValueError(f"The file {path} permission for others is not 0, or is group writable.")
        if warn_exists:
            logger.warning("%r already exists. The original file will be overwritten.", path)
    return real_path


def json_safe_load(path, extensions="json", size_max=MAX_READ_FILE_SIZE_4G, key_max_len=512, check_user_stat=True):
    path = get_valid_read_path(path, extensions, size_max, check_user_stat)
    with open(path, encoding="utf-8") as json_file:
        raw_dict = json.load(json_file)
    if isinstance(raw_dict, dict):
        check_dict_character(raw_dict, key_max_len)
    return raw_dict


def json_safe_dump(obj, path, indent=None, extensions="json", check_user_stat=True):
    if isinstance(obj, dict):
        check_dict_character(obj)
    write_path = get_valid_write_path(path, extensions, check_user_stat)
    with os.fdopen(os.open(write_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=0o600), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def safe_copy_file(src_path, dest_path, size_max=MAX_READ_FILE_SIZE_4G):
    src_path = get_valid_read_path(src_path, size_max=size_max)
    if os.path.isdir(dest_path):
        dest_path = os.path.join(dest_path, os.path.basename(src_path))
    dest_path = get_valid_write_path(dest_path)
    shutil.copy2(src_path, dest_path, follow_symlinks=False)


def set_file_stat(path, stat_mode="640"):
    real_path = get_valid_path(path)
    if os.path.isfile(real_path):
        os.chmod(real_path, int(stat_mode, 8))


def safe_delete_path_if_exists(path, logger_level="info"):
    if os.path.exists(path):
        is_dir = os.path.isdir(path)
        path = get_valid_write_path(path, check_user_stat=True, is_dir=is_dir, warn_exists=False)
        logger_func = LOGGER_FUNC[logger_level]
        if os.path.isfile(path):
            logger_func(f"File '{path}' exists and will be deleted.")
            os.remove(path)
        else:
            logger_func(f"Folder '{path}' exists and will be deleted.")
            shutil.rmtree(path)
