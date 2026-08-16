import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_copy_module(monkeypatch):
    security_path = types.ModuleType("common.security.path")
    security_path.get_valid_read_path = lambda path, **_kwargs: path
    security_path.get_valid_write_path = lambda path, **_kwargs: path
    security_path.json_safe_load = lambda path, **_kwargs: json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    security_path.json_safe_dump = lambda value, path, **_kwargs: Path(path).write_text(
        json.dumps(value), encoding="utf-8"
    )
    security_path.safe_copy_file = lambda src, dst: Path(dst).write_bytes(
        Path(src).read_bytes()
    )
    security_path.set_file_stat = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "common.security.path", security_path)

    spec = importlib.util.spec_from_file_location(
        "w4a8_copy_config_files", ROOT / "common" / "copy_config_files.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _quant_config(quant_type, contract=None):
    return SimpleNamespace(
        model_quant_type=SimpleNamespace(value=quant_type),
        use_kvcache_quant=False,
        use_fa_quant=False,
        group_size=0,
        w4a8_format_contract=contract,
    )


def _complete_contract():
    return {
        "weight_packing": "owner-declared",
        "weight_signedness": "owner-declared",
        "weight_scale_granularity": "owner-declared",
        "activation_scale_granularity": "owner-declared",
        "zero_point_semantics": "owner-declared",
        "supported_shapes": ["owner-declared"],
        "operator_name": "owner-declared",
    }


def test_incomplete_w4a8_fails_before_output_mutation(tmp_path, monkeypatch):
    module = _load_copy_module(monkeypatch)
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    output_path.mkdir()
    (input_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(TypeError, match="explicit w4a8_format_contract"):
        module.copy_config_files(input_path, output_path, _quant_config("W4A8"))

    assert list(output_path.iterdir()) == []


def test_complete_w4a8_contract_passes_the_format_boundary(monkeypatch):
    module = _load_copy_module(monkeypatch)
    module.validate_w4a8_format_contract(
        _quant_config("w4a8_candidate", _complete_contract())
    )


def test_supported_shapes_must_be_a_list(monkeypatch):
    module = _load_copy_module(monkeypatch)
    contract = _complete_contract()
    contract["supported_shapes"] = "owner-declared"

    with pytest.raises(TypeError, match="supported_shapes"):
        module.validate_w4a8_format_contract(_quant_config("w4a8", contract))


def test_existing_quant_formats_remain_unchanged(monkeypatch):
    module = _load_copy_module(monkeypatch)
    module.validate_w4a8_format_contract(_quant_config("w4a4_flatquant_dynamic"))
    module.validate_w4a8_format_contract(_quant_config("w8a8"))
