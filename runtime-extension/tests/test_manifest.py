import json
from importlib import resources

from vllm_ascend_quant_ext import EXTENSION_ID, __version__


def test_manifest_is_inert_and_matches_package():
    path = resources.files("vllm_ascend_quant_ext").joinpath(
        "vllm-hust-extension-v0.3.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "0.3-experimental"
    assert manifest["extension_id"] == EXTENSION_ID
    assert manifest["extension_version"] == __version__
    assert manifest["host"]["api_range"] == ">=1,<2"
    assert manifest["requires_extensions"] == []
    assert manifest["resource_claims"] == [
        {
            "resource": "vllm-ascend.quantized-artifact.validation",
            "scope": "vllm-process",
            "mode": "shared",
        }
    ]
    assert manifest["implementation"][0]["status"] == "import_only"
    assert manifest["activation"] == {
        "entry_points": [],
        "environment": {},
        "additional_config": {},
    }
