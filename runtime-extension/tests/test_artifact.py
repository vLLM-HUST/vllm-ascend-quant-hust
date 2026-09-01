import pytest

from vllm_ascend_quant_ext.artifact import (
    SCHEMA_VERSION,
    ArtifactContractError,
    ArtifactContractValidator,
)


def _contract():
    return {
        "schema_version": SCHEMA_VERSION,
        "weight_packing": "owner-declared",
        "weight_signedness": "owner-declared",
        "weight_scale_granularity": "owner-declared",
        "activation_scale_granularity": "owner-declared",
        "zero_point_semantics": "owner-declared",
        "supported_shapes": ["owner-declared"],
        "operator_name": "owner-declared",
    }


def test_known_contract_is_accepted():
    assert ArtifactContractValidator.validate(_contract()) == _contract()


def test_unknown_schema_fails_closed():
    contract = _contract()
    contract["schema_version"] = "unknown/v9"
    with pytest.raises(ArtifactContractError, match="unsupported"):
        ArtifactContractValidator.validate(contract)


def test_missing_layout_fails_closed():
    contract = _contract()
    del contract["weight_packing"]
    with pytest.raises(ArtifactContractError, match="incomplete"):
        ArtifactContractValidator.validate(contract)


def test_unknown_contract_field_fails_closed():
    contract = _contract()
    contract["future_layout"] = "unknown"
    with pytest.raises(ArtifactContractError, match="unsupported fields"):
        ArtifactContractValidator.validate(contract)


def test_malformed_operator_fails_closed():
    contract = _contract()
    contract["operator_name"] = {"name": "unknown"}
    with pytest.raises(ArtifactContractError, match="non-empty strings"):
        ArtifactContractValidator.validate(contract)
