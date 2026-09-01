"""Static identity and artifact checks for the Ascend quant runtime extension."""

from .artifact import ArtifactContractError, ArtifactContractValidator

EXTENSION_ID = "org.vllm-hust.ascend-quant-runtime"
__version__ = "0.1.0.dev0"

__all__ = [
    "EXTENSION_ID",
    "ArtifactContractError",
    "ArtifactContractValidator",
    "__version__",
]
