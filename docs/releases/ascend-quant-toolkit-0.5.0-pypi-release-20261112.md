# ascend-quant-toolkit 0.5.0 PyPI release record (2026-11-12)

PyPI-only release record for `ascend-quant-toolkit` 0.5.0. This records the
exact published bytes; it is not a performance or accuracy claim.

## Object

- Distribution: `ascend-quant-toolkit` 0.5.0.
- PyPI project: https://pypi.org/project/ascend-quant-toolkit/0.5.0/
- Source baseline in this repository: `feat/toolkit-0.5.0-qwen35-w4a8`
  @ `b93c073a60a1e6b28dc29c68b56358b8c8c27e88` (`__version__ == "0.5.0"`).
- **No GitHub Release for 0.5.0 was found.** 0.5.0 is distributed through PyPI
  only; do not cite a GitHub Release asset for it.

## Published bytes (PyPI JSON, checked 2026-11-17)

| file | size (bytes) | sha256 | uploaded (UTC) |
|---|---:|---|---|
| `ascend_quant_toolkit-0.5.0-py3-none-any.whl` | 48,923 | `ad3b320f8cbcabd4527eba0b7180808be7190cc0b4b1bb18636232cea3f6657f` | 2026-11-12T10:00:44Z |
| `ascend_quant_toolkit-0.5.0.tar.gz` | 66,144 | `ea065055cabfa3e87c89584068a1953311c547ee092fea9f255fa0df0a86381f` | 2026-11-12T10:00:45Z |

## Local build is not the PyPI byte

The local checkout contains `dist/ascend_quant_toolkit-0.5.0-py3-none-any.whl`
(48,217 B, sha256 `9b55d3d12c408ae0496fa5e69c33552823e6afdabe25b80d7e8c43eeb66d295b`).
Its hash differs from the PyPI wheel, so this record does not claim the PyPI
wheel was built from `b93c073`; the source baseline and the published artifact
are recorded separately.

## Recipe / checkpoint mapping

- Recipe `qwen35-35b-a3b-w4a8-gdnw8a8`
  sha256 `0bcef0f56ddebea8f34841127e6ac5b369ffee20853e5a7d148c344387fd9384`.
- Checkpoint `Qwen3.5-35B-A3B-W8A8-NoRot`: 6 shards, 24,047,130,272 B;
  shard hashes in `docs/artifacts/qwen35-w4a8-gdnw8a8-tp1-20261023.md`.
- `verify-artifact` result: `valid: true` (2026-10-28).
- 0.5.0 also keeps the `qwen25-w8a8` recipe.

## Limitations

- The runtime counterpart is unresolved: the 0.4.2 release record names
  `vllm-ascend-quant-ext` 0.5.0, while the 0.5.0 toolkit README says the runtime
  extension is frozen at 0.4.1a4. This record does not choose a side.
- `prepare-runtime` / `export-contract` reject this mixed-precision artifact by
  design; artifact contract 1.1.0 is single-scheme.
- No GitHub Release/asset for 0.5.0; only the PyPI bytes above are fixed.
