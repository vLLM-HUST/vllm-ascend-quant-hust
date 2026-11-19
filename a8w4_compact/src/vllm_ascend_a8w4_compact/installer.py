"""Install the A8W4 MSD compact kernel package into a full vendor OPP.

The bundled artifact is a *partial* custom_transformer package containing only
the rebuilt grouped_matmul_swiglu_quant_v2 kernels. It must be merged into the
user's full in-tree vendor package: putting the partial package first on
ASCEND_CUSTOM_OPP_PATH shadows the other custom ops (notably the GDN ops of
W4A8-GDN models) and breaks engine initialization.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import tarfile
import tempfile
from importlib import resources
from pathlib import Path
from typing import Optional

PACKAGE = "vllm_ascend_a8w4_compact"
BUNDLE = "bundles/a8w4_compact_910b_cann9.1.tar.gz"
DEFAULT_OPP_ROOT = Path.home() / ".cache" / "vllm_ascend_a8w4_compact"
VENDOR_REL = Path("_cann_ops_custom/vendors/custom_transformer")
KDIR_REL = Path("op_impl/ai_core/tbe/kernel/ascend910b/grouped_matmul_swiglu_quant_v2")
KCFG_REL = Path("op_impl/ai_core/tbe/kernel/config/ascend910b/grouped_matmul_swiglu_quant_v2.json")
MIN_FULL_OPS = 10


class InstallError(RuntimeError):
    pass


def bundled_bundle() -> Path:
    ref = resources.files(PACKAGE).joinpath(BUNDLE)
    try:
        return Path(ref)
    except TypeError as exc:  # zipped import fallback
        raise InstallError(f"bundled artifact is not available as a file: {ref}") from exc


def bundled_assets_dir() -> Path:
    ref = resources.files(PACKAGE).joinpath("bundles/assets")
    return Path(ref)


def find_in_tree_vendor() -> Optional[Path]:
    try:
        import vllm_ascend  # noqa: F401
    except Exception:
        return None
    vendor = Path(vllm_ascend.__file__).resolve().parent / VENDOR_REL
    return vendor if (vendor / "version.info").is_file() else None


def list_ops(vendor: Path) -> list[str]:
    root = vendor / "op_impl/ai_core/tbe/kernel/ascend910b"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def plugin_available() -> bool:
    try:
        import vllm_ascend.plugins.a8w4_compact  # noqa: F401
    except Exception:
        return False
    return True


def platform_issues() -> list[str]:
    issues: list[str] = []
    machine = platform.machine().lower()
    if machine not in ("aarch64", "arm64"):
        issues.append(f"host machine is {machine!r}; the bundled kernel is built for aarch64")
    can = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_OPP_PATH") or ""
    if can and "9.1" not in can:
        issues.append(f"ASCEND_HOME_PATH/ASCEND_OPP_PATH does not look like CANN 9.1: {can}")
    if not can:
        issues.append("ASCEND_HOME_PATH/ASCEND_OPP_PATH is not set; the runtime must use CANN 9.1")
    return issues


def _extract_bundle(dest: Path) -> Path:
    bundle = bundled_bundle()
    if not bundle.is_file():
        raise InstallError(f"bundled artifact missing: {bundle}")
    with tarfile.open(bundle, "r:gz") as tf:
        tf.extractall(dest, filter="data")
    partial = dest / "custom_transformer"
    if not (partial / KDIR_REL).is_dir():
        raise InstallError(f"bundle does not contain {KDIR_REL}")
    return partial


def install(
    vendor: Optional[Path],
    opp_root: Path = DEFAULT_OPP_ROOT,
    *,
    force: bool = False,
    skip_platform_check: bool = False,
) -> Path:
    if not skip_platform_check:
        issues = platform_issues()
        if issues:
            raise InstallError("platform check failed: " + "; ".join(issues))
    if vendor is None:
        vendor = find_in_tree_vendor()
    if vendor is None:
        raise InstallError(
            "cannot locate the in-tree custom_transformer vendor; "
            "install vllm-ascend first or pass --vendor <path>"
        )
    vendor = Path(vendor).expanduser().resolve()
    if not (vendor / "version.info").is_file():
        raise InstallError(f"not a custom_transformer vendor package: {vendor}")
    ops = list_ops(vendor)
    if len(ops) < MIN_FULL_OPS:
        raise InstallError(
            f"vendor has only {len(ops)} ops; a partial package shadows the in-tree vendor "
            "and breaks other custom ops (GDN on W4A8-GDN models). Pass the full in-tree vendor."
        )

    opp_root = Path(opp_root).expanduser().resolve()
    out = opp_root / "custom_transformer"
    if out.exists():
        if not force:
            raise InstallError(f"{out} already exists; pass --force to replace it")
        shutil.rmtree(out)

    with tempfile.TemporaryDirectory(prefix="a8w4-compact-") as tmp:
        partial = _extract_bundle(Path(tmp))
        opp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(vendor, out)
        for rel in (KDIR_REL,):
            target = out / rel
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(partial / rel, target)
        cfg = partial / KCFG_REL
        if cfg.is_file():
            dst = out / KCFG_REL
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cfg, dst)

    _verify(out)
    return out


def _verify(out: Path) -> None:
    if not (out / "version.info").is_file():
        raise InstallError(f"merge failed: missing {out / 'version.info'}")
    if not (out / KDIR_REL).is_dir():
        raise InstallError(f"merge failed: missing {out / KDIR_REL}")
    ops = list_ops(out)
    if len(ops) < MIN_FULL_OPS:
        raise InstallError(f"merge failed: only {len(ops)} ops in {out}")
    # tiling lib must come from the in-tree vendor (this patch changes no tiling)
    tiling = out / "op_impl/ai_core/tbe/op_tiling/liboptiling.so"
    if not tiling.is_file():
        raise InstallError(f"merge failed: missing {tiling}")


def status(opp_root: Path = DEFAULT_OPP_ROOT) -> dict:
    out = Path(opp_root).expanduser() / "custom_transformer"
    return {
        "plugin_available": plugin_available(),
        "opp_root": str(Path(opp_root).expanduser()),
        "opp_installed": (out / "version.info").is_file() and (out / KDIR_REL).is_dir(),
        "opp_ops": list_ops(out) if out.is_dir() else [],
        "bundle": str(bundled_bundle()),
        "platform_issues": platform_issues(),
    }


def env_lines(opp_root: Path = DEFAULT_OPP_ROOT) -> list[str]:
    out = (Path(opp_root).expanduser() / "custom_transformer").resolve()
    lines = ["export VLLM_ASCEND_A8W4_COMPACT=1"]
    if out != (DEFAULT_OPP_ROOT / "custom_transformer").resolve():
        lines.append(f"export VLLM_ASCEND_A8W4_COMPACT_OPP={out}")
    return lines
