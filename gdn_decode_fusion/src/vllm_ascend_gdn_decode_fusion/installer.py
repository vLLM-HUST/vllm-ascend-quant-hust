"""Install the GDN decode-path fusion patch into a vllm-ascend checkout.

The patch is pure Python/Triton: it adds ``gdn_decode_fused.py`` and edits
``ops/gdn.py`` and ``ops/triton/fused_gdn_gating.py``.  Because the edits are
version-sensitive, the installer applies a context patch (``git apply`` when the
tree is a git checkout, ``patch`` otherwise), keeps a backup of the touched
files, and aborts without touching anything when the context does not match.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Optional

PACKAGE = "vllm_ascend_gdn_decode_fusion"
PATCH_REL = "bundles/assets/patches/gdn-decode-fusion.patch"
BACKUP_DIRNAME = ".gdn_decode_fusion"
MARKER = "_rearrange_mixed_qkv_fused"
TOUCHED = (
    Path("vllm_ascend/ops/gdn.py"),
    Path("vllm_ascend/ops/triton/fused_gdn_gating.py"),
)
ADDED = Path("vllm_ascend/ops/triton/gdn_decode_fused.py")


class InstallError(RuntimeError):
    pass


def bundled_patch() -> Path:
    ref = resources.files(PACKAGE).joinpath(PATCH_REL)
    try:
        return Path(ref)
    except TypeError as exc:  # zipped import fallback
        raise InstallError(f"bundled patch is not available as a file: {ref}") from exc


def bundled_assets_dir() -> Path:
    return Path(resources.files(PACKAGE).joinpath("bundles/assets"))


def _as_tree(path: Path) -> Path:
    """Accept either the checkout root or the ``vllm_ascend`` package dir."""
    path = Path(path).expanduser().resolve()
    if (path / "vllm_ascend/ops/gdn.py").is_file():
        return path
    if path.name == "vllm_ascend" and (path / "ops/gdn.py").is_file():
        return path.parent
    return path


def find_tree() -> Optional[Path]:
    try:
        import vllm_ascend  # noqa: F401
    except Exception:
        return None
    return _as_tree(Path(vllm_ascend.__file__).resolve().parent)


def validate_tree(tree: Path) -> None:
    for rel in TOUCHED:
        if not (tree / rel).is_file():
            raise InstallError(f"not a vllm-ascend checkout (missing {tree / rel})")
    gdn = (tree / TOUCHED[0]).read_text(encoding="utf-8", errors="replace")
    if "class AscendGatedDeltaNetAttention" not in gdn:
        raise InstallError(
            f"{tree / TOUCHED[0]} does not look like the Ascend GDN implementation"
        )


def is_installed(tree: Path) -> bool:
    gdn_path = tree / TOUCHED[0]
    if not gdn_path.is_file():
        return False
    text = gdn_path.read_text(encoding="utf-8", errors="replace")
    return MARKER in text and (tree / ADDED).is_file()


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _apply_patch(tree: Path, patch: Path, reverse: bool = False) -> None:
    git = shutil.which("git")
    if git and (tree / ".git").exists():
        extra = ["--reverse"] if reverse else []
        check = _run([git, "apply", "--check", *extra, str(patch)], tree)
        if check.returncode != 0:
            raise InstallError(
                "git apply --check failed (the checkout does not match the patch context): "
                + (check.stderr or check.stdout).strip()
            )
        res = _run([git, "apply", *extra, str(patch)], tree)
        if res.returncode != 0:
            raise InstallError("git apply failed: " + (res.stderr or res.stdout).strip())
        return

    patch_bin = shutil.which("patch")
    if not patch_bin:
        raise InstallError("neither git nor patch is available to apply the bundled patch")
    extra = ["--reverse"] if reverse else ["--forward"]
    dry = _run([patch_bin, "-p1", "--dry-run", *extra, "-i", str(patch)], tree)
    if dry.returncode != 0:
        raise InstallError(
            "patch --dry-run failed (the checkout does not match the patch context):\n"
            + (dry.stdout + dry.stderr).strip()
        )
    res = _run([patch_bin, "-p1", "--no-backup-if-mismatch", *extra, "-i", str(patch)], tree)
    if res.returncode != 0:
        raise InstallError("patch failed:\n" + (res.stdout + res.stderr).strip())


def _clear_pycache(tree: Path) -> None:
    for rel in (Path("vllm_ascend/ops"), Path("vllm_ascend/ops/triton")):
        cache = tree / rel / "__pycache__"
        if not cache.is_dir():
            continue
        for pattern in ("gdn*.pyc", "fused_gdn_gating*.pyc"):
            for item in cache.glob(pattern):
                item.unlink(missing_ok=True)


def _backup(tree: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = tree / BACKUP_DIRNAME / f"backup-{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for rel in TOUCHED:
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tree / rel, dst)
    return backup


def latest_backup(tree: Path) -> Optional[Path]:
    root = tree / BACKUP_DIRNAME
    if not root.is_dir():
        return None
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    return candidates[-1] if candidates else None


def _restore(tree: Path, backup: Path) -> None:
    for rel in TOUCHED:
        src = backup / rel
        if src.is_file():
            dst = tree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    added = tree / ADDED
    if added.is_file():
        added.unlink()
    _clear_pycache(tree)


def install(tree: Optional[Path] = None, *, force: bool = False) -> Path:
    tree = _as_tree(tree) if tree is not None else find_tree()
    if tree is None:
        raise InstallError(
            "cannot locate vllm-ascend; pass --vllm-ascend /path/to/vllm-ascend"
        )
    validate_tree(tree)
    if is_installed(tree):
        if not force:
            return tree
        revert(tree)
    patch = bundled_patch()
    if not patch.is_file():
        raise InstallError(f"bundled patch missing: {patch}")
    backup = _backup(tree)
    try:
        _apply_patch(tree, patch)
        if not is_installed(tree):
            raise InstallError("patch applied but the expected markers are missing")
    except Exception:
        _restore(tree, backup)
        raise
    _clear_pycache(tree)
    return tree


def revert(tree: Optional[Path] = None) -> Path:
    tree = _as_tree(tree) if tree is not None else find_tree()
    if tree is None:
        raise InstallError(
            "cannot locate vllm-ascend; pass --vllm-ascend /path/to/vllm-ascend"
        )
    validate_tree(tree)
    backup = latest_backup(tree)
    if backup is not None:
        _restore(tree, backup)
        return tree
    if not is_installed(tree):
        raise InstallError("the patch does not look installed and no backup exists")
    _apply_patch(tree, bundled_patch(), reverse=True)
    if is_installed(tree):
        raise InstallError("reverse patch left the markers in place")
    _clear_pycache(tree)
    return tree


def status(tree: Optional[Path] = None) -> dict:
    resolved = _as_tree(tree) if tree is not None else find_tree()
    info: dict = {
        "tree": str(resolved) if resolved else None,
        "installed": False,
        "backup": None,
        "patch": str(bundled_patch()),
        "markers": False,
        "added_file": False,
    }
    if resolved is None:
        return info
    try:
        validate_tree(resolved)
    except InstallError as exc:
        info["error"] = str(exc)
        return info
    info["installed"] = is_installed(resolved)
    info["markers"] = MARKER in (resolved / TOUCHED[0]).read_text(
        encoding="utf-8", errors="replace"
    )
    info["added_file"] = (resolved / ADDED).is_file()
    backup = latest_backup(resolved)
    info["backup"] = str(backup) if backup else None
    return info
