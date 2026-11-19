"""Command line interface for vllm-ascend-a8w4-compact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .installer import (
    DEFAULT_OPP_ROOT,
    InstallError,
    bundled_assets_dir,
    env_lines,
    install,
    status,
)


def _print_env(lines: list[str]) -> None:
    print("Add these to the environment that starts vLLM:")
    for line in lines:
        print("  " + line)
    print("Then restart the vLLM service. The a8w4_compact plugin in vllm-ascend")
    print("prepends the package to ASCEND_CUSTOM_OPP_PATH when the switch is set.")


def cmd_install(args: argparse.Namespace) -> int:
    try:
        out = install(
            args.vendor,
            args.opp_dir,
            force=args.force,
            skip_platform_check=args.skip_platform_check,
        )
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"installed full vendor package: {out}")
    print("The in-tree liboptiling.so is kept; only the GMM v2 kernel dir was replaced.")
    _print_env(env_lines(args.opp_dir))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    st = status(args.opp_dir)
    print(f"a8w4-compact {__version__}")
    print(f"plugin available: {st['plugin_available']}")
    print(f"opp root:         {st['opp_root']}")
    print(f"opp installed:    {st['opp_installed']} ({len(st['opp_ops'])} ops)")
    print(f"bundle:           {st['bundle']}")
    issues = st["platform_issues"]
    print(f"platform issues:  {len(issues)}")
    for issue in issues:
        print(f"  - {issue}")
    ok = st["plugin_available"] and st["opp_installed"] and not issues
    print("ready" if ok else "not ready")
    return 0 if ok else 1


def cmd_env(args: argparse.Namespace) -> int:
    for line in env_lines(args.opp_dir):
        print(line)
    return 0


def cmd_assets(_: argparse.Namespace) -> int:
    print(bundled_assets_dir())
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="a8w4-compact",
        description="Install the A8W4 MSD compact kernel package for vllm-ascend.",
    )
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("install", help="merge the kernel into a full vendor OPP")
    p.add_argument("--vendor", type=Path, default=None,
                   help="in-tree custom_transformer vendor (auto-detected from vllm_ascend)")
    p.add_argument("--opp-dir", type=Path, default=DEFAULT_OPP_ROOT,
                   help=f"destination OPP root (default: {DEFAULT_OPP_ROOT})")
    p.add_argument("--force", action="store_true", help="replace an existing installation")
    p.add_argument("--skip-platform-check", action="store_true",
                   help="install even if the host/CANN platform check fails")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("check", help="report plugin, installation and platform status")
    p.add_argument("--opp-dir", type=Path, default=DEFAULT_OPP_ROOT)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("env", help="print the environment variables to set")
    p.add_argument("--opp-dir", type=Path, default=DEFAULT_OPP_ROOT)
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("assets", help="print the directory with patches, sources and docs")
    p.set_defaults(func=cmd_assets)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
