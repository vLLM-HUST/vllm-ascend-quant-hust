"""Command line interface for vllm-ascend-gdn-decode-fusion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .installer import InstallError, bundled_assets_dir, install, revert, status


def cmd_install(args: argparse.Namespace) -> int:
    try:
        tree = install(args.vllm_ascend, force=args.force)
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"installed GDN decode fusion into: {tree}")
    print("Restart the vLLM service so the patched modules are imported.")
    return 0


def cmd_revert(args: argparse.Namespace) -> int:
    try:
        tree = revert(args.vllm_ascend)
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"reverted GDN decode fusion in: {tree}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    st = status(args.vllm_ascend)
    print(f"gdn-decode-fusion {__version__}")
    print(f"tree:        {st['tree']}")
    print(f"markers:     {st['markers']}")
    print(f"added file:  {st['added_file']}")
    print(f"installed:   {st['installed']}")
    print(f"backup:      {st['backup']}")
    print(f"patch:       {st['patch']}")
    if "error" in st:
        print(f"error:       {st['error']}")
        return 1
    print("ready" if st["installed"] else "not installed")
    return 0 if st["installed"] else 1


def cmd_assets(_: argparse.Namespace) -> int:
    print(bundled_assets_dir())
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gdn-decode-fusion",
        description="Install the GDN decode-path fusion patch for vllm-ascend.",
    )
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("install", help="apply the patch to a vllm-ascend checkout")
    p.add_argument("--vllm-ascend", type=Path, default=None,
                   help="path to the vllm-ascend checkout (auto-detected from the import)")
    p.add_argument("--force", action="store_true",
                   help="revert an existing installation and apply the patch again")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("revert", help="restore the files touched by the patch")
    p.add_argument("--vllm-ascend", type=Path, default=None)
    p.set_defaults(func=cmd_revert)

    p = sub.add_parser("check", help="report the patch state of a checkout")
    p.add_argument("--vllm-ascend", type=Path, default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("assets", help="print the directory with the patch, docs and validation")
    p.set_defaults(func=cmd_assets)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
