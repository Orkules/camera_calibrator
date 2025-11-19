#!/usr/bin/env python3
"""
PyInstaller wrapper.
"""

from __future__ import annotations
import runpy
import shutil
import sys
from pathlib import Path


def get_app_root() -> Path:
    """Location of bundled assets (templates, default config)."""
    if getattr(sys, "frozen", False):
        bundle_dir = getattr(sys, "_MEIPASS", None)
        if bundle_dir:
            return Path(bundle_dir)
    return Path(__file__).parent


def get_run_root() -> Path:
    """Writable root (next to script during dev, next to exe when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


APP_ROOT = get_app_root()
RUN_ROOT = get_run_root()


def ensure_exists(src: Path, dest: Path) -> None:
    """
    Copy a file or directory from src to dest if dest does not exist yet.
    This keeps user-modified files intact between runs.
    """
    if not src.exists() or dest.exists():
        return

    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def bootstrap_runtime_assets() -> None:
    """Ensure user-writable copies of config and calibration resources exist."""
    ensure_exists(APP_ROOT / "config.yaml", RUN_ROOT / "config.yaml")
    ensure_exists(APP_ROOT / "calibration_files", RUN_ROOT / "calibration_files")
    ensure_exists(APP_ROOT / "luminar_eth_operation.txt", RUN_ROOT / "luminar_eth_operation.txt")


if __name__ == "__main__":
    bootstrap_runtime_assets()
    # Delegate to the real application entry point exactly as `python app.py` would.
    runpy.run_module("app", run_name="__main__")
