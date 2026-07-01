#!/usr/bin/env python3
"""Run the full Shinagawa 3D pipeline in order: download → preprocess → route → viewer.

Equivalent to running each numbered script manually. Stops on first failure.

Examples:
    python scripts/run_all.py
    python scripts/run_all.py --rebuild --small-bbox
    python scripts/run_all.py --no-viewer           # preprocess only, skip viewer
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

# (script filename, human-readable label)
_STEPS: Sequence[Tuple[str, str]] = (
    ("01_download_osm.py",      "Step 1/4: Download OSM data"),
    ("02_preprocess_assets.py", "Step 2/4: Preprocess assets"),
    ("03_build_route.py",       "Step 3/4: Build route"),
    ("04_run_viewer.py",        "Step 4/4: Launch viewer"),
)

_REBUILD_SCRIPTS = {"02_preprocess_assets.py", "03_build_route.py"}


def build_cmds(
    config: str,
    small_bbox: bool,
    rebuild: bool,
    no_viewer: bool,
    scripts_dir: Path,
) -> List[Tuple[str, List[str]]]:
    """Return list of (script_name, command_list) for each pipeline step.

    Exported for testing — does not call subprocess.
    """
    cmds = []
    for script, _ in _STEPS:
        if no_viewer and script == "04_run_viewer.py":
            continue
        cmd = [sys.executable, str(scripts_dir / script), "--config", config]
        if small_bbox:
            cmd.append("--small-bbox")
        if rebuild and script in _REBUILD_SCRIPTS:
            cmd.append("--rebuild")
        cmds.append((script, cmd))
    return cmds


def run_pipeline(
    config: str,
    small_bbox: bool = False,
    rebuild: bool = False,
    no_viewer: bool = False,
    scripts_dir: Path = None,
) -> bool:
    """Execute the pipeline. Return True on success, False on first failure."""
    if scripts_dir is None:
        scripts_dir = Path(__file__).parent.resolve()

    steps = build_cmds(config, small_bbox, rebuild, no_viewer, scripts_dir)
    for script, cmd in steps:
        label = next(lbl for s, lbl in _STEPS if s == script)
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"\nERROR: {script} exited with code {result.returncode}.",
                file=sys.stderr,
            )
            print("Fix the error above, then re-run (use --rebuild if needed).", file=sys.stderr)
            return False
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full Shinagawa 3D pipeline (steps 01–04).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config",     default="config/shinagawa_poc.yaml", metavar="PATH",
                   help="Path to YAML config file (passed to all steps)")
    p.add_argument("--small-bbox", action="store_true",
                   help="Use small debug bounding box (passed to all steps)")
    p.add_argument("--rebuild",    action="store_true",
                   help="Force rebuild of cached assets (passed to preprocess + route steps)")
    p.add_argument("--no-viewer",  action="store_true",
                   help="Run preprocessing only; skip launching the viewer")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ok = run_pipeline(
        config=args.config,
        small_bbox=args.small_bbox,
        rebuild=args.rebuild,
        no_viewer=args.no_viewer,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
