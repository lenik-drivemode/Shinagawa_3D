#!/usr/bin/env python3
"""Shinagawa 3D viewer entry point (PRD §21)."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from osm3d_poc.config import load_config
from osm3d_poc.logging_config import setup_logging


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shinagawa 3D Viewer")
    p.add_argument("--config",     default="config/shinagawa_poc.yaml", metavar="PATH",
                   help="Path to YAML config file")
    p.add_argument("--demo-cube",  action="store_true",
                   help="Render a rotating demo cube (Phase 0 smoke-test)")
    p.add_argument("--no-buildings", action="store_true",
                   help="Skip building rendering for faster load")
    p.add_argument("--wireframe",  action="store_true",
                   help="Render geometry in wireframe mode")
    p.add_argument("--top-down",   action="store_true",
                   help="Start in top-down camera mode")
    p.add_argument("--follow",     action="store_true",
                   help="Start in close-follow camera mode")
    p.add_argument("--long-route", action="store_true",
                   help="Load the long route (~9 km) from route.long_output_file")
    p.add_argument("--speed-kmh",  type=float, default=None, metavar="KMH",
                   help="Override vehicle simulation speed in km/h")
    p.add_argument("--small-bbox", action="store_true",
                   help="Use smaller debug bounding box")
    p.add_argument("--show-stats", action="store_true",
                   help="Print detailed render statistics at startup")
    p.add_argument("--screenshot", metavar="PATH", default=None,
                   help="Save a screenshot to PATH after the first rendered frame, then exit")
    p.add_argument("--debug",      action="store_true",
                   help="Log FPS, camera mode, and route progress every second (debug panel)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.demo_cube:
        from osm3d_poc.render.demo_cube import run_demo_cube
        run_demo_cube()
        return

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger(__name__)

    if args.show_stats:
        out_dir = Path(cfg["preprocess"]["output_dir"])
        for name in ("roads_mesh.npz", "buildings_mesh.npz"):
            p = out_dir / name
            if p.exists():
                import numpy as np
                d = np.load(p)
                log.info("%s: %d verts  %d tris", name,
                         len(d["vertices"]), len(d["indices"]) // 3)

    # Apply CLI overrides to cfg
    if args.long_route:
        cfg.setdefault("route", {})["output_file"] = cfg["route"]["long_output_file"]
    if args.speed_kmh is not None:
        cfg.setdefault("route", {})["default_speed_kmh"] = args.speed_kmh

    from osm3d_poc.render.renderer import run_viewer
    run_viewer(cfg, cli_args=args)


if __name__ == "__main__":
    main()
