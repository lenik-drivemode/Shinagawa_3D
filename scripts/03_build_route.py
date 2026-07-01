#!/usr/bin/env python3
"""Phase 5: Generate route from configured waypoints on the OSM road graph."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from osm3d_poc.config import load_config
from osm3d_poc.logging_config import setup_logging
from osm3d_poc.geo.download_osm import download_or_load_graph
from osm3d_poc.geo.projection import Projector
from osm3d_poc.geo.route_builder import build_route, save_route


def main() -> None:
    p = argparse.ArgumentParser(description="Generate vehicle route from OSM road graph")
    p.add_argument("--config",     default="config/shinagawa_poc.yaml", metavar="PATH")
    p.add_argument("--small-bbox", action="store_true",
                   help="Use smaller debug bounding box")
    p.add_argument("--rebuild",    action="store_true",
                   help="Rebuild route even if output file already exists")
    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger(__name__)

    out_path = Path(cfg["route"]["output_file"])
    if out_path.exists() and not args.rebuild:
        log.info("Route already exists at %s; skipping (use --rebuild to force).", out_path)
        return

    log.info("Loading road graph …")
    G = download_or_load_graph(cfg, small=args.small_bbox)
    log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    projector = Projector(cfg)
    log.info("Projector origin: easting=%.1f  northing=%.1f",
             projector.origin_easting, projector.origin_northing)

    # Use debug_waypoints when running with --small-bbox so all points are inside the graph
    route_cfg = dict(cfg)
    if args.small_bbox and "debug_waypoints" in cfg.get("route", {}):
        route_cfg = {**cfg, "route": {**cfg["route"], "waypoints": cfg["route"]["debug_waypoints"]}}
        log.info("Using debug_waypoints (%d points) for small-bbox run.",
                 len(cfg["route"]["debug_waypoints"]))

    try:
        route_dict = build_route(G, projector, route_cfg)
    except RuntimeError as exc:
        log.error("Route generation failed: %s", exc)
        _save_diagnostics(route_cfg, G, args.small_bbox)
        sys.exit(1)

    route_path = save_route(route_dict, cfg)
    _update_metadata(cfg, route_dict, route_path)
    log.info("=== Phase 5 complete — route saved to %s ===", route_path)


def _save_diagnostics(cfg: dict, G, small: bool) -> None:
    """Save diagnostic JSON on route generation failure (NFR-REL-003)."""
    out_dir = Path(cfg["preprocess"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = {
        "n_nodes":     G.number_of_nodes(),
        "n_edges":     G.number_of_edges(),
        "small_bbox":  small,
        "waypoints":   cfg["route"]["waypoints"],
        "note":        "Shortest-path failed — check graph connectivity and waypoint coverage.",
    }
    diag_path = out_dir / "route_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    logging.getLogger(__name__).info("Diagnostics saved → %s", diag_path)


def _update_metadata(cfg: dict, route_dict: dict, route_path: Path) -> None:
    meta_path = Path(cfg["preprocess"]["output_dir"]) / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    meta.setdefault("route", {})
    meta["route"]["file"]        = str(route_path)
    meta["route"]["length_m"]    = route_dict["length_m"]
    meta["route"]["n_points"]    = len(route_dict["points"])
    meta["route"]["closed_loop"] = route_dict["closed_loop"]
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
