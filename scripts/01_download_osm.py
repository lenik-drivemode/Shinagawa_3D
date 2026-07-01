#!/usr/bin/env python3
"""Phase 1: Download and cache OSM road graph and building footprints."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from osm3d_poc.config import load_config
from osm3d_poc.logging_config import setup_logging
from osm3d_poc.geo.download_osm import (
    download_or_load_buildings,
    download_or_load_graph,
    get_bbox,
    save_initial_metadata,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Download OSM data for Shinagawa area")
    p.add_argument("--config", default="config/shinagawa_poc.yaml", metavar="PATH")
    p.add_argument("--small-bbox", action="store_true",
                   help="Use smaller debug bounding box")
    p.add_argument("--no-cache", action="store_true",
                   help="Force re-download even if cache files exist")
    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger(__name__)

    if args.no_cache:
        cfg["osm"]["use_cache"] = False

    north, south, east, west = get_bbox(cfg, small=args.small_bbox)
    log.info("Bounding box: N=%.4f S=%.4f E=%.4f W=%.4f", north, south, east, west)

    try:
        G = download_or_load_graph(cfg, small=args.small_bbox)
    except Exception as exc:
        log.error("Failed to acquire road graph: %s", exc)
        log.error("Check network connection or use cached data (remove --no-cache).")
        sys.exit(1)

    try:
        buildings_gdf = download_or_load_buildings(cfg, small=args.small_bbox)
    except Exception as exc:
        log.error("Failed to acquire building footprints: %s", exc)
        log.error("Check network connection or use cached data (remove --no-cache).")
        sys.exit(1)

    meta_path = save_initial_metadata(cfg, G, buildings_gdf, small=args.small_bbox)

    log.info("=== Phase 1 complete ===")
    log.info("  Road graph : %d nodes, %d edges", len(G.nodes), len(G.edges))
    log.info("  Buildings  : %d footprints", len(buildings_gdf))
    log.info("  Metadata   : %s", meta_path)


if __name__ == "__main__":
    main()
