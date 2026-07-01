#!/usr/bin/env python3
"""Phase 3/4: Preprocess OSM data into render-ready mesh assets."""
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
from osm3d_poc.geo.preprocess_roads import build_road_mesh, save_road_mesh


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess OSM data into render-ready assets")
    p.add_argument("--config", default="config/shinagawa_poc.yaml", metavar="PATH")
    p.add_argument("--small-bbox", action="store_true",
                   help="Use smaller debug bounding box")
    p.add_argument("--skip-buildings", action="store_true",
                   help="Skip building preprocessing (not yet implemented; future Phase 4)")
    p.add_argument("--rebuild", action="store_true",
                   help="Force rebuild even if output files already exist")
    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger(__name__)

    out_dir = Path(cfg["preprocess"]["output_dir"])
    npz_path = out_dir / "roads_mesh.npz"
    if npz_path.exists() and not args.rebuild:
        log.info("roads_mesh.npz already exists; skipping (use --rebuild to force).")
    else:
        log.info("Loading road graph …")
        G = download_or_load_graph(cfg, small=args.small_bbox)

        projector = Projector(cfg)
        log.info(
            "Projector origin: easting=%.1f  northing=%.1f",
            projector.origin_easting, projector.origin_northing,
        )

        vertices, indices, edge_meta = build_road_mesh(G, projector, cfg)
        save_road_mesh(vertices, indices, edge_meta, cfg)

        _update_metadata(cfg, projector, vertices, indices)
        log.info(
            "Road mesh: %d vertices, %d triangles",
            len(vertices), len(indices) // 3,
        )

    if not args.skip_buildings:
        log.info("Building preprocessing not yet implemented (coming in Phase 4).")

    log.info("=== Phase 3 complete ===")


def _update_metadata(cfg: dict, projector, vertices, indices) -> None:
    out_dir = Path(cfg["preprocess"]["output_dir"])
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"counts": {}, "projection": {}}

    meta.setdefault("counts", {})
    meta["counts"]["road_vertices"]  = int(len(vertices))
    meta["counts"]["road_triangles"] = int(len(indices) // 3)
    meta.setdefault("projection", {})
    meta["projection"]["origin_easting"]  = projector.origin_easting
    meta["projection"]["origin_northing"] = projector.origin_northing

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
