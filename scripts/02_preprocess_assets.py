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
from osm3d_poc.geo.download_osm import download_or_load_buildings, download_or_load_graph
from osm3d_poc.geo.projection import Projector
from osm3d_poc.geo.preprocess_roads import build_road_mesh, save_road_mesh
from osm3d_poc.geo.preprocess_buildings import build_building_mesh, save_building_mesh


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess OSM data into render-ready assets")
    p.add_argument("--config", default="config/shinagawa_poc.yaml", metavar="PATH")
    p.add_argument("--small-bbox", action="store_true",
                   help="Use smaller debug bounding box")
    p.add_argument("--skip-buildings", action="store_true",
                   help="Skip building preprocessing")
    p.add_argument("--rebuild", action="store_true",
                   help="Force rebuild even if output files already exist")
    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger(__name__)

    out_dir = Path(cfg["preprocess"]["output_dir"])
    projector = None

    # --- Roads ---
    roads_npz = out_dir / "roads_mesh.npz"
    if roads_npz.exists() and not args.rebuild:
        log.info("roads_mesh.npz already exists; skipping (use --rebuild to force).")
    else:
        log.info("Loading road graph …")
        G = download_or_load_graph(cfg, small=args.small_bbox)
        projector = projector or Projector(cfg)
        log.info("Projector origin: easting=%.1f  northing=%.1f",
                 projector.origin_easting, projector.origin_northing)
        road_verts, road_idxs, road_meta = build_road_mesh(G, projector, cfg)
        save_road_mesh(road_verts, road_idxs, road_meta, cfg)
        log.info("Road mesh: %d vertices, %d triangles",
                 len(road_verts), len(road_idxs) // 3)
        _update_road_metadata(cfg, projector, road_verts, road_idxs)

    # --- Buildings ---
    if args.skip_buildings:
        log.info("Building preprocessing skipped (--skip-buildings).")
    else:
        bldg_npz = out_dir / "buildings_mesh.npz"
        if bldg_npz.exists() and not args.rebuild:
            log.info("buildings_mesh.npz already exists; skipping (use --rebuild to force).")
        else:
            log.info("Loading building footprints …")
            gdf = download_or_load_buildings(cfg, small=args.small_bbox)
            projector = projector or Projector(cfg)
            bldg_verts, bldg_idxs, bldg_meta = build_building_mesh(gdf, projector, cfg)
            save_building_mesh(bldg_verts, bldg_idxs, bldg_meta, cfg)
            log.info("Building mesh: %d vertices, %d triangles",
                     len(bldg_verts), len(bldg_idxs) // 3)
            _update_building_metadata(cfg, bldg_verts, bldg_idxs, len(bldg_meta))

    log.info("=== Phase 3/4 complete ===")


def _load_metadata(cfg: dict) -> dict:
    meta_path = Path(cfg["preprocess"]["output_dir"]) / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"counts": {}, "projection": {}}


def _write_metadata(cfg: dict, meta: dict) -> None:
    meta_path = Path(cfg["preprocess"]["output_dir"]) / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _update_road_metadata(cfg, projector, vertices, indices) -> None:
    meta = _load_metadata(cfg)
    meta.setdefault("counts", {})
    meta["counts"]["road_vertices"]  = int(len(vertices))
    meta["counts"]["road_triangles"] = int(len(indices) // 3)
    meta.setdefault("projection", {})
    meta["projection"]["origin_easting"]  = projector.origin_easting
    meta["projection"]["origin_northing"] = projector.origin_northing
    _write_metadata(cfg, meta)


def _update_building_metadata(cfg, vertices, indices, n_buildings) -> None:
    meta = _load_metadata(cfg)
    meta.setdefault("counts", {})
    meta["counts"]["buildings"]          = n_buildings
    meta["counts"]["building_vertices"]  = int(len(vertices))
    meta["counts"]["building_triangles"] = int(len(indices) // 3)
    _write_metadata(cfg, meta)


if __name__ == "__main__":
    main()
