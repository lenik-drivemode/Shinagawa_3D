"""OSM data acquisition: road graph and building footprints."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------

def get_bbox(cfg: dict, small: bool = False) -> Tuple[float, float, float, float]:
    """Return (north, south, east, west) from config."""
    key = "debug_bbox" if small else "bbox"
    b = cfg["area"][key]
    return b["north"], b["south"], b["east"], b["west"]


def _bbox_tag(small: bool) -> str:
    return "debug" if small else "full"


def _graph_path(cache_dir: Path, small: bool) -> Path:
    return cache_dir / f"road_graph_{_bbox_tag(small)}.graphml"


def _buildings_path(cache_dir: Path, small: bool) -> Path:
    return cache_dir / f"buildings_{_bbox_tag(small)}.gpkg"


# ---------------------------------------------------------------------------
# Download / load
# ---------------------------------------------------------------------------

def download_or_load_graph(cfg: dict, small: bool = False):
    """Return an osmnx/networkx MultiDiGraph, using cache when available."""
    import osmnx as ox

    cache_dir = Path(cfg["osm"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_path = _graph_path(cache_dir, small)

    if cfg["osm"]["use_cache"] and graph_path.exists():
        log.info("Loading road graph from cache: %s", graph_path)
        G = ox.load_graphml(graph_path)
        log.info("Road graph: %d nodes, %d edges", len(G.nodes), len(G.edges))
        return G

    north, south, east, west = get_bbox(cfg, small)
    log.info(
        "Downloading road graph  N=%.4f S=%.4f E=%.4f W=%.4f  network_type=%s",
        north, south, east, west, cfg["osm"]["network_type"],
    )
    # osmnx 2.x: bbox=(left, bottom, right, top) = (west, south, east, north)
    G = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type=cfg["osm"]["network_type"],
    )
    log.info("Road graph: %d nodes, %d edges", len(G.nodes), len(G.edges))
    ox.save_graphml(G, filepath=graph_path)
    log.info("Saved road graph → %s", graph_path)
    return G


def download_or_load_buildings(cfg: dict, small: bool = False):
    """Return a GeoDataFrame of building footprints, using cache when available."""
    import geopandas as gpd
    import osmnx as ox

    cache_dir = Path(cfg["osm"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    bldg_path = _buildings_path(cache_dir, small)

    if cfg["osm"]["use_cache"] and bldg_path.exists():
        log.info("Loading buildings from cache: %s", bldg_path)
        gdf = gpd.read_file(bldg_path)
        log.info("Buildings: %d footprints", len(gdf))
        return gdf

    north, south, east, west = get_bbox(cfg, small)
    log.info("Downloading buildings  N=%.4f S=%.4f E=%.4f W=%.4f", north, south, east, west)
    # osmnx 2.x: bbox=(left, bottom, right, top) = (west, south, east, north)
    gdf = ox.features_from_bbox(
        bbox=(west, south, east, north),
        tags=cfg["osm"]["building_tags"],
    )
    log.info("Buildings: %d features", len(gdf))
    gdf.to_file(bldg_path, driver="GPKG")
    log.info("Saved buildings → %s", bldg_path)
    return gdf


# ---------------------------------------------------------------------------
# Initial metadata (filled in further by later phases)
# ---------------------------------------------------------------------------

def save_initial_metadata(
    cfg: dict,
    G: Any,
    buildings_gdf: Any,
    small: bool = False,
) -> Path:
    """Write partial metadata.json to the processed output directory.

    Fields that require preprocessing (vertices, triangles, route length)
    are left as null and filled in by later phases.
    """
    out_dir = Path(cfg["preprocess"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"

    north, south, east, west = get_bbox(cfg, small)

    metadata: dict = {
        "schema_version": SCHEMA_VERSION,
        "area_name": cfg["area"]["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "small_bbox": small,
        "bbox": {"north": north, "south": south, "east": east, "west": west},
        "projection": {
            "source_epsg": cfg["projection"]["source_epsg"],
            "target_epsg": cfg["projection"]["target_epsg"],
            "origin_lat": cfg["area"]["center"]["lat"],
            "origin_lon": cfg["area"]["center"]["lon"],
            "origin_easting": None,
            "origin_northing": None,
        },
        "counts": {
            "graph_nodes": len(G.nodes),
            "graph_edges": len(G.edges),
            "buildings": len(buildings_gdf),
            "road_vertices": None,
            "road_triangles": None,
            "building_vertices": None,
            "building_triangles": None,
            "skipped_buildings": None,
        },
        "route": {
            "file": cfg["route"]["output_file"],
            "length_m": None,
            "closed_loop": None,
        },
        "preprocessing_version": SCHEMA_VERSION,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved initial metadata → %s", meta_path)
    return meta_path
