"""Tests for Phase 1 data acquisition logic. No network access required."""
import json
from pathlib import Path

import pytest

from osm3d_poc.config import load_config
from osm3d_poc.geo.download_osm import (
    _bbox_tag,
    _buildings_path,
    _graph_path,
    get_bbox,
    save_initial_metadata,
)

_CFG_PATH = Path(__file__).parent.parent / "config" / "shinagawa_poc.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(_CFG_PATH)


# ---------------------------------------------------------------------------
# get_bbox
# ---------------------------------------------------------------------------

def test_get_bbox_full_values(cfg):
    north, south, east, west = get_bbox(cfg, small=False)
    assert north == pytest.approx(35.6420)
    assert south == pytest.approx(35.6150)
    assert east  == pytest.approx(139.7600)
    assert west  == pytest.approx(139.7150)


def test_get_bbox_full_is_valid(cfg):
    north, south, east, west = get_bbox(cfg, small=False)
    assert north > south
    assert east > west


def test_get_bbox_small_values(cfg):
    north, south, east, west = get_bbox(cfg, small=True)
    assert north == pytest.approx(35.6350)
    assert south == pytest.approx(35.6220)
    assert east  == pytest.approx(139.7500)
    assert west  == pytest.approx(139.7270)


def test_get_bbox_small_is_strictly_inside_full(cfg):
    fn, fs, fe, fw = get_bbox(cfg, small=False)
    dn, ds, de, dw = get_bbox(cfg, small=True)
    assert dn < fn
    assert ds > fs
    assert de < fe
    assert dw > fw


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------

def test_bbox_tag_values():
    assert _bbox_tag(small=False) == "full"
    assert _bbox_tag(small=True)  == "debug"


def test_graph_and_buildings_paths_are_distinct(cfg):
    d = Path(cfg["osm"]["cache_dir"])
    assert _graph_path(d, False) != _buildings_path(d, False)
    assert _graph_path(d, True)  != _buildings_path(d, True)


def test_full_and_debug_paths_are_distinct(cfg):
    d = Path(cfg["osm"]["cache_dir"])
    assert _graph_path(d, False)     != _graph_path(d, True)
    assert _buildings_path(d, False) != _buildings_path(d, True)


def test_cache_path_extensions(cfg):
    d = Path(cfg["osm"]["cache_dir"])
    assert _graph_path(d, False).suffix     == ".graphml"
    assert _buildings_path(d, False).suffix == ".gpkg"


# ---------------------------------------------------------------------------
# save_initial_metadata (uses stub objects, no network)
# ---------------------------------------------------------------------------

def _make_stub_graph():
    networkx = pytest.importorskip("networkx")
    G = networkx.MultiDiGraph()
    G.add_node(1, x=139.7388, y=35.6285)
    G.add_node(2, x=139.7390, y=35.6286)
    G.add_node(3, x=139.7392, y=35.6287)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 1)
    return G


def _make_stub_buildings():
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),
    ]
    return gpd.GeoDataFrame({"geometry": polys}, crs="EPSG:4326")


def test_metadata_file_is_created(tmp_path, cfg):
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    G = _make_stub_graph()
    gdf = _make_stub_buildings()
    meta_path = save_initial_metadata(test_cfg, G, gdf, small=False)
    assert meta_path.exists()


def test_metadata_schema_version(tmp_path, cfg):
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    meta_path = save_initial_metadata(test_cfg, _make_stub_graph(), _make_stub_buildings())
    meta = json.loads(meta_path.read_text())
    assert meta["schema_version"] == 1


def test_metadata_counts(tmp_path, cfg):
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    G = _make_stub_graph()
    gdf = _make_stub_buildings()
    meta_path = save_initial_metadata(test_cfg, G, gdf)
    meta = json.loads(meta_path.read_text())
    assert meta["counts"]["graph_nodes"]  == 3
    assert meta["counts"]["graph_edges"]  == 3
    assert meta["counts"]["buildings"]    == 3


def test_metadata_bbox_matches_config(tmp_path, cfg):
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    meta_path = save_initial_metadata(test_cfg, _make_stub_graph(), _make_stub_buildings(), small=False)
    meta = json.loads(meta_path.read_text())
    assert meta["bbox"]["north"] == pytest.approx(cfg["area"]["bbox"]["north"])
    assert meta["bbox"]["west"]  == pytest.approx(cfg["area"]["bbox"]["west"])


def test_metadata_projection_epsgs(tmp_path, cfg):
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    meta_path = save_initial_metadata(test_cfg, _make_stub_graph(), _make_stub_buildings())
    meta = json.loads(meta_path.read_text())
    assert meta["projection"]["source_epsg"] == 4326
    assert meta["projection"]["target_epsg"] == 32654


def test_metadata_deferred_fields_are_null(tmp_path, cfg):
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    meta_path = save_initial_metadata(test_cfg, _make_stub_graph(), _make_stub_buildings())
    meta = json.loads(meta_path.read_text())
    for key in ("road_vertices", "road_triangles", "building_vertices",
                "building_triangles", "skipped_buildings"):
        assert meta["counts"][key] is None, f"counts.{key} should be null at Phase 1"
    assert meta["route"]["length_m"] is None
    assert meta["projection"]["origin_easting"] is None


def test_metadata_small_bbox_flag(tmp_path, cfg):
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    meta_path = save_initial_metadata(test_cfg, _make_stub_graph(), _make_stub_buildings(), small=True)
    meta = json.loads(meta_path.read_text())
    assert meta["small_bbox"] is True
    assert meta["bbox"]["north"] == pytest.approx(cfg["area"]["debug_bbox"]["north"])


def test_metadata_has_created_at(tmp_path, cfg):
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    meta_path = save_initial_metadata(test_cfg, _make_stub_graph(), _make_stub_buildings())
    meta = json.loads(meta_path.read_text())
    assert "created_at" in meta
    assert "T" in meta["created_at"]  # ISO 8601 format
