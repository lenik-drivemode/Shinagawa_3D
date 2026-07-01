"""Tests for building mesh generation (PRD §7.3, §13.1, §15.1).

No network access required; uses synthetic Shapely / GeoDataFrame objects.
"""
from pathlib import Path

import numpy as np
import pytest

from osm3d_poc.config import load_config
from osm3d_poc.geo.preprocess_buildings import (
    _extrude_building,
    _get_polygons,
    _clean_polygon,
    build_building_mesh,
    get_building_height,
    parse_height,
    save_building_mesh,
)
from osm3d_poc.geo.projection import Projector

_CFG_PATH = Path(__file__).parent.parent / "config" / "shinagawa_poc.yaml"

# Simple 10×10 m square in local XZ (already projected)
_SQUARE_XZ = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float64)
# With closing vertex
_SQUARE_XZ_CLOSED = np.vstack([_SQUARE_XZ, _SQUARE_XZ[:1]])


@pytest.fixture(scope="module")
def cfg():
    return load_config(_CFG_PATH)


@pytest.fixture(scope="module")
def proj(cfg):
    return Projector(cfg)


# ---------------------------------------------------------------------------
# parse_height (PRD §13.1, §15.1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("12",    12.0),
    ("12.5",  12.5),
    ("12 m",  12.0),
    ("12.5m", 12.5),
    ("0",      0.0),
    ("100",  100.0),
    (15,      15.0),
    (15.5,    15.5),
])
def test_parse_height_valid(raw, expected):
    assert parse_height(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "tall", "12 ft", "6'2\"", "abc"])
def test_parse_height_invalid_returns_none(raw):
    assert parse_height(raw) is None


# ---------------------------------------------------------------------------
# get_building_height (FR-BLDG-005 priority chain)
# ---------------------------------------------------------------------------

def test_height_tag_takes_priority(cfg):
    row = _make_row(height="15", building_levels="2")
    assert get_building_height(row, cfg) == pytest.approx(15.0)


def test_levels_used_when_no_height_tag(cfg):
    floor_h = cfg["preprocess"]["floor_height_m"]
    row = _make_row(building_levels="4")
    assert get_building_height(row, cfg) == pytest.approx(4 * floor_h)


def test_default_height_used_when_no_tags(cfg):
    default = cfg["preprocess"]["default_building_height_m"]
    row = _make_row()
    assert get_building_height(row, cfg) == pytest.approx(default)


def test_height_clamped_below_min(cfg):
    row = _make_row(height="1")
    assert get_building_height(row, cfg) >= 3.0


def test_height_clamped_above_max(cfg):
    row = _make_row(height="999")
    assert get_building_height(row, cfg) <= 250.0


def test_height_nan_falls_through_to_levels(cfg):
    floor_h = cfg["preprocess"]["floor_height_m"]
    row = _make_row(height=float("nan"), building_levels="3")
    assert get_building_height(row, cfg) == pytest.approx(3 * floor_h)


def test_invalid_height_tag_falls_through_to_levels(cfg):
    floor_h = cfg["preprocess"]["floor_height_m"]
    row = _make_row(height="tall", building_levels="2")
    assert get_building_height(row, cfg) == pytest.approx(2 * floor_h)


# ---------------------------------------------------------------------------
# _extrude_building
# ---------------------------------------------------------------------------

def test_extrude_square_vertex_count():
    verts, _ = _extrude_building(_SQUARE_XZ, height=10.0)
    # 4 walls × 4 verts + 4 roof verts = 20
    assert len(verts) == 20


def test_extrude_square_index_count():
    _, idxs = _extrude_building(_SQUARE_XZ, height=10.0)
    # 4 walls × 2 triangles × 3 idx = 24, plus roof (2 triangles × 3 = 6) = 30
    assert len(idxs) == 30


def test_extrude_closed_ring_same_as_open(cfg):
    v_open,  i_open  = _extrude_building(_SQUARE_XZ,        height=10.0)
    v_closed, i_closed = _extrude_building(_SQUARE_XZ_CLOSED, height=10.0)
    assert v_open.shape == v_closed.shape
    assert i_open.shape == i_closed.shape


def test_extrude_vertex_format_is_6_components():
    verts, _ = _extrude_building(_SQUARE_XZ, height=12.0)
    assert verts.ndim == 2 and verts.shape[1] == 6


def test_extrude_dtype():
    verts, idxs = _extrude_building(_SQUARE_XZ, height=5.0)
    assert verts.dtype == np.float32
    assert idxs.dtype == np.uint32


def test_extrude_no_nan_vertices():
    verts, _ = _extrude_building(_SQUARE_XZ, height=10.0)
    assert not np.any(np.isnan(verts))


def test_extrude_indices_in_range():
    verts, idxs = _extrude_building(_SQUARE_XZ, height=10.0)
    assert np.all(idxs < len(verts))


def test_extrude_index_multiple_of_3():
    _, idxs = _extrude_building(_SQUARE_XZ, height=10.0)
    assert len(idxs) % 3 == 0


def test_extrude_wall_y_range(cfg):
    height = 12.0
    verts, _ = _extrude_building(_SQUARE_XZ, height=height, base_y=0.0)
    y_vals = verts[:, 1]
    assert float(y_vals.min()) == pytest.approx(0.0, abs=1e-4)
    assert float(y_vals.max()) == pytest.approx(height, abs=1e-4)


def test_extrude_roof_normals_point_up():
    verts, _ = _extrude_building(_SQUARE_XZ, height=10.0)
    # Roof vertices are the last 4; their ny component should be 1.0
    roof_verts = verts[-4:]
    assert np.allclose(roof_verts[:, 4], 1.0), "Roof normals must have ny=1"


def test_extrude_wall_normals_are_horizontal():
    verts, _ = _extrude_building(_SQUARE_XZ, height=10.0)
    # Wall vertices are the first 16 (4 walls × 4 verts)
    wall_verts = verts[:16]
    assert np.allclose(wall_verts[:, 4], 0.0), "Wall normals must have ny=0"


def test_extrude_wall_normals_are_unit_length():
    verts, _ = _extrude_building(_SQUARE_XZ, height=10.0)
    wall_verts = verts[:16]
    normals = wall_verts[:, 3:]
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_extrude_degenerate_ring_returns_empty():
    xz = np.array([[0.0, 0.0], [1.0, 0.0]])  # only 2 points
    verts, idxs = _extrude_building(xz, height=10.0)
    assert len(verts) == 0
    assert len(idxs) == 0


# ---------------------------------------------------------------------------
# _get_polygons and _clean_polygon
# ---------------------------------------------------------------------------

def test_get_polygons_from_polygon():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon
    poly = Polygon(_SQUARE_XZ)
    result = _get_polygons(poly)
    assert len(result) == 1


def test_get_polygons_from_multipolygon():
    from shapely.geometry import MultiPolygon, Polygon
    mp = MultiPolygon([Polygon(_SQUARE_XZ), Polygon(_SQUARE_XZ + 20)])
    result = _get_polygons(mp)
    assert len(result) == 2


def test_get_polygons_from_non_polygon_returns_empty():
    from shapely.geometry import LineString
    result = _get_polygons(LineString([[0, 0], [1, 1]]))
    assert result == []


def test_clean_valid_polygon_unchanged():
    from shapely.geometry import Polygon
    poly = Polygon(_SQUARE_XZ)
    result = _clean_polygon(poly)
    assert result is not None
    assert result.is_valid


# ---------------------------------------------------------------------------
# build_building_mesh (full pipeline with synthetic GeoDataFrame)
# ---------------------------------------------------------------------------

def _make_synthetic_gdf():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    # Two buildings near Shinagawa centre — tiny footprints in lat/lon
    delta = 0.0001   # ~10 m in degrees
    b1 = Polygon([
        (139.7388,        35.6285),
        (139.7388+delta,  35.6285),
        (139.7388+delta,  35.6285+delta),
        (139.7388,        35.6285+delta),
    ])
    b2 = Polygon([
        (139.7395,        35.6290),
        (139.7395+delta,  35.6290),
        (139.7395+delta,  35.6290+delta),
        (139.7395,        35.6290+delta),
    ])
    return gpd.GeoDataFrame({
        "geometry":          [b1,    b2],
        "height":            ["15",  None],
        "building:levels":   [None,  "3"],
    }, crs="EPSG:4326")


def test_build_building_mesh_no_nan_vertices(cfg, proj):
    gdf = _make_synthetic_gdf()
    verts, _, _ = build_building_mesh(gdf, proj, cfg)
    assert len(verts) > 0
    assert not np.any(np.isnan(verts))


def test_build_building_mesh_indices_in_range(cfg, proj):
    gdf = _make_synthetic_gdf()
    verts, idxs, _ = build_building_mesh(gdf, proj, cfg)
    assert np.all(idxs < len(verts))


def test_build_building_mesh_vertex_shape(cfg, proj):
    gdf = _make_synthetic_gdf()
    verts, _, _ = build_building_mesh(gdf, proj, cfg)
    assert verts.ndim == 2 and verts.shape[1] == 6


def test_build_building_mesh_index_multiple_of_3(cfg, proj):
    gdf = _make_synthetic_gdf()
    _, idxs, _ = build_building_mesh(gdf, proj, cfg)
    assert len(idxs) % 3 == 0


def test_build_building_mesh_produces_two_buildings(cfg, proj):
    gdf = _make_synthetic_gdf()
    _, _, meta = build_building_mesh(gdf, proj, cfg)
    assert len(meta) == 2


def test_build_building_mesh_heights_applied(cfg, proj):
    gdf = _make_synthetic_gdf()
    _, _, meta = build_building_mesh(gdf, proj, cfg)
    heights = [m["height_m"] for m in meta]
    assert heights[0] == pytest.approx(15.0)
    floor_h = cfg["preprocess"]["floor_height_m"]
    assert heights[1] == pytest.approx(3 * floor_h)


# ---------------------------------------------------------------------------
# save_building_mesh
# ---------------------------------------------------------------------------

def test_save_building_mesh_creates_npz(tmp_path, cfg, proj):
    gdf = _make_synthetic_gdf()
    verts, idxs, meta = build_building_mesh(gdf, proj, cfg)
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    npz_path = save_building_mesh(verts, idxs, meta, test_cfg)
    assert npz_path.exists()
    data = np.load(npz_path)
    assert "vertices" in data and "indices" in data
    assert data["vertices"].dtype == np.float32
    assert data["indices"].dtype == np.uint32


def test_save_building_mesh_roundtrip(tmp_path, cfg, proj):
    gdf = _make_synthetic_gdf()
    verts, idxs, meta = build_building_mesh(gdf, proj, cfg)
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    npz_path = save_building_mesh(verts, idxs, meta, test_cfg)
    data = np.load(npz_path)
    assert np.array_equal(data["vertices"], verts.astype(np.float32))
    assert np.array_equal(data["indices"],  idxs.astype(np.uint32))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(height=None, building_levels=None):
    """Return a dict-like object mimicking a GeoDataFrame row."""
    import pandas as pd
    return pd.Series({
        "height": height,
        "building:levels": building_levels,
    })
