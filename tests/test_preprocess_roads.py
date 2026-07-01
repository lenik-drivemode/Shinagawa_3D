"""Tests for road mesh generation (PRD §7.2, §13.2, §15.1).

No network access required; uses synthetic networkx graphs.
"""
from pathlib import Path

import numpy as np
import pytest

from osm3d_poc.config import load_config
from osm3d_poc.geo.preprocess_roads import (
    build_road_mesh,
    get_edge_width,
    highway_width,
    parse_width,
    polyline_to_strip,
    save_road_mesh,
)
from osm3d_poc.geo.projection import Projector

_CFG_PATH = Path(__file__).parent.parent / "config" / "shinagawa_poc.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(_CFG_PATH)


@pytest.fixture(scope="module")
def proj(cfg):
    return Projector(cfg)


# ---------------------------------------------------------------------------
# parse_width (PRD §13.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("6",     6.0),
    ("6.5",   6.5),
    ("6 m",   6.0),
    ("6.5m",  6.5),
    ("10",   10.0),
    ("10.0", 10.0),
    (6,       6.0),    # numeric input
    (6.5,     6.5),
])
def test_parse_width_valid(raw, expected):
    assert parse_width(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "wide", "6 ft", "abc"])
def test_parse_width_invalid_returns_none(raw):
    assert parse_width(raw) is None


# ---------------------------------------------------------------------------
# highway_width (PRD §7.2, FR-ROAD-004 fallback table)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hw_type, expected", [
    ("motorway",     12.0),
    ("trunk",        12.0),
    ("primary",      10.0),
    ("secondary",     8.0),
    ("tertiary",      7.0),
    ("residential",   5.5),
    ("service",       4.0),
    ("unclassified",  5.0),
    ("unknown",       5.0),   # default
    ("footway",       5.0),   # default
])
def test_highway_width(hw_type, expected):
    assert highway_width(hw_type) == pytest.approx(expected)


def test_highway_width_list_picks_first_known():
    assert highway_width(["footway", "primary"]) == pytest.approx(10.0)


def test_highway_width_list_all_unknown():
    assert highway_width(["footway", "cycleway"]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# get_edge_width
# ---------------------------------------------------------------------------

def test_get_edge_width_uses_osm_tag_when_valid():
    assert get_edge_width({"width": "8 m", "highway": "residential"}) == pytest.approx(8.0)


def test_get_edge_width_falls_back_on_missing_tag():
    assert get_edge_width({"highway": "primary"}) == pytest.approx(10.0)


def test_get_edge_width_falls_back_on_invalid_tag():
    assert get_edge_width({"width": "wide", "highway": "secondary"}) == pytest.approx(8.0)


def test_get_edge_width_clamps_out_of_range_tag():
    # width > 50 m is rejected as insane
    assert get_edge_width({"width": "200", "highway": "primary"}) == pytest.approx(10.0)


def test_get_edge_width_clamps_below_1m():
    assert get_edge_width({"width": "0.5", "highway": "tertiary"}) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# polyline_to_strip
# ---------------------------------------------------------------------------

def test_strip_two_points_produces_one_quad():
    xz = np.array([[0.0, 0.0], [0.0, 100.0]])
    verts, idxs = polyline_to_strip(xz, width=10.0, y=0.02)
    assert verts.shape == (4, 3)
    assert idxs.shape == (6,)


def test_strip_three_points_produces_two_quads():
    xz = np.array([[0.0, 0.0], [0.0, 50.0], [0.0, 100.0]])
    verts, idxs = polyline_to_strip(xz, width=6.0, y=0.0)
    assert verts.shape == (8, 3)
    assert idxs.shape == (12,)


def test_strip_vertex_y_is_set_correctly():
    xz = np.array([[0.0, 0.0], [1.0, 0.0]])
    verts, _ = polyline_to_strip(xz, width=2.0, y=0.02)
    assert np.allclose(verts[:, 1], 0.02)


def test_strip_road_width_in_x_for_eastward_road():
    """Road going east (along X): left/right sides should be ±half_width in Z."""
    xz = np.array([[0.0, 0.0], [100.0, 0.0]])
    verts, _ = polyline_to_strip(xz, width=10.0, y=0.0)
    z_vals = np.unique(verts[:, 2])
    # Should have exactly two Z values: -5 and +5
    assert len(z_vals) == 2
    assert pytest.approx(sorted(z_vals), abs=1e-5) == [-5.0, 5.0]


def test_strip_road_width_in_z_for_northward_road():
    """Road going north (along Z): left/right sides should be ±half_width in X."""
    xz = np.array([[0.0, 0.0], [0.0, 100.0]])
    verts, _ = polyline_to_strip(xz, width=8.0, y=0.0)
    x_vals = np.unique(verts[:, 0])
    assert len(x_vals) == 2
    assert pytest.approx(sorted(x_vals), abs=1e-5) == [-4.0, 4.0]


def test_strip_single_point_returns_empty():
    xz = np.array([[0.0, 0.0]])
    verts, idxs = polyline_to_strip(xz, width=5.0)
    assert len(verts) == 0
    assert len(idxs) == 0


def test_strip_degenerate_segment_skipped():
    # Two identical points in the middle
    xz = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 100.0]])
    verts, idxs = polyline_to_strip(xz, width=5.0)
    # Only the non-degenerate segment should produce geometry
    assert verts.shape == (4, 3)


def test_strip_dtype_is_float32_and_uint32():
    xz = np.array([[0.0, 0.0], [10.0, 0.0]])
    verts, idxs = polyline_to_strip(xz, width=4.0)
    assert verts.dtype == np.float32
    assert idxs.dtype == np.uint32


# ---------------------------------------------------------------------------
# Mesh validity (PRD §15.1): no NaN, all indices in range
# ---------------------------------------------------------------------------

def _make_simple_graph():
    nx = pytest.importorskip("networkx")
    G = nx.MultiDiGraph()
    # Three nodes roughly in Shinagawa: ~100 m apart each
    G.add_node(1, y=35.6285, x=139.7388)   # centre
    G.add_node(2, y=35.6294, x=139.7388)   # ~100 m north
    G.add_node(3, y=35.6285, x=139.7399)   # ~100 m east
    G.add_edge(1, 2, key=0, highway="primary")
    G.add_edge(2, 3, key=0, highway="residential")
    G.add_edge(3, 1, key=0, highway="secondary")
    return G


def test_build_road_mesh_no_nan_vertices(cfg, proj):
    G = _make_simple_graph()
    verts, _, _ = build_road_mesh(G, proj, cfg)
    assert len(verts) > 0
    assert not np.any(np.isnan(verts)), "Mesh must not contain NaN vertices"


def test_build_road_mesh_indices_in_range(cfg, proj):
    G = _make_simple_graph()
    verts, idxs, _ = build_road_mesh(G, proj, cfg)
    assert np.all(idxs < len(verts)), "All indices must be < vertex count"


def test_build_road_mesh_vertex_shape(cfg, proj):
    G = _make_simple_graph()
    verts, _, _ = build_road_mesh(G, proj, cfg)
    assert verts.ndim == 2 and verts.shape[1] == 3


def test_build_road_mesh_index_multiple_of_3(cfg, proj):
    G = _make_simple_graph()
    _, idxs, _ = build_road_mesh(G, proj, cfg)
    assert len(idxs) % 3 == 0, "Index count must be a multiple of 3 (triangles)"


def test_build_road_mesh_meta_has_highway(cfg, proj):
    G = _make_simple_graph()
    _, _, meta = build_road_mesh(G, proj, cfg)
    assert all("highway" in m for m in meta)
    assert all("width_m" in m for m in meta)


def test_build_road_mesh_y_matches_config(cfg, proj):
    G = _make_simple_graph()
    verts, _, _ = build_road_mesh(G, proj, cfg)
    road_y = cfg["preprocess"]["road_y_m"]
    assert np.allclose(verts[:, 1], road_y), "All road vertices must be at road_y_m"


# ---------------------------------------------------------------------------
# save_road_mesh
# ---------------------------------------------------------------------------

def test_save_road_mesh_creates_npz(tmp_path, cfg, proj):
    G = _make_simple_graph()
    verts, idxs, meta = build_road_mesh(G, proj, cfg)
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    npz_path = save_road_mesh(verts, idxs, meta, test_cfg)
    assert npz_path.exists()
    data = np.load(npz_path)
    assert "vertices" in data
    assert "indices" in data
    assert data["vertices"].dtype == np.float32
    assert data["indices"].dtype == np.uint32


def test_save_road_mesh_creates_meta_json(tmp_path, cfg, proj):
    G = _make_simple_graph()
    verts, idxs, meta = build_road_mesh(G, proj, cfg)
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    save_road_mesh(verts, idxs, meta, test_cfg)
    meta_path = tmp_path / "roads_meta.json"
    assert meta_path.exists()


def test_save_road_mesh_roundtrip(tmp_path, cfg, proj):
    """Loaded .npz must have the same arrays as what was saved."""
    G = _make_simple_graph()
    verts, idxs, meta = build_road_mesh(G, proj, cfg)
    test_cfg = {**cfg, "preprocess": {**cfg["preprocess"], "output_dir": str(tmp_path)}}
    npz_path = save_road_mesh(verts, idxs, meta, test_cfg)
    data = np.load(npz_path)
    assert np.array_equal(data["vertices"], verts.astype(np.float32))
    assert np.array_equal(data["indices"],  idxs.astype(np.uint32))
