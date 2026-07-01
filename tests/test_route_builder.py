"""Tests for route generation (PRD §7.4, §10.4, §13.3, §15.1).

No network access required; uses synthetic networkx graphs.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from osm3d_poc.config import load_config
from osm3d_poc.geo.route_builder import (
    build_node_path,
    compute_cumulative_s,
    densify_xz,
    load_route,
    save_route,
    build_route,
    snap_waypoints,
    _densify_columns,
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
# _densify_columns / densify_xz (PRD §13.3)
# ---------------------------------------------------------------------------

def test_densify_no_change_when_short_segments():
    """Points 4 m apart, max 5 m → no new points inserted."""
    xz = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    result = densify_xz(xz, max_seg_m=5.0)
    assert len(result) == 2
    assert np.allclose(result[0], [0.0, 0.0])
    assert np.allclose(result[1], [4.0, 0.0])


def test_densify_inserts_points_for_long_segment():
    """12 m segment with max 5 m → ceil(12/5)=3 sub-segments → 4 points."""
    xz = np.array([[0.0, 0.0], [12.0, 0.0]], dtype=np.float64)
    result = densify_xz(xz, max_seg_m=5.0)
    assert len(result) == 4   # 0, 4, 8, 12


def test_densify_preserves_first_and_last():
    xz = np.array([[1.0, 2.0], [20.0, 5.0]], dtype=np.float64)
    result = densify_xz(xz, max_seg_m=5.0)
    assert np.allclose(result[0], xz[0])
    assert np.allclose(result[-1], xz[-1])


def test_densify_no_segment_exceeds_max():
    xz = np.array([[0.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float64)
    max_seg = 5.0
    result = densify_xz(xz, max_seg_m=max_seg)
    diffs = np.diff(result, axis=0)
    segs = np.hypot(diffs[:, 0], diffs[:, 1])
    assert np.all(segs <= max_seg + 1e-9)


def test_densify_single_point_returns_single():
    xz = np.array([[3.0, 7.0]], dtype=np.float64)
    result = densify_xz(xz, max_seg_m=5.0)
    assert len(result) == 1


def test_densify_empty_returns_empty():
    xz = np.empty((0, 2), dtype=np.float64)
    result = densify_xz(xz, max_seg_m=5.0)
    assert len(result) == 0


def test_densify_columns_interpolates_all_dims():
    """Extra columns (lat, lon) should be linearly interpolated too."""
    cols = np.array([[0.0, 0.0, 10.0, 20.0],
                     [10.0, 0.0, 10.1, 20.1]], dtype=np.float64)
    result = _densify_columns(cols, max_seg_m=3.0)
    assert result.shape[1] == 4
    # lat midpoint should be approximately 10.05
    mid = result[len(result) // 2]
    assert 10.0 <= mid[2] <= 10.1


def test_densify_monotone_x_along_axis():
    """Densified x-coords for a straight East-going road should be monotone."""
    xz = np.array([[0.0, 0.0], [50.0, 0.0]], dtype=np.float64)
    result = densify_xz(xz, max_seg_m=5.0)
    assert np.all(np.diff(result[:, 0]) > 0)


# ---------------------------------------------------------------------------
# compute_cumulative_s (FR-ROUTE-005)
# ---------------------------------------------------------------------------

def test_cumulative_s_starts_at_zero():
    xz = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    s = compute_cumulative_s(xz)
    assert s[0] == 0.0


def test_cumulative_s_single_segment():
    xz = np.array([[0.0, 0.0], [0.0, 100.0]])
    s = compute_cumulative_s(xz)
    assert np.isclose(s[-1], 100.0)


def test_cumulative_s_right_angle():
    xz = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    s = compute_cumulative_s(xz)
    assert np.isclose(s[1], 3.0)
    assert np.isclose(s[2], 7.0)


def test_cumulative_s_monotonic():
    xz = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 10.0], [15.0, 10.0]])
    s = compute_cumulative_s(xz)
    assert np.all(np.diff(s) > 0)


def test_cumulative_s_length():
    xz = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    s = compute_cumulative_s(xz)
    assert len(s) == len(xz)


def test_cumulative_s_dtype():
    xz = np.array([[0.0, 0.0], [10.0, 0.0]])
    s = compute_cumulative_s(xz)
    assert s.dtype == np.float64


# ---------------------------------------------------------------------------
# build_node_path (FR-ROUTE-003, FR-ROUTE-004)
# ---------------------------------------------------------------------------

def _make_simple_graph():
    """4-node diamond graph with edge lengths in metres."""
    nx = pytest.importorskip("networkx")
    G = nx.MultiDiGraph()
    # nodes 1-4 with arbitrary lat/lon near Shinagawa
    G.add_node(1, y=35.6285, x=139.7388)
    G.add_node(2, y=35.6310, x=139.7388)
    G.add_node(3, y=35.6310, x=139.7420)
    G.add_node(4, y=35.6285, x=139.7420)
    edges = [(1,2,300), (2,3,300), (3,4,300), (4,1,300),
             (2,1,300), (3,2,300), (4,3,300), (1,4,300)]
    for u, v, ln in edges:
        G.add_edge(u, v, key=0, length=float(ln), highway="secondary")
    return G


def test_build_node_path_simple():
    G = _make_simple_graph()
    path = build_node_path(G, [1, 3])
    assert path is not None
    assert path[0] == 1
    assert path[-1] == 3
    assert len(path) >= 2


def test_build_node_path_closed_loop():
    G = _make_simple_graph()
    path = build_node_path(G, [1, 2, 3, 4, 1])
    assert path is not None
    assert path[0] == 1
    assert path[-1] == 1


def test_build_node_path_no_duplicate_at_join():
    G = _make_simple_graph()
    path = build_node_path(G, [1, 2, 3])
    assert path is not None
    # Node 2 should appear exactly once at the join
    assert path.count(2) == 1


def test_build_node_path_no_path_returns_none():
    nx = pytest.importorskip("networkx")
    G = nx.MultiDiGraph()
    G.add_node(1, y=35.6285, x=139.7388)
    G.add_node(2, y=35.6310, x=139.7388)
    # No edges → no path
    path = build_node_path(G, [1, 2])
    assert path is None


def test_build_node_path_same_node_twice():
    G = _make_simple_graph()
    path = build_node_path(G, [1, 1, 2])
    assert path is not None
    assert path[0] == 1
    assert path[-1] == 2


def test_build_node_path_at_least_two_nodes():
    G = _make_simple_graph()
    path = build_node_path(G, [1, 2])
    assert path is not None
    assert len(path) >= 2


# ---------------------------------------------------------------------------
# build_route + snap_waypoints (integration with synthetic graph)
# ---------------------------------------------------------------------------

def _make_loop_graph():
    """3×3 grid of nodes with bidirectional edges near Shinagawa."""
    nx = pytest.importorskip("networkx")
    G = nx.MultiDiGraph()

    # dlat≈0.004° ~ 445m, dlon≈0.005° ~ 445m
    base_lat, base_lon = 35.622, 35.622
    dlat, dlon = 0.004, 0.006
    base_lat, base_lon = 35.624, 139.731

    positions = {}
    for i in range(3):
        for j in range(3):
            nid = i * 3 + j + 1
            lat = base_lat + i * dlat
            lon = base_lon + j * dlon
            G.add_node(nid, y=lat, x=lon)
            positions[nid] = (lat, lon)

    def _length(u, v):
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:32654", always_xy=True)
        eu, nu = t.transform(positions[u][1], positions[u][0])
        ev, nv = t.transform(positions[v][1], positions[v][0])
        return float(np.hypot(ev - eu, nv - nu))

    for i in range(3):
        for j in range(2):
            u = i * 3 + j + 1
            v = i * 3 + j + 2
            ln = _length(u, v)
            G.add_edge(u, v, key=0, length=ln, highway="secondary")
            G.add_edge(v, u, key=0, length=ln, highway="secondary")
    for i in range(2):
        for j in range(3):
            u = i * 3 + j + 1
            v = (i + 1) * 3 + j + 1
            ln = _length(u, v)
            G.add_edge(u, v, key=0, length=ln, highway="secondary")
            G.add_edge(v, u, key=0, length=ln, highway="secondary")
    return G, positions


def _make_route_cfg(cfg, tmp_path):
    """Return a cfg copy that routes between 3×3 grid corners."""
    base_lat, base_lon = 35.624, 139.731
    dlat, dlon = 0.004, 0.006
    return {
        **cfg,
        "route": {
            **cfg["route"],
            "output_file": str(tmp_path / "test_route.json"),
            "desired_min_length_m": 0,
            "desired_max_length_m": 999999,
            "default_speed_kmh": 40,
            "waypoints": [
                {"name": "A", "lat": base_lat,           "lon": base_lon},
                {"name": "B", "lat": base_lat + dlat,    "lon": base_lon + dlon},
                {"name": "C", "lat": base_lat + 2*dlat,  "lon": base_lon + 2*dlon},
                {"name": "A2","lat": base_lat,            "lon": base_lon},
            ],
        },
    }


def test_build_route_returns_required_keys(cfg, proj, tmp_path):
    G, _ = _make_loop_graph()
    rcfg = _make_route_cfg(cfg, tmp_path)
    route = build_route(G, proj, rcfg)
    for key in ("schema_version", "name", "closed_loop", "speed_default_kmh",
                "length_m", "points", "source"):
        assert key in route, f"Missing key: {key}"


def test_build_route_schema_version(cfg, proj, tmp_path):
    G, _ = _make_loop_graph()
    rcfg = _make_route_cfg(cfg, tmp_path)
    route = build_route(G, proj, rcfg)
    assert route["schema_version"] == 1


def test_build_route_points_have_required_fields(cfg, proj, tmp_path):
    G, _ = _make_loop_graph()
    rcfg = _make_route_cfg(cfg, tmp_path)
    route = build_route(G, proj, rcfg)
    assert len(route["points"]) > 0
    p = route["points"][0]
    for field in ("x", "y", "z", "lat", "lon", "s"):
        assert field in p, f"Point missing field: {field}"


def test_build_route_first_s_is_zero(cfg, proj, tmp_path):
    G, _ = _make_loop_graph()
    rcfg = _make_route_cfg(cfg, tmp_path)
    route = build_route(G, proj, rcfg)
    assert route["points"][0]["s"] == pytest.approx(0.0)


def test_build_route_s_monotone(cfg, proj, tmp_path):
    G, _ = _make_loop_graph()
    rcfg = _make_route_cfg(cfg, tmp_path)
    route = build_route(G, proj, rcfg)
    s_vals = [p["s"] for p in route["points"]]
    assert all(s_vals[i] <= s_vals[i + 1] for i in range(len(s_vals) - 1))


def test_build_route_length_matches_last_s(cfg, proj, tmp_path):
    G, _ = _make_loop_graph()
    rcfg = _make_route_cfg(cfg, tmp_path)
    route = build_route(G, proj, rcfg)
    assert route["length_m"] == pytest.approx(route["points"][-1]["s"], rel=1e-6)


def test_build_route_route_y_applied(cfg, proj, tmp_path):
    G, _ = _make_loop_graph()
    rcfg = _make_route_cfg(cfg, tmp_path)
    route = build_route(G, proj, rcfg)
    route_y = rcfg["preprocess"]["route_y_m"]
    assert all(abs(p["y"] - route_y) < 1e-6 for p in route["points"])


def test_build_route_no_dense_segment_exceeds_max(cfg, proj, tmp_path):
    G, _ = _make_loop_graph()
    rcfg = _make_route_cfg(cfg, tmp_path)
    route = build_route(G, proj, rcfg)
    pts = route["points"]
    for i in range(len(pts) - 1):
        dx = pts[i + 1]["x"] - pts[i]["x"]
        dz = pts[i + 1]["z"] - pts[i]["z"]
        seg = np.hypot(dx, dz)
        assert seg <= 5.0 + 1e-6, f"Segment {i} too long: {seg:.2f} m"


# ---------------------------------------------------------------------------
# save_route / load_route (FR-ROUTE-004)
# ---------------------------------------------------------------------------

def _make_minimal_route_dict(cfg):
    return {
        "schema_version":    1,
        "name":              "Test",
        "closed_loop":       True,
        "speed_default_kmh": 40,
        "length_m":          100.0,
        "points": [
            {"x": 0.0,  "y": 0.08, "z": 0.0,  "lat": 35.6285, "lon": 139.7388, "s": 0.0},
            {"x": 50.0, "y": 0.08, "z": 0.0,  "lat": 35.6286, "lon": 139.7389, "s": 50.0},
            {"x": 100.0,"y": 0.08, "z": 0.0,  "lat": 35.6287, "lon": 139.7390, "s": 100.0},
        ],
        "source": {"method": "test"},
    }


def test_save_route_creates_file(tmp_path, cfg):
    rcfg = {**cfg, "route": {**cfg["route"], "output_file": str(tmp_path / "r.json")}}
    rd   = _make_minimal_route_dict(cfg)
    path = save_route(rd, rcfg)
    assert path.exists()


def test_save_route_valid_json(tmp_path, cfg):
    rcfg = {**cfg, "route": {**cfg["route"], "output_file": str(tmp_path / "r.json")}}
    rd   = _make_minimal_route_dict(cfg)
    path = save_route(rd, rcfg)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["schema_version"] == 1


def test_load_route_returns_xyz_and_s(tmp_path, cfg):
    rcfg = {**cfg, "route": {**cfg["route"], "output_file": str(tmp_path / "r.json")}}
    rd   = _make_minimal_route_dict(cfg)
    save_route(rd, rcfg)
    xyz, s = load_route(rcfg)
    assert xyz.shape == (3, 3)
    assert s.shape  == (3,)


def test_load_route_dtype(tmp_path, cfg):
    rcfg = {**cfg, "route": {**cfg["route"], "output_file": str(tmp_path / "r.json")}}
    rd   = _make_minimal_route_dict(cfg)
    save_route(rd, rcfg)
    xyz, s = load_route(rcfg)
    assert xyz.dtype == np.float32
    assert s.dtype   == np.float32


def test_load_route_s_starts_at_zero(tmp_path, cfg):
    rcfg = {**cfg, "route": {**cfg["route"], "output_file": str(tmp_path / "r.json")}}
    rd   = _make_minimal_route_dict(cfg)
    save_route(rd, rcfg)
    _, s = load_route(rcfg)
    assert s[0] == pytest.approx(0.0)


def test_save_load_route_roundtrip(tmp_path, cfg):
    rcfg = {**cfg, "route": {**cfg["route"], "output_file": str(tmp_path / "r.json")}}
    rd   = _make_minimal_route_dict(cfg)
    save_route(rd, rcfg)
    xyz, s = load_route(rcfg)
    expected_x = [p["x"] for p in rd["points"]]
    assert np.allclose(xyz[:, 0], expected_x)
    expected_s = [p["s"] for p in rd["points"]]
    assert np.allclose(s, expected_s)
