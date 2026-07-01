"""Tests for the projection module (PRD §15.1 — projection tests).

All assertions use real pyproj calls; no mocking.  No network needed.
"""
from pathlib import Path

import numpy as np
import pytest

from osm3d_poc.config import load_config
from osm3d_poc.geo.projection import Projector, make_transformer, project_origin

_CFG_PATH = Path(__file__).parent.parent / "config" / "shinagawa_poc.yaml"

# Shinagawa area reference points (WGS84)
_CENTER_LAT = 35.6285
_CENTER_LON = 139.7388

# 1 degree of latitude ≈ 111 km; 1 degree of lon ≈ 91 km at 35.6°N
_M_PER_DEG_LAT = 111_000.0
_M_PER_DEG_LON =  91_000.0


@pytest.fixture(scope="module")
def cfg():
    return load_config(_CFG_PATH)


@pytest.fixture(scope="module")
def proj(cfg):
    return Projector(cfg)


# ---------------------------------------------------------------------------
# Transformer creation
# ---------------------------------------------------------------------------

def test_transformer_is_created(cfg):
    t = make_transformer(cfg)
    assert t is not None


def test_origin_easting_is_in_utm54n_range(cfg):
    t = make_transformer(cfg)
    easting, _ = project_origin(cfg, t)
    # UTM zone 54N easting for Tokyo should be ~380 000 – 410 000 m
    assert 370_000 < easting < 420_000, f"Unexpected easting {easting:.0f}"


def test_origin_northing_is_in_utm54n_range(cfg):
    t = make_transformer(cfg)
    _, northing = project_origin(cfg, t)
    # UTM zone 54N northing for Tokyo (35.6°N) ≈ 3 940 000 – 3 960 000 m
    assert 3_900_000 < northing < 4_000_000, f"Unexpected northing {northing:.0f}"


# ---------------------------------------------------------------------------
# Single-point: Projector.to_xz
# ---------------------------------------------------------------------------

def test_center_maps_to_origin(proj):
    x, z = proj.to_xz(_CENTER_LAT, _CENTER_LON)
    assert abs(x) < 1.0, f"Center should be at x≈0, got {x:.3f}"
    assert abs(z) < 1.0, f"Center should be at z≈0, got {z:.3f}"


def test_north_of_center_has_positive_z(proj):
    """Moving north increases Z (PRD §4.3: Z positive north)."""
    _, z = proj.to_xz(_CENTER_LAT + 0.01, _CENTER_LON)
    assert z > 0, f"North offset should give positive Z, got {z:.1f}"


def test_east_of_center_has_positive_x(proj):
    """Moving east increases X (PRD §4.3: X positive east)."""
    x, _ = proj.to_xz(_CENTER_LAT, _CENTER_LON + 0.01)
    assert x > 0, f"East offset should give positive X, got {x:.1f}"


def test_north_offset_distance_is_plausible(proj):
    """0.009° latitude ≈ 1 000 m; accept ±15% tolerance."""
    delta_deg = 0.009
    _, z = proj.to_xz(_CENTER_LAT + delta_deg, _CENTER_LON)
    expected = delta_deg * _M_PER_DEG_LAT
    assert abs(z - expected) / expected < 0.15, (
        f"Northing offset {z:.1f} m too far from expected {expected:.1f} m"
    )


def test_east_offset_distance_is_plausible(proj):
    """0.011° longitude ≈ 1 000 m at this latitude; accept ±15% tolerance."""
    delta_deg = 0.011
    x, _ = proj.to_xz(_CENTER_LAT, _CENTER_LON + delta_deg)
    expected = delta_deg * _M_PER_DEG_LON
    assert abs(x - expected) / expected < 0.15, (
        f"Easting offset {x:.1f} m too far from expected {expected:.1f} m"
    )


def test_two_close_points_distance_is_plausible(proj):
    """Two points ~141 m apart (100 m N + 100 m E) should project ~141 m apart."""
    lat_delta = 100 / _M_PER_DEG_LAT   # ≈ 0.0009°
    lon_delta = 100 / _M_PER_DEG_LON   # ≈ 0.0011°
    x1, z1 = proj.to_xz(_CENTER_LAT, _CENTER_LON)
    x2, z2 = proj.to_xz(_CENTER_LAT + lat_delta, _CENTER_LON + lon_delta)
    dist = np.hypot(x2 - x1, z2 - z1)
    assert 100 < dist < 200, f"Expected ~141 m, got {dist:.1f} m"


def test_local_coords_are_in_metre_scale_not_degree_scale(proj):
    """Sanity: local coords for bbox corners are in km range, not degree range."""
    x, z = proj.to_xz(35.6420, 139.7600)  # NE corner
    # Should be a few km from origin, not ~0.02 (degree range)
    assert abs(x) > 100, "X should be in metres (hundreds to thousands)"
    assert abs(z) > 100, "Z should be in metres (hundreds to thousands)"
    assert abs(x) < 10_000, "X should be < 10 km for this bbox"
    assert abs(z) < 10_000, "Z should be < 10 km for this bbox"


# ---------------------------------------------------------------------------
# Batch: Projector.batch_to_xz
# ---------------------------------------------------------------------------

def test_batch_matches_single_point(proj):
    lats = np.array([_CENTER_LAT, _CENTER_LAT + 0.01, _CENTER_LAT - 0.005])
    lons = np.array([_CENTER_LON, _CENTER_LON + 0.01, _CENTER_LON - 0.005])
    xs_batch, zs_batch = proj.batch_to_xz(lats, lons)
    for i in range(len(lats)):
        x_single, z_single = proj.to_xz(float(lats[i]), float(lons[i]))
        assert abs(xs_batch[i] - x_single) < 1e-6
        assert abs(zs_batch[i] - z_single) < 1e-6


def test_batch_returns_float64(proj):
    xs, zs = proj.batch_to_xz([_CENTER_LAT], [_CENTER_LON])
    assert xs.dtype == np.float64
    assert zs.dtype == np.float64


def test_batch_accepts_lists(proj):
    xs, zs = proj.batch_to_xz([_CENTER_LAT, _CENTER_LAT + 0.01],
                               [_CENTER_LON, _CENTER_LON + 0.01])
    assert len(xs) == 2
    assert len(zs) == 2


# ---------------------------------------------------------------------------
# polygon_to_xz
# ---------------------------------------------------------------------------

def test_polygon_exterior_projected(proj):
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon
    # Small square around the center (~0.001° ≈ ~90 m per side)
    poly = Polygon([
        (_CENTER_LON,        _CENTER_LAT),
        (_CENTER_LON + 0.001, _CENTER_LAT),
        (_CENTER_LON + 0.001, _CENTER_LAT + 0.001),
        (_CENTER_LON,        _CENTER_LAT + 0.001),
    ])
    xs, zs = proj.polygon_to_xz(poly)
    # Exterior ring includes closing vertex, so 5 points for a 4-vertex polygon
    assert len(xs) == 5
    assert len(zs) == 5
    # All local coords should be near the origin (within ±200 m)
    assert np.all(np.abs(xs) < 200)
    assert np.all(np.abs(zs) < 200)
