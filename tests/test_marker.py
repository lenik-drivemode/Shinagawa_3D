"""Tests for vehicle marker geometry (PRD §12.5)."""
from pathlib import Path

import numpy as np
import pytest

from osm3d_poc.render.marker import make_marker_mesh
from osm3d_poc.config import load_config

_CFG_PATH = Path(__file__).parent.parent / "config" / "shinagawa_poc.yaml"

_MARKER_CFG = {
    "render": {
        "marker_length_m": 8.0,
        "marker_width_m":  4.0,
        "marker_height_m": 2.0,
    }
}


@pytest.fixture(scope="module")
def marker_mesh():
    return make_marker_mesh(_MARKER_CFG)


def test_marker_vertex_count(marker_mesh):
    verts, _ = marker_mesh
    assert len(verts) == 8   # 8 unique corners


def test_marker_index_count(marker_mesh):
    _, idxs = marker_mesh
    assert len(idxs) == 36   # 6 faces × 2 triangles × 3 vertices


def test_marker_dtype(marker_mesh):
    verts, idxs = marker_mesh
    assert verts.dtype == np.float32
    assert idxs.dtype == np.uint32


def test_marker_indices_are_triangles(marker_mesh):
    _, idxs = marker_mesh
    assert len(idxs) % 3 == 0


def test_marker_indices_in_range(marker_mesh):
    verts, idxs = marker_mesh
    assert np.all(idxs < len(verts))


def test_marker_no_nan_vertices(marker_mesh):
    verts, _ = marker_mesh
    assert not np.any(np.isnan(verts))


def test_marker_x_symmetric(marker_mesh):
    """Marker should be symmetric about x=0."""
    verts, _ = marker_mesh
    hw = _MARKER_CFG["render"]["marker_width_m"] / 2
    assert np.isclose(verts[:, 0].max(),  hw, atol=1e-5)
    assert np.isclose(verts[:, 0].min(), -hw, atol=1e-5)


def test_marker_z_symmetric(marker_mesh):
    """Marker should be symmetric about z=0."""
    verts, _ = marker_mesh
    hl = _MARKER_CFG["render"]["marker_length_m"] / 2
    assert np.isclose(verts[:, 2].max(),  hl, atol=1e-5)
    assert np.isclose(verts[:, 2].min(), -hl, atol=1e-5)


def test_marker_y_range(marker_mesh):
    """Marker bottom at y=0, top at y=height."""
    verts, _ = marker_mesh
    h = _MARKER_CFG["render"]["marker_height_m"]
    assert np.isclose(verts[:, 1].min(), 0.0, atol=1e-5)
    assert np.isclose(verts[:, 1].max(), h,   atol=1e-5)


def test_marker_vertex_shape(marker_mesh):
    verts, _ = marker_mesh
    assert verts.ndim == 2 and verts.shape[1] == 3


def test_marker_with_real_config():
    cfg = load_config(_CFG_PATH)
    verts, idxs = make_marker_mesh(cfg)
    assert len(verts) == 8
    assert len(idxs)  == 36
    hw = cfg["render"]["marker_width_m"]  / 2
    hl = cfg["render"]["marker_length_m"] / 2
    assert np.isclose(verts[:, 0].max(),  hw, atol=1e-5)
    assert np.isclose(verts[:, 2].max(),  hl, atol=1e-5)
