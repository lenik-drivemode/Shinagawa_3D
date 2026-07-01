"""Tests for camera matrix math and mode logic (PRD §7.7, §12.4).

No GL context required — all tests operate on pure numpy math.
"""
from pathlib import Path

import numpy as np
import pytest

from osm3d_poc.render.camera import (
    Camera, look_at, ortho, perspective, rotate_y, translate,
)
from osm3d_poc.config import load_config

_CFG_PATH = Path(__file__).parent.parent / "config" / "shinagawa_poc.yaml"

_RENDER_CFG = {
    "render": {
        "initial_camera_height_m": 1200,
        "initial_pitch_deg":  45,
        "initial_yaw_deg":     0,
        "follow_close_back_m": 25,
        "follow_close_up_m":   12,
        "follow_aerial_up_m": 120,
    }
}


@pytest.fixture(scope="module")
def cfg():
    return _RENDER_CFG


# ---------------------------------------------------------------------------
# look_at
# ---------------------------------------------------------------------------

def test_look_at_shape():
    eye    = np.array([0, 0, 5], dtype=np.float64)
    center = np.zeros(3, dtype=np.float64)
    up     = np.array([0, 1, 0], dtype=np.float64)
    mat = look_at(eye, center, up)
    assert mat.shape == (4, 4)


def test_look_at_dtype():
    mat = look_at(np.array([0,0,5.0]), np.zeros(3), np.array([0,1,0.0]))
    assert mat.dtype == np.float32


def test_look_at_transforms_eye_to_origin():
    """After applying the view matrix, the eye should map to (0,0,0)."""
    eye    = np.array([3.0, 2.0, 5.0])
    center = np.zeros(3, dtype=np.float64)
    up     = np.array([0.0, 1.0, 0.0])
    mat = look_at(eye, center, up)
    # eye in homogeneous coords
    eye_h  = np.array([*eye, 1.0], dtype=np.float64)
    result = mat @ eye_h
    assert np.allclose(result[:3], [0.0, 0.0, 0.0], atol=1e-5)


def test_look_at_transforms_target_to_negative_z():
    """In a standard right-handed view, target should be at negative Z."""
    eye    = np.array([0.0, 0.0, 10.0])
    center = np.zeros(3, dtype=np.float64)
    up     = np.array([0.0, 1.0, 0.0])
    mat = look_at(eye, center, up)
    center_h = np.array([0.0, 0.0, 0.0, 1.0])
    result = mat @ center_h
    assert result[2] < 0, "Target should be at negative Z in camera space"


def test_look_at_world_up_y_is_camera_y():
    """When looking straight forward with world up = (0,1,0), camera up = (0,1,0)."""
    eye    = np.array([0.0, 0.0, 5.0])
    center = np.zeros(3, dtype=np.float64)
    up     = np.array([0.0, 1.0, 0.0])
    mat = look_at(eye, center, up)
    # The Y row of the view matrix is the camera's up direction
    up_world = np.array([0.0, 1.0, 0.0, 0.0])
    up_cam   = mat @ up_world
    assert up_cam[1] > 0.99, "World up should mostly align with camera up"


# ---------------------------------------------------------------------------
# perspective
# ---------------------------------------------------------------------------

def test_perspective_shape():
    mat = perspective(60.0, 16/9, 0.1, 10000.0)
    assert mat.shape == (4, 4)


def test_perspective_dtype():
    mat = perspective(60.0, 16/9, 0.1, 10000.0)
    assert mat.dtype == np.float32


def test_perspective_w_component():
    """Perspective matrix maps to clip-space; [3,2] must be -1 (for NDC divide)."""
    mat = perspective(60.0, 16/9, 0.1, 10000.0)
    assert mat[3, 2] == pytest.approx(-1.0)


def test_perspective_fov_scaling():
    """Narrower FOV → larger projection scaling (M[0,0] and M[1,1])."""
    wide   = perspective(90.0, 1.0, 0.1, 1000.0)
    narrow = perspective(30.0, 1.0, 0.1, 1000.0)
    assert narrow[1, 1] > wide[1, 1]


def test_perspective_aspect_scaling():
    """Wider aspect ratio → smaller horizontal scale (M[0,0])."""
    sq   = perspective(60.0, 1.0,    0.1, 1000.0)
    wide = perspective(60.0, 16/9.0, 0.1, 1000.0)
    assert sq[0, 0] > wide[0, 0]


# ---------------------------------------------------------------------------
# ortho
# ---------------------------------------------------------------------------

def test_ortho_shape():
    assert ortho(-100, 100, -100, 100, -1000, 1000).shape == (4, 4)


def test_ortho_maps_centre_to_origin():
    mat = ortho(-100.0, 100.0, -100.0, 100.0, -1000.0, 1000.0)
    pt  = np.array([0.0, 0.0, 0.0, 1.0])
    out = mat @ pt
    assert np.allclose(out[:2], [0.0, 0.0], atol=1e-5)


def test_ortho_maps_corners():
    mat = ortho(-100.0, 100.0, -100.0, 100.0, -1000.0, 1000.0)
    corner = np.array([100.0, 100.0, 0.0, 1.0])
    out    = mat @ corner
    assert np.allclose(out[:2], [1.0, 1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# translate / rotate_y
# ---------------------------------------------------------------------------

def test_translate_moves_point():
    mat = translate(3.0, 5.0, 7.0)
    pt  = np.array([0.0, 0.0, 0.0, 1.0])
    out = mat @ pt
    assert np.allclose(out[:3], [3.0, 5.0, 7.0], atol=1e-6)


def test_rotate_y_90_degrees():
    """Right-hand rotation: +π/2 around Y maps +X toward -Z."""
    mat = rotate_y(np.pi / 2)
    pt  = np.array([1.0, 0.0, 0.0, 1.0])
    out = mat @ pt
    # After +90° CCW around Y: (1,0,0) → (0,0,-1)
    assert np.allclose(out[2], -1.0, atol=1e-5)
    assert np.allclose(out[0],  0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Camera class
# ---------------------------------------------------------------------------

def test_camera_initial_mode(cfg):
    cam = Camera(cfg)
    assert cam.mode == "orbit"


def test_camera_orbit_eye_distance(cfg):
    cam = Camera(cfg)
    eye  = cam.orbit_eye()
    dist = np.linalg.norm(eye - cam.target)
    assert np.isclose(dist, cam.distance, rtol=1e-5)


def test_camera_orbit_eye_above_target(cfg):
    cam = Camera(cfg)
    cam.pitch = 45.0
    eye = cam.orbit_eye()
    assert eye[1] > cam.target[1], "Eye must be above target when pitch > 0"


def test_camera_pitch_clamp(cfg):
    cam = Camera(cfg)
    cam.orbit_drag(0, -9999)   # large upward drag should clamp pitch
    assert cam.pitch <= 89.0


def test_camera_pitch_floor(cfg):
    cam = Camera(cfg)
    cam.orbit_drag(0, 9999)   # large downward drag should clamp pitch
    assert cam.pitch >= 5.0


def test_camera_zoom_clamps_min(cfg):
    cam = Camera(cfg)
    for _ in range(1000):
        cam.zoom(10.0)
    assert cam.distance >= 20.0


def test_camera_zoom_clamps_max(cfg):
    cam = Camera(cfg)
    for _ in range(1000):
        cam.zoom(-10.0)
    assert cam.distance <= 15000.0


def test_camera_view_matrix_shape(cfg):
    cam = Camera(cfg)
    assert cam.get_view_matrix().shape == (4, 4)


def test_camera_projection_matrix_shape(cfg):
    cam = Camera(cfg)
    assert cam.get_projection_matrix(16 / 9).shape == (4, 4)


def test_camera_cycle_follow_changes_mode(cfg):
    cam = Camera(cfg)
    assert cam.mode == "orbit"
    cam.cycle_follow_mode()
    assert cam.mode == "follow_close"
    cam.cycle_follow_mode()
    assert cam.mode == "follow_aerial"
    cam.cycle_follow_mode()
    assert cam.mode == "orbit"


def test_camera_toggle_top_down(cfg):
    cam = Camera(cfg)
    assert cam.mode == "orbit"
    cam.toggle_top_down()
    assert cam.mode == "top_down"
    cam.toggle_top_down()
    assert cam.mode == "orbit"


def test_camera_top_down_uses_ortho(cfg):
    cam = Camera(cfg)
    cam.toggle_top_down()
    proj_td    = cam.get_projection_matrix(1.0)
    cam.toggle_top_down()
    proj_orbit = cam.get_projection_matrix(1.0)
    # Ortho matrix has M[3,2] = 0; perspective has M[3,2] = -1
    assert proj_td[3, 2]    == pytest.approx( 0.0, abs=1e-5)
    assert proj_orbit[3, 2] == pytest.approx(-1.0, abs=1e-5)


def test_camera_move_forward_changes_target(cfg):
    cam = Camera(cfg)
    t0 = cam.target.copy()
    cam.move_forward(100.0)
    assert not np.allclose(cam.target, t0)


def test_camera_pan_drag_changes_target(cfg):
    cam = Camera(cfg)
    t0 = cam.target.copy()
    cam.pan_drag(50, 50)
    assert not np.allclose(cam.target, t0)


def test_set_follow_target_moves_orbit_target(cfg):
    """Orbit mode must track the vehicle so it stays centred on screen."""
    cam = Camera(cfg)
    assert cam.mode == "orbit"
    pos = np.array([100.0, 0.0, 200.0])
    cam.set_follow_target(pos, heading=0.0)
    assert np.allclose(cam.target, pos)


def test_set_follow_target_moves_target_in_all_modes(cfg):
    """Target follows vehicle regardless of which mode is active."""
    pos = np.array([50.0, 0.0, 75.0])
    for _ in range(3):   # orbit → follow_close → follow_aerial
        cam = Camera(cfg)
        # Advance to mode i without calling set_follow_target yet
        for _ in range(_):
            cam.cycle_follow_mode()
        cam.set_follow_target(pos, heading=1.0)
        assert np.allclose(cam.target, pos), f"target not updated in mode {cam.mode}"
