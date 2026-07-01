from pathlib import Path
import pytest
from osm3d_poc.config import load_config

_CFG_PATH = Path(__file__).parent.parent / "config" / "shinagawa_poc.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(_CFG_PATH)


def test_config_loads(cfg):
    assert isinstance(cfg, dict)


def test_required_top_level_sections(cfg):
    for section in ("area", "projection", "osm", "preprocess", "route", "render", "logging"):
        assert section in cfg, f"Missing top-level config section: {section}"


def test_bbox_is_valid(cfg):
    bbox = cfg["area"]["bbox"]
    assert bbox["north"] > bbox["south"], "bbox north must be greater than south"
    assert bbox["east"] > bbox["west"], "bbox east must be greater than west"


def test_debug_bbox_is_valid(cfg):
    dbbox = cfg["area"]["debug_bbox"]
    assert dbbox["north"] > dbbox["south"]
    assert dbbox["east"] > dbbox["west"]


def test_projection_epsgs(cfg):
    proj = cfg["projection"]
    assert proj["source_epsg"] == 4326
    assert proj["target_epsg"] == 32654


def test_preprocess_y_layers_are_ordered(cfg):
    pre = cfg["preprocess"]
    assert pre["road_y_m"] < pre["route_y_m"] < pre["marker_y_m"], (
        "Layer Y values must be ordered: road < route < marker"
    )


def test_preprocess_heights_are_positive(cfg):
    pre = cfg["preprocess"]
    assert pre["default_building_height_m"] > 0
    assert pre["floor_height_m"] > 0
    assert pre["max_building_height_m"] > pre["default_building_height_m"]


def test_render_window_size(cfg):
    r = cfg["render"]
    assert r["window_width"] >= 1280
    assert r["window_height"] >= 720


def test_render_follow_camera_params_present(cfg):
    r = cfg["render"]
    assert "follow_close_back_m" in r
    assert "follow_close_up_m" in r
    assert "follow_aerial_up_m" in r
    assert r["follow_aerial_up_m"] > r["follow_close_up_m"]


def test_route_speed_bounds(cfg):
    ro = cfg["route"]
    assert ro["speed_min_kmh"] > 0
    assert ro["speed_max_kmh"] > ro["default_speed_kmh"] >= ro["speed_min_kmh"]


def test_route_waypoints_form_loop(cfg):
    wps = cfg["route"]["waypoints"]
    assert len(wps) >= 2
    first, last = wps[0], wps[-1]
    assert abs(first["lat"] - last["lat"]) < 0.001, "Route should start and end near the same point"
    assert abs(first["lon"] - last["lon"]) < 0.001, "Route should start and end near the same point"
