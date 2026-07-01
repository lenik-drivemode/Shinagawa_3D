"""Tests for RoutePlayer vehicle simulation (PRD §7.5, §12.6, FR-SIM-000…007)."""
import numpy as np
import pytest

from osm3d_poc.sim.route_player import RoutePlayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_route():
    """3-point L-shaped route: (0,0,0)→(100,0,0)→(100,0,100), total 200 m."""
    xyz = np.array([
        [  0.0, 0.0,   0.0],
        [100.0, 0.0,   0.0],
        [100.0, 0.0, 100.0],
    ], dtype=np.float32)
    s = np.array([0.0, 100.0, 200.0], dtype=np.float32)
    return xyz, s


@pytest.fixture
def player(simple_route):
    xyz, s = simple_route
    return RoutePlayer(xyz, s, speed_mps=10.0)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_initial_dist_is_zero(player):
    assert player.dist_m == 0.0


def test_total_m(player):
    assert player.total_m == pytest.approx(200.0)


def test_not_paused_initially(player):
    assert player.paused is False


def test_update_advances_dist(player):
    player.update(1.0)
    assert player.dist_m > 0.0


def test_update_distance_proportional_to_speed(simple_route):
    xyz, s = simple_route
    slow = RoutePlayer(xyz, s, speed_mps= 5.0)
    fast = RoutePlayer(xyz, s, speed_mps=20.0)
    slow.update(1.0)
    fast.update(1.0)
    assert fast.dist_m == pytest.approx(slow.dist_m * 4, rel=1e-6)


def test_update_multiple_frames_accumulates(player):
    player.update(0.1)
    player.update(0.1)
    player.update(0.1)
    assert player.dist_m == pytest.approx(3.0, rel=1e-5)   # 10 m/s × 0.3 s


# ---------------------------------------------------------------------------
# Wrap-around (FR-SIM-004)
# ---------------------------------------------------------------------------

def test_wrap_around_keeps_dist_in_range(simple_route):
    xyz, s = simple_route
    p = RoutePlayer(xyz, s, speed_mps=10.0)
    # Jump almost to the end
    p._dist = 195.0
    p.update(1.0)   # 10 m → should wrap (195 + 10 > 200)
    assert 0.0 <= p.dist_m < 200.0


def test_wrap_around_position_near_start(simple_route):
    xyz, s = simple_route
    p = RoutePlayer(xyz, s, speed_mps=10.0)
    p._dist = 198.0
    p.update(1.0)   # wraps → dist ≈ 8 m into first segment
    pos, _ = p.get_pose()
    assert pos[0] == pytest.approx(8.0, abs=0.5)
    assert pos[2] == pytest.approx(0.0, abs=0.5)


# ---------------------------------------------------------------------------
# Pause / resume (FR-SIM-005 implied; Space key)
# ---------------------------------------------------------------------------

def test_paused_does_not_advance(player):
    player.set_paused(True)
    d_before = player.dist_m
    player.update(1.0)
    assert player.dist_m == d_before


def test_set_paused_false_resumes(player):
    player.set_paused(True)
    player.set_paused(False)
    d_before = player.dist_m
    player.update(1.0)
    assert player.dist_m > d_before


def test_toggle_pause_cycles(player):
    assert player.paused is False
    player.toggle_pause()
    assert player.paused is True
    player.toggle_pause()
    assert player.paused is False


# ---------------------------------------------------------------------------
# Reset (R key)
# ---------------------------------------------------------------------------

def test_reset_returns_dist_to_zero(player):
    player.update(5.0)
    player.reset()
    assert player.dist_m == 0.0


def test_reset_position_at_route_origin(simple_route):
    xyz, s = simple_route
    p = RoutePlayer(xyz, s, speed_mps=10.0)
    p._dist = 150.0
    p.reset()
    pos, _ = p.get_pose()
    assert np.allclose(pos, xyz[0], atol=1e-4)


# ---------------------------------------------------------------------------
# Position interpolation (FR-SIM-002)
# ---------------------------------------------------------------------------

def test_position_at_dist_zero(simple_route):
    xyz, s = simple_route
    p = RoutePlayer(xyz, s, speed_mps=10.0)
    pos, _ = p.get_pose()
    assert np.allclose(pos, xyz[0], atol=1e-4)


def test_position_at_midpoint_first_segment(simple_route):
    xyz, s = simple_route
    p = RoutePlayer(xyz, s, speed_mps=10.0)
    p._dist = 50.0
    pos, _ = p.get_pose()
    assert pos[0] == pytest.approx(50.0, abs=1e-4)
    assert pos[2] == pytest.approx( 0.0, abs=1e-4)


def test_position_at_segment_boundary(simple_route):
    xyz, s = simple_route
    p = RoutePlayer(xyz, s, speed_mps=10.0)
    p._dist = 100.0
    pos, _ = p.get_pose()
    assert pos[0] == pytest.approx(100.0, abs=1e-4)
    assert pos[2] == pytest.approx(  0.0, abs=1e-4)


def test_position_midpoint_second_segment(simple_route):
    xyz, s = simple_route
    p = RoutePlayer(xyz, s, speed_mps=10.0)
    p._dist = 150.0
    pos, _ = p.get_pose()
    assert pos[0] == pytest.approx(100.0, abs=1e-4)
    assert pos[2] == pytest.approx( 50.0, abs=1e-4)


def test_position_dtype(player):
    pos, _ = player.get_pose()
    assert pos.dtype == np.float32


def test_no_nan_position(player):
    pos, heading = player.get_pose()
    assert not np.any(np.isnan(pos))
    assert not np.isnan(heading)


# ---------------------------------------------------------------------------
# Heading (FR-SIM-003): atan2(dx, dz) → 0=north, π/2=east
# ---------------------------------------------------------------------------

def test_heading_east_first_segment(simple_route):
    """Segment (0,0,0)→(100,0,0) runs east → heading ≈ π/2."""
    xyz, s = simple_route
    p = RoutePlayer(xyz, s, speed_mps=10.0)
    p._dist = 50.0
    _, heading = p.get_pose()
    assert heading == pytest.approx(np.pi / 2, abs=1e-4)


def test_heading_north_second_segment(simple_route):
    """Segment (100,0,0)→(100,0,100) runs north → heading ≈ 0."""
    xyz, s = simple_route
    p = RoutePlayer(xyz, s, speed_mps=10.0)
    p._dist = 150.0
    _, heading = p.get_pose()
    assert heading == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Speed controls (FR-SIM-005)
# ---------------------------------------------------------------------------

def test_speed_kmh_setter(player):
    player.speed_kmh = 40.0
    assert player.speed_mps == pytest.approx(40.0 / 3.6, rel=1e-5)


def test_speed_mps_setter(player):
    player.speed_mps = 10.0
    assert player.speed_kmh == pytest.approx(36.0, rel=1e-5)


def test_speed_clamp_min(player):
    player.speed_kmh = 1.0    # below 5 km/h
    assert player.speed_kmh >= 5.0


def test_speed_clamp_max(player):
    player.speed_kmh = 999.0  # above 200 km/h
    assert player.speed_kmh <= 200.0


def test_speed_decrement_stays_clamped(player):
    player.speed_kmh = 6.0
    player.speed_kmh -= 10    # would go to -4 → clamped to 5
    assert player.speed_kmh >= 5.0


def test_speed_increment_stays_clamped(player):
    player.speed_kmh = 198.0
    player.speed_kmh += 10    # would go to 208 → clamped to 200
    assert player.speed_kmh <= 200.0
