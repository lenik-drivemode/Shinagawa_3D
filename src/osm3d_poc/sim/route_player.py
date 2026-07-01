"""Vehicle simulation: advance a marker along a pre-built route (PRD §7.5, §12.6).

RoutePlayer consumes the float32 xyz array and float32 cumulative-s array
produced by route_builder.load_route(), and exposes:
  - update(dt): advance position based on elapsed time and current speed
  - get_pose(): return (position_xyz float32[3], heading_radians float)
  - pause/resume/reset/speed controls

Position interpolation uses binary search on the cumulative-s array, so
update() and get_pose() are O(log N).

Heading convention (matches renderer's rotate_y semantics):
  heading = atan2(dx, dz)
  - 0      → facing +Z (north)
  - +π/2   → facing +X (east)
  - ±π     → facing −Z (south)
"""
from typing import Tuple

import numpy as np


class RoutePlayer:
    """Simulates a vehicle moving along a dense route at configurable speed."""

    MIN_SPEED_MPS: float = 5.0  / 3.6    # 5 km/h
    MAX_SPEED_MPS: float = 200.0 / 3.6   # 200 km/h

    def __init__(
        self,
        route_xyz: np.ndarray,
        route_s:   np.ndarray,
        speed_mps: float,
    ) -> None:
        """Initialise the player.

        Args:
            route_xyz:  (N, 3) float32 world-space positions along the route.
            route_s:    (N,)   float32/64 cumulative arc-lengths (s[0] == 0).
            speed_mps:  Initial speed in metres per second (clamped to [5, 200] km/h).
        """
        self._xyz   = np.asarray(route_xyz, dtype=np.float32)
        self._s     = np.asarray(route_s,   dtype=np.float64)
        self._total = float(self._s[-1])
        self._dist  = 0.0
        self._speed = float(np.clip(speed_mps, self.MIN_SPEED_MPS, self.MAX_SPEED_MPS))
        self._paused = False

    # ------------------------------------------------------------------
    # Simulation tick
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Advance the route position by dt seconds at current speed.

        Wraps around silently when the end is reached (FR-SIM-004).
        No-op when paused or route is degenerate.
        """
        if self._paused or self._total <= 0.0:
            return
        self._dist = (self._dist + self._speed * float(dt)) % self._total

    # ------------------------------------------------------------------
    # Pose query
    # ------------------------------------------------------------------

    def get_pose(self) -> Tuple[np.ndarray, float]:
        """Return the current (position_xyz float32[3], heading_radians float).

        Position is linearly interpolated between adjacent route samples.
        Heading is the tangent direction of the current segment.
        """
        d   = self._dist % self._total if self._total > 0.0 else 0.0
        idx = int(np.searchsorted(self._s, d, side="right")) - 1
        idx = max(0, min(idx, len(self._s) - 2))

        s0, s1 = float(self._s[idx]), float(self._s[idx + 1])
        t      = (d - s0) / (s1 - s0) if s1 > s0 else 0.0

        pos     = self._xyz[idx] * (1.0 - t) + self._xyz[idx + 1] * t
        dx      = float(self._xyz[idx + 1][0] - self._xyz[idx][0])
        dz      = float(self._xyz[idx + 1][2] - self._xyz[idx][2])
        heading = float(np.arctan2(dx, dz))   # 0=north, π/2=east

        return pos.astype(np.float32), heading

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    def toggle_pause(self) -> None:
        self._paused = not self._paused

    def reset(self) -> None:
        """Return to the route origin."""
        self._dist = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def dist_m(self) -> float:
        return self._dist

    @property
    def total_m(self) -> float:
        return self._total

    @property
    def speed_mps(self) -> float:
        return self._speed

    @speed_mps.setter
    def speed_mps(self, v: float) -> None:
        self._speed = float(np.clip(v, self.MIN_SPEED_MPS, self.MAX_SPEED_MPS))

    @property
    def speed_kmh(self) -> float:
        return self._speed * 3.6

    @speed_kmh.setter
    def speed_kmh(self, v: float) -> None:
        self.speed_mps = float(v) / 3.6
