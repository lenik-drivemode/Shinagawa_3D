"""Camera: orbit / follow-close / follow-aerial / top-down modes (PRD §7.7, §12.4)."""
import numpy as np

_PITCH_MIN_DEG =    5.0
_PITCH_MAX_DEG =   89.0
_DIST_MIN_M    =   20.0
_DIST_MAX_M    = 15000.0
_MODES         = ("orbit", "follow_close", "follow_aerial")


# ---------------------------------------------------------------------------
# Matrix math (row-major; transposed before writing to GLSL column-major uniform)
# ---------------------------------------------------------------------------

def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Standard look-at view matrix (row-major float32)."""
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array([
        [ s[0],  s[1],  s[2], -np.dot(s, eye)],
        [ u[0],  u[1],  u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [ 0.0,   0.0,   0.0,   1.0           ],
    ], dtype=np.float32)


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Standard perspective projection matrix (row-major float32)."""
    f = 1.0 / np.tan(np.radians(fovy_deg) * 0.5)
    return np.array([
        [f / aspect, 0.0,  0.0,                           0.0                          ],
        [0.0,        f,    0.0,                           0.0                          ],
        [0.0,        0.0, -(far + near) / (far - near),  -2.0 * far * near / (far - near)],
        [0.0,        0.0, -1.0,                           0.0                          ],
    ], dtype=np.float32)


def ortho(left: float, right: float, bottom: float, top: float,
          near: float, far: float) -> np.ndarray:
    """Orthographic projection matrix (row-major float32)."""
    return np.array([
        [2 / (right - left), 0,                  0,               -(right + left) / (right - left)],
        [0,                  2 / (top - bottom),  0,               -(top + bottom) / (top - bottom)],
        [0,                  0,                  -2 / (far - near), -(far + near)  / (far - near)  ],
        [0,                  0,                   0,                1.0                             ],
    ], dtype=np.float32)


def translate(x: float, y: float, z: float) -> np.ndarray:
    """Translation matrix (row-major float32)."""
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def rotate_y(angle_rad: float) -> np.ndarray:
    """Rotation around Y axis (row-major float32)."""
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    """Interactive camera supporting orbit, follow, and top-down modes.

    In orbit mode:
      - yaw (degrees): horizontal rotation around target; 0 = looking from south
      - pitch (degrees): elevation above horizontal plane; clamped to [5, 89]
      - distance (metres): radius from target

    Follow modes are enabled by set_follow_target() (called each frame by RoutePlayer
    in Phase 7). Until set_follow_target is called, follow modes show origin.

    Pressing F cycles through: orbit → follow_close → follow_aerial.
    Pressing T toggles top-down orthographic view independently.
    """

    def __init__(self, cfg: dict) -> None:
        r = cfg["render"]
        self.distance = float(r["initial_camera_height_m"])
        self.pitch    = float(r["initial_pitch_deg"])
        self.yaw      = float(r["initial_yaw_deg"])
        self.target   = np.zeros(3, dtype=np.float64)

        self._mode_idx = 0          # index into _MODES
        self._top_down = False

        self._fov_deg = 60.0
        self._near    = 1.0
        self._far     = 20000.0

        self._follow_back_m  = float(r["follow_close_back_m"])
        self._follow_up_m    = float(r["follow_close_up_m"])
        self._aerial_up_m    = float(r["follow_aerial_up_m"])
        self._aerial_back_m  = float(r.get("follow_aerial_back_m", 0.0))

        self._follow_pos     = np.zeros(3, dtype=np.float64)
        self._follow_heading = 0.0

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "top_down" if self._top_down else _MODES[self._mode_idx]

    def cycle_follow_mode(self) -> None:
        """F key: orbit → follow_close → follow_aerial → orbit."""
        self._top_down = False
        self._mode_idx = (self._mode_idx + 1) % len(_MODES)

    def toggle_top_down(self) -> None:
        """T key: toggle top-down orthographic view."""
        self._top_down = not self._top_down

    def set_follow_target(self, pos: np.ndarray, heading: float) -> None:
        """Phase 7 hook: update marker position and heading each frame."""
        self._follow_pos     = np.asarray(pos, dtype=np.float64)
        self._follow_heading = float(heading)
        self.target = self._follow_pos.copy()   # all modes keep vehicle centred

    # ------------------------------------------------------------------
    # Mouse / keyboard input
    # ------------------------------------------------------------------

    def orbit_drag(self, dx: float, dy: float, sensitivity: float = 0.4) -> None:
        """Left-drag: rotate yaw/pitch around target."""
        self.yaw    = self.yaw + dx * sensitivity
        new_pitch   = self.pitch - dy * sensitivity   # drag up → pitch increases
        self.pitch  = float(np.clip(new_pitch, _PITCH_MIN_DEG, _PITCH_MAX_DEG))

    def pan_drag(self, dx: float, dy: float, sensitivity: float = None) -> None:
        """Right-drag: translate target in the camera horizontal plane."""
        if sensitivity is None:
            sensitivity = self.distance * 0.001
        yaw_r  = np.radians(self.yaw)
        # camera right direction in world XZ (horizontal)
        right  = np.array([ np.cos(yaw_r), 0.0, -np.sin(yaw_r)])
        # camera forward in world XZ
        fwd_h  = np.array([-np.sin(yaw_r), 0.0, -np.cos(yaw_r)])
        self.target = self.target - dx * sensitivity * right + dy * sensitivity * fwd_h

    def zoom(self, delta: float, factor: float = 0.1) -> None:
        """Scroll wheel: zoom in/out."""
        self.distance = float(
            np.clip(self.distance * (1.0 - delta * factor), _DIST_MIN_M, _DIST_MAX_M)
        )

    def move_forward(self, distance_m: float) -> None:
        """W/S: translate target along horizontal forward direction."""
        yaw_r = np.radians(self.yaw)
        self.target += distance_m * np.array([-np.sin(yaw_r), 0.0, -np.cos(yaw_r)])

    def move_strafe(self, distance_m: float) -> None:
        """A/D: translate target along camera right direction."""
        yaw_r = np.radians(self.yaw)
        self.target += distance_m * np.array([np.cos(yaw_r), 0.0, -np.sin(yaw_r)])

    # ------------------------------------------------------------------
    # Matrix computation
    # ------------------------------------------------------------------

    def orbit_eye(self) -> np.ndarray:
        """Compute eye position for orbit mode (public for testing)."""
        yaw_r   = np.radians(self.yaw)
        pitch_r = np.radians(self.pitch)
        return self.target + self.distance * np.array([
            np.sin(yaw_r)  * np.cos(pitch_r),
            np.sin(pitch_r),
            np.cos(yaw_r)  * np.cos(pitch_r),
        ], dtype=np.float64)

    def get_view_matrix(self) -> np.ndarray:
        """Return 4×4 row-major float32 view matrix for the current mode."""
        up = np.array([0.0, 1.0, 0.0])

        if self._top_down:
            eye = self.target + np.array([0.0, max(self.distance, 10.0), 0.0])
            return look_at(eye, self.target, np.array([0.0, 0.0, -1.0]))

        mode = _MODES[self._mode_idx]
        if mode == "orbit":
            return look_at(self.orbit_eye(), self.target, up)

        # Follow modes
        pos = self._follow_pos
        h   = self._follow_heading

        if mode == "follow_close":
            back  = np.array([-np.sin(h), 0.0, -np.cos(h)]) * self._follow_back_m
            above = np.array([0.0, self._follow_up_m, 0.0])
            eye   = pos + back + above
            # look toward the marker position
            return look_at(eye, pos, up)
        else:   # follow_aerial
            back = np.array([-np.sin(h), 0.0, -np.cos(h)]) * self._aerial_back_m
            eye  = pos + back + np.array([0.0, self._aerial_up_m, 0.0])
            fwd  = np.array([np.sin(h), 0.0, np.cos(h)])
            return look_at(eye, pos, fwd)

    def get_projection_matrix(self, aspect: float) -> np.ndarray:
        """Return 4×4 row-major float32 projection matrix."""
        if self._top_down:
            h = self.distance * 0.5
            w = h * aspect
            return ortho(-w, w, -h, h, -10000.0, 10000.0)
        return perspective(self._fov_deg, aspect, self._near, self._far)
