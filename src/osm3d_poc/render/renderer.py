"""Main 3D renderer: loads assets, runs the ModernGL render loop (PRD §7.6, §12).

Usage (called from scripts/04_run_viewer.py):

    Renderer.configure(cfg, cli_args)
    mglw.run_window_config(Renderer)

or equivalently:

    run_viewer(cfg, cli_args)
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import moderngl
import moderngl_window as mglw

from .camera import Camera, translate, rotate_y
from .shaders import FLAT_VERT, FLAT_FRAG, BUILDING_VERT, BUILDING_FRAG
from .mesh import GpuMesh
from .marker import make_marker_mesh
from ..geo.preprocess_roads import polyline_to_strip
from ..geo.route_builder import load_route

log = logging.getLogger(__name__)


class Renderer(mglw.WindowConfig):
    """ModernGL-window WindowConfig subclass for the Shinagawa 3D map viewer."""

    title      = "Shinagawa 3D"
    gl_version = (3, 3)
    window_size = (1280, 720)
    resizable  = True

    # Class-level config injected before mglw.run_window_config
    _cfg       = None
    _cli_args  = None

    @classmethod
    def configure(cls, cfg: dict, cli_args=None) -> None:
        cls._cfg      = cfg
        cls._cli_args = cli_args

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        cfg = self._cfg

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        if self._cli_args and getattr(self._cli_args, "wireframe", False):
            self.ctx.wireframe = True

        # Shader programs
        self._flat_prog     = self.ctx.program(vertex_shader=FLAT_VERT,     fragment_shader=FLAT_FRAG)
        self._building_prog = self.ctx.program(vertex_shader=BUILDING_VERT, fragment_shader=BUILDING_FRAG)

        # GPU meshes (None if asset not available)
        self._road_mesh:    Optional[GpuMesh] = None
        self._route_mesh:   Optional[GpuMesh] = None
        self._building_mesh: Optional[GpuMesh] = None
        self._marker_mesh:  Optional[GpuMesh] = None

        # Route arrays (for marker position in Phase 6, animation in Phase 7)
        self._route_xyz: Optional[np.ndarray] = None
        self._route_s:   Optional[np.ndarray] = None

        self._load_assets(cfg)

        self._camera = Camera(cfg)
        self._camera.target = self._scene_center()

        # FPS tracking
        self._fps_frames = 0
        self._fps_acc    = 0.0

        bg = cfg["render"]["background_color"]
        self._bg = tuple(bg[:4] if len(bg) >= 4 else (*bg, 1.0))

        r = cfg["render"]
        self._road_color     = _vec4(r["road_color"])
        self._route_color    = _vec4(r["route_color"])
        self._building_color = _vec4(r["building_color"])
        self._marker_color   = _vec4(r["marker_color"])
        self._marker_y       = float(cfg["preprocess"]["marker_y_m"])

        log.info("Renderer ready (GL %s).", self.ctx.version_code)

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------

    def _load_assets(self, cfg: dict) -> None:
        out_dir = Path(cfg["preprocess"]["output_dir"])
        skip_bldg = self._cli_args and getattr(self._cli_args, "no_buildings", False)

        # Roads
        roads_path = out_dir / "roads_mesh.npz"
        if roads_path.exists():
            data = np.load(roads_path)
            self._road_mesh = GpuMesh(
                self.ctx, self._flat_prog,
                data["vertices"], data["indices"],
                "3f", "in_position",
            )
            log.info("Loaded road mesh: %d verts, %d tris.",
                     len(data["vertices"]), len(data["indices"]) // 3)
        else:
            log.warning("roads_mesh.npz not found — run 02_preprocess_assets.py first.")

        # Buildings
        if not skip_bldg:
            bldg_path = out_dir / "buildings_mesh.npz"
            if bldg_path.exists():
                data = np.load(bldg_path)
                self._building_mesh = GpuMesh(
                    self.ctx, self._building_prog,
                    data["vertices"], data["indices"],
                    "3f 3f", "in_position", "in_normal",
                )
                log.info("Loaded building mesh: %d verts, %d tris.",
                         len(data["vertices"]), len(data["indices"]) // 3)
            else:
                log.warning("buildings_mesh.npz not found — run 02_preprocess_assets.py first.")

        # Route + route strip + marker
        route_path = Path(cfg["route"]["output_file"])
        if route_path.exists():
            self._route_xyz, self._route_s = load_route(cfg)
            xz = self._route_xyz[:, [0, 2]]
            route_y = float(self._route_xyz[0, 1])
            route_verts, route_idxs = polyline_to_strip(xz, width=3.5, y=route_y)
            if len(route_verts):
                self._route_mesh = GpuMesh(
                    self.ctx, self._flat_prog,
                    route_verts, route_idxs,
                    "3f", "in_position",
                )
            log.info("Loaded route: %d points, %.1f m.",
                     len(self._route_xyz), float(self._route_s[-1]))

            # Marker geometry
            marker_verts, marker_idxs = make_marker_mesh(cfg)
            self._marker_mesh = GpuMesh(
                self.ctx, self._flat_prog,
                marker_verts, marker_idxs,
                "3f", "in_position",
            )
        else:
            log.warning("Route file not found — run 03_build_route.py first.")

    def _scene_center(self) -> np.ndarray:
        """Estimate the scene centre for initial camera target."""
        if self._route_xyz is not None:
            midpoint = (self._route_xyz.max(axis=0) + self._route_xyz.min(axis=0)) * 0.5
            return midpoint.astype(np.float64)
        return np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # Per-frame render
    # ------------------------------------------------------------------

    def render(self, time: float, frame_time: float) -> None:
        self.ctx.clear(*self._bg)

        w, h  = self.wnd.size
        aspect = w / h if h > 0 else 1.0

        proj = self._camera.get_projection_matrix(aspect)
        view = self._camera.get_view_matrix()
        vp   = proj @ view   # model = identity for static world geometry

        # --- Roads ---
        if self._road_mesh is not None:
            mvp = _mvp_bytes(vp)
            self._flat_prog["mvp"].write(mvp)
            self._flat_prog["color"].write(self._road_color)
            self._road_mesh.draw()

        # --- Route highlight ---
        if self._route_mesh is not None:
            self._flat_prog["mvp"].write(_mvp_bytes(vp))
            self._flat_prog["color"].write(self._route_color)
            self._route_mesh.draw()

        # --- Buildings ---
        if self._building_mesh is not None:
            self._building_prog["mvp"].write(_mvp_bytes(vp))
            self._building_prog["color"].write(self._building_color)
            self._building_mesh.draw()

        # --- Marker (Phase 6: static at route[0]; Phase 7: animated) ---
        if self._marker_mesh is not None and self._route_xyz is not None:
            pos     = self._route_xyz[0]
            heading = 0.0   # replaced by RoutePlayer.get_pose()[1] in Phase 7
            model   = translate(float(pos[0]), self._marker_y, float(pos[2])) @ rotate_y(heading)
            mvp     = _mvp_bytes(proj @ view @ model)
            self._flat_prog["mvp"].write(mvp)
            self._flat_prog["color"].write(self._marker_color)
            self._marker_mesh.draw()

        # --- FPS in window title ---
        self._fps_frames += 1
        self._fps_acc    += frame_time
        if self._fps_acc >= 0.5:
            fps = self._fps_frames / self._fps_acc
            mode_str = self._camera.mode
            self.wnd.title = f"Shinagawa 3D | {fps:.0f} FPS | {mode_str}"
            self._fps_frames = 0
            self._fps_acc    = 0.0

    # ------------------------------------------------------------------
    # Window / input events
    # ------------------------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        self.ctx.viewport = (0, 0, width, height)

    def key_event(self, key, action, modifiers) -> None:
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        if key == keys.ESCAPE:
            self.wnd.close()
        elif key == keys.T:
            self._camera.toggle_top_down()
            log.info("Camera mode: %s", self._camera.mode)
        elif key == keys.F:
            self._camera.cycle_follow_mode()
            log.info("Camera mode: %s", self._camera.mode)
        # W/S/A/D: camera movement
        move_speed = self._camera.distance * 0.05
        if key == keys.W:
            self._camera.move_forward( move_speed)
        elif key == keys.S:
            self._camera.move_forward(-move_speed)
        elif key == keys.A:
            self._camera.move_strafe(-move_speed)
        elif key == keys.D:
            self._camera.move_strafe( move_speed)

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
        if self.wnd.mouse_states.left:
            self._camera.orbit_drag(dx, dy)
        elif self.wnd.mouse_states.right:
            self._camera.pan_drag(dx, dy)

    def mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
        self._camera.zoom(y_offset)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mvp_bytes(mvp: np.ndarray) -> bytes:
    """Transpose row-major numpy matrix to column-major bytes for GLSL uniform."""
    return mvp.T.astype(np.float32).tobytes()


def _vec4(color) -> bytes:
    """Convert a config color list to float32 bytes for a vec4 uniform."""
    c = list(color)
    if len(c) == 3:
        c.append(1.0)
    return np.array(c[:4], dtype=np.float32).tobytes()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_viewer(cfg: dict, cli_args=None) -> None:
    """Configure and launch the 3D viewer window."""
    Renderer.configure(cfg, cli_args)
    mglw.run_window_config(Renderer)
