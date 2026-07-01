import numpy as np
import moderngl
import moderngl_window as mglw

_VERT = """
#version 330 core
in vec3 in_position;
in vec3 in_color;
uniform mat4 mvp;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
"""

_FRAG = """
#version 330 core
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""

# 24 vertices (4 per face × 6 faces): position xyz + color rgb
_VERTICES = np.array([
    # front  (blue)
    -1, -1,  1,  0.20, 0.40, 0.80,
     1, -1,  1,  0.20, 0.40, 0.80,
     1,  1,  1,  0.20, 0.40, 0.80,
    -1,  1,  1,  0.20, 0.40, 0.80,
    # back   (red)
     1, -1, -1,  0.80, 0.20, 0.20,
    -1, -1, -1,  0.80, 0.20, 0.20,
    -1,  1, -1,  0.80, 0.20, 0.20,
     1,  1, -1,  0.80, 0.20, 0.20,
    # left   (green)
    -1, -1, -1,  0.20, 0.70, 0.30,
    -1, -1,  1,  0.20, 0.70, 0.30,
    -1,  1,  1,  0.20, 0.70, 0.30,
    -1,  1, -1,  0.20, 0.70, 0.30,
    # right  (yellow)
     1, -1,  1,  0.90, 0.80, 0.10,
     1, -1, -1,  0.90, 0.80, 0.10,
     1,  1, -1,  0.90, 0.80, 0.10,
     1,  1,  1,  0.90, 0.80, 0.10,
    # top    (cyan)
    -1,  1,  1,  0.10, 0.80, 0.80,
     1,  1,  1,  0.10, 0.80, 0.80,
     1,  1, -1,  0.10, 0.80, 0.80,
    -1,  1, -1,  0.10, 0.80, 0.80,
    # bottom (magenta)
    -1, -1, -1,  0.80, 0.20, 0.70,
     1, -1, -1,  0.80, 0.20, 0.70,
     1, -1,  1,  0.80, 0.20, 0.70,
    -1, -1,  1,  0.80, 0.20, 0.70,
], dtype=np.float32)

_INDICES = np.array([
     0,  1,  2,   2,  3,  0,   # front
     4,  5,  6,   6,  7,  4,   # back
     8,  9, 10,  10, 11,  8,   # left
    12, 13, 14,  14, 15, 12,   # right
    16, 17, 18,  18, 19, 16,   # top
    20, 21, 22,  22, 23, 20,   # bottom
], dtype=np.uint32)


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fovy_deg) * 0.5)
    return np.array([
        [f / aspect, 0,  0,                         0                        ],
        [0,          f,  0,                         0                        ],
        [0,          0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0,          0, -1,                         0                        ],
    ], dtype=np.float32)


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return np.array([
        [ s[0],  s[1],  s[2], -np.dot(s, eye)],
        [ u[0],  u[1],  u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [ 0,     0,     0,     1              ],
    ], dtype=np.float32)


def _rotate_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float32)


class _DemoCubeWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Shinagawa 3D — Phase 0 Demo Cube"
    window_size = (1280, 720)
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        vbo = self.ctx.buffer(_VERTICES.tobytes())
        ibo = self.ctx.buffer(_INDICES.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(vbo, "3f 3f", "in_position", "in_color")],
            ibo,
        )
        self.ctx.enable(moderngl.DEPTH_TEST)
        self._fps_frames = 0
        self._fps_acc = 0.0

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.08, 0.09, 0.10, 1.0)

        w, h = self.wnd.size
        proj  = _perspective(45.0, w / h, 0.1, 100.0)
        eye   = np.array([3.0, 2.0, 5.0], dtype=np.float32)
        view  = _look_at(eye, np.zeros(3, dtype=np.float32), np.array([0.0, 1.0, 0.0]))
        model = _rotate_y(time)
        mvp   = proj @ view @ model

        # Transpose row-major → column-major before writing to GLSL uniform
        self.prog["mvp"].write(mvp.T.astype("f4").tobytes())
        self.vao.render(moderngl.TRIANGLES)

        self._fps_frames += 1
        self._fps_acc += frame_time
        if self._fps_acc >= 0.5:
            fps = self._fps_frames / self._fps_acc
            self.wnd.title = f"Shinagawa 3D — Demo Cube | {fps:.0f} FPS"
            self._fps_frames = 0
            self._fps_acc = 0.0

    def on_key_event(self, key, action, modifiers):
        if key == self.wnd.keys.ESCAPE and action == self.wnd.keys.ACTION_PRESS:
            self.wnd.close()


def run_demo_cube() -> None:
    mglw.run_window_config(_DemoCubeWindow, args=["--window", "pyglet"])
