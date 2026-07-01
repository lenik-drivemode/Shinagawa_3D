"""GPU mesh wrapper: upload vertex/index buffers and issue draw calls (PRD §12.1)."""
import numpy as np
import moderngl


class GpuMesh:
    """One VBO + IBO → VAO for a single program and vertex layout.

    Args:
        ctx:        ModernGL context.
        prog:       Compiled ModernGL program.
        vertices:   float32 ndarray (N, D).
        indices:    uint32 ndarray (M,).
        fmt:        moderngl vertex format string, e.g. "3f" or "3f 3f".
        attr_names: attribute names matching the format components.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        prog: moderngl.Program,
        vertices: np.ndarray,
        indices: np.ndarray,
        fmt: str,
        *attr_names: str,
    ) -> None:
        self._vbo = ctx.buffer(vertices.astype(np.float32).tobytes())
        self._ibo = ctx.buffer(indices.astype(np.uint32).tobytes())
        self._vao = ctx.vertex_array(
            prog,
            [(self._vbo, fmt, *attr_names)],
            self._ibo,
        )
        self._n_indices = int(len(indices))

    def draw(self) -> None:
        self._vao.render(moderngl.TRIANGLES)

    def release(self) -> None:
        self._vao.release()
        self._vbo.release()
        self._ibo.release()
